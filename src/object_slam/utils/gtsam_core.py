from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L

from rclpy.logging import get_logger


### DEFAULT PARAMETERS
_DEFAULTS: Dict[str, object] = {
    "prior_floor_std": (1e-3, 1e-3, 1e-3),
    "robust_loss": None,  # or "Huber", "Cauchy"
    # std for keyframe factor in meters / radians
    "odom_std_base": (0.05, 0.05, 0.5 * math.pi / 180.0),
    "odom_std_per_meter": (0.01, 0.01, 0.2 * math.pi / 180.0),
    "odom_std_per_second": (0.005, 0.005, 0.1 * math.pi / 180.0),
    # stf for measurement factor meters / radians
    "meas_std_bearing": 3.0 * math.pi / 180.0, 
    "meas_std_range": 0.5,
    # iSAM2 params
    "relinearize_thresh": 0.1,
    "relinearize_skip": 1,
}

### HELPER FUNCTIONS
def _wrap_angle(theta: float) -> float:
    """Normalize any angle to (-pi, pi]."""
    t = (theta + math.pi) % (2.0 * math.pi)
    if t <= 0.0:
        t += 2.0 * math.pi
    return t - math.pi

def _make_isam2(relin_thresh: float, relin_skip: int) -> gtsam.ISAM2:
    """Create ISAM2 with API-compat for different gtsam versions."""
    params = gtsam.ISAM2Params()
    # threshold
    try:
        params.setRelinearizeThreshold(float(relin_thresh))
    except AttributeError:
        # older wheels expose attributes directly
        params.relinearizeThreshold = float(relin_thresh)
    # skip
    try:
        params.setRelinearizeSkip(int(relin_skip))
    except AttributeError:
        params.relinearizeSkip = int(relin_skip)
    return gtsam.ISAM2(params)

### CORE CLASS
class BackendCore:
    def __init__(self, params: Optional[Dict[str, object]] = None) -> None:
        # Params
        self.params: Dict[str, object] = dict(_DEFAULTS)
        if params:
            self.params.update(params)

        # Bookkeeping
        self._initialized: bool = False
        self._num_nodes: int = 0
        self._stamps_ns: List[int] = []
        self._latest_estimate: Optional[Tuple[float, float, float]] = None  # (x, y, yaw)
        self._last_guess_pose2: Optional[gtsam.Pose2] = None

        # Pending graph/values
        self._graph_pending = gtsam.NonlinearFactorGraph()
        self._values_pending = gtsam.Values()

        # iSAM2
        self._isam2 = _make_isam2(
            float(self.params["relinearize_thresh"]),
            int(self.params["relinearize_skip"]),
        )

        # Current estimate cache
        self._estimate_values = gtsam.Values()

        self.log = get_logger("gtsam_core")

    def reset(self) -> None:
        """Clear graph and all state (keeps params)."""
        self._initialized = False
        self._num_nodes = 0
        self._stamps_ns.clear()
        self._latest_estimate = None
        self._last_guess_pose2 = None

        self._graph_pending = gtsam.NonlinearFactorGraph()
        self._values_pending = gtsam.Values()
        self._isam2 = _make_isam2(
            float(self.params["relinearize_thresh"]),
            int(self.params["relinearize_skip"]),
        )
        self._estimate_values = gtsam.Values()

    ### INTERNAL HELPER FUNCTION
    def _ensure_cov_spd_with_floor(self, cov, floor_std):
        """Add a tiny diagonal floor (std^2) to keep covariance Symmetric Positive-(Semi)definitive."""
        cov = np.array(cov, dtype=float).reshape(3, 3)
        floor_var = np.square(np.asarray(floor_std, dtype=float))
        cov = cov.copy()
        cov[0, 0] += floor_var[0]
        cov[1, 1] += floor_var[1]
        cov[2, 2] += floor_var[2]
        return cov

    def _make_gaussian_from_cov(self, cov: np.ndarray) -> gtsam.noiseModel.Gaussian:
        return gtsam.noiseModel.Gaussian.Covariance(cov)

    def _make_odom_std(self, dx: float, dy: float, dyaw: float, dt_ms: int) -> np.ndarray:
        """Diagonal stds using base + per_meter*dist + per_second*dt."""
        base = np.asarray(self.params["odom_std_base"], dtype=float)
        per_m = np.asarray(self.params["odom_std_per_meter"], dtype=float)
        per_s = np.asarray(self.params["odom_std_per_second"], dtype=float)
        dist = math.hypot(dx, dy)
        dt_s = max(0.0, float(dt_ms) / 1000.0)
        std = base + per_m * dist + per_s * dt_s
        return np.maximum(std, 1e-9)

    def _apply_robust_loss(self, gaussian: gtsam.noiseModel.Base) -> gtsam.noiseModel.Base:
        robust = self.params.get("robust_loss", None)
        if robust is None:
            return gaussian
        if robust == "Huber":
            loss = gtsam.noiseModel.mEstimator.Huber.Create(k=1.345)
        elif robust == "Cauchy":
            loss = gtsam.noiseModel.mEstimator.Cauchy.Create(k=1.0)
        else:
            return gaussian
        return gtsam.noiseModel.Robust.Create(loss, gaussian)

    ### INPUT FUNCTION
    def add_anchor(self, stamp_ns: int, x: float, y: float, yaw: float, cov3x3):
        """Add the initial anchor (prior) node X(0)."""
        yaw = _wrap_angle(yaw)
        cov3x3 = self._ensure_cov_spd_with_floor(cov3x3, self.params["prior_floor_std"])
        noise = self._make_gaussian_from_cov(cov3x3)

        if not self._initialized:
            key = X(0)
            pose = gtsam.Pose2(x, y, yaw)
            self._graph_pending.add(gtsam.PriorFactorPose2(key, pose, noise))
            self._values_pending.insert(key, pose)

            self._stamps_ns.append(int(stamp_ns))
            self._num_nodes = 1
            self._initialized = True
            self._last_guess_pose2 = pose

    def add_keyframe(self, stamp_ns: int, dx: float, dy: float, dyaw: float, dt_ms: int, cov_hint3x3: Optional[np.ndarray] = None):
        """
        Add body-frame odometry delta connecting x_k → x_{k+1}.
        If cov_hint3x3 is None, a diagonal covariance is generated from dx, dy, dyaw, dt.
        """
        if not self._initialized:
            raise RuntimeError("add_keyframe() called before any anchor.")

        dx = float(dx)
        dy = float(dy)
        dyaw = _wrap_angle(float(dyaw))

        from_key = X(self._num_nodes - 1)
        to_key = X(self._num_nodes)

        meas = gtsam.Pose2(dx, dy, dyaw)

        # Noise model
        if cov_hint3x3 is not None:
            cov = self._ensure_cov_spd_with_floor(cov_hint3x3, self.params["prior_floor_std"])
            gaussian = self._make_gaussian_from_cov(cov)
        else: # <- currently always None as cov from DRIFT has not been defined
            std = self._make_odom_std(dx, dy, dyaw, dt_ms) # Generating std myself using dx, dy, dyaw, dt
            gaussian = gtsam.noiseModel.Diagonal.Sigmas(std)
        noise = self._apply_robust_loss(gaussian)

        # Factor
        self._graph_pending.add(gtsam.BetweenFactorPose2(from_key, to_key, meas, noise))

        # Initial guess for new node
        prev_guess = self._last_guess_pose2 if self._last_guess_pose2 is not None else gtsam.Pose2(0.0, 0.0, 0.0)
        new_guess = prev_guess.compose(meas)
        self._values_pending.insert(to_key, new_guess)
        self._last_guess_pose2 = new_guess

        # Bookkeeping
        self._stamps_ns.append(int(stamp_ns))
        self._num_nodes += 1
        
    def add_measurement(self, landmark_id: int, bearing_rad: float, range_m: float, cov2x2: Optional[np.ndarray] = None):
        """
        Add a 2D bearing+range observation:
        - bearing_rad: angle of landmark in the *body frame* (radians)
        - range_m: distance from body origin to landmark (meters)
        By default, attaches to the *latest* pose node X(self._num_nodes-1).
        """
        if not self._initialized or self._num_nodes == 0:
            raise RuntimeError("add_measurement() called before any anchor/keyframe.")

        # Choose which pose node to attach this measurement to
        k = int(self._num_nodes - 1)

        ### ROOM FOR IMPROVEMENT: data association with closest timestamp
        x_key = X(k)
        l_key = L(int(landmark_id))

        # Noise model: either provided covariance (2x2) or diagonal from defaults
        if cov2x2 is not None:
            cov = np.array(cov2x2, dtype=float).reshape(2, 2).copy()
            # tiny PD floor
            eps = 1e-9
            cov[0, 0] += eps
            cov[1, 1] += eps
            gaussian = gtsam.noiseModel.Gaussian.Covariance(cov)
        else:
            sig_b = float(self.params["meas_std_bearing"])
            sig_r = float(self.params["meas_std_range"])
            gaussian = gtsam.noiseModel.Diagonal.Sigmas(np.array([sig_b, sig_r], dtype=float))
        noise = self._apply_robust_loss(gaussian)

        # Add the factor (bearing in Rot2, range as float)
        self._graph_pending.add(
            gtsam.BearingRangeFactor2D(x_key, l_key, gtsam.Rot2(float(bearing_rad)), float(range_m), noise)
        )

        # Initialize landmark if we haven't seen it before
        if not (self._estimate_values.exists(l_key) or self._values_pending.exists(l_key)):
            # pick a pose guess for X(k): prefer optimized, then pending, then last guess
            if self._estimate_values.exists(x_key):
                p: gtsam.Pose2 = self._estimate_values.atPose2(x_key)
            elif self._values_pending.exists(x_key):
                p: gtsam.Pose2 = self._values_pending.atPose2(x_key)
            else:
                p: gtsam.Pose2 = self._last_guess_pose2 if self._last_guess_pose2 is not None else gtsam.Pose2(0.0, 0.0, 0.0)

            gb = p.theta() + float(bearing_rad)  # global bearing = yaw + body-bearing
            lx = float(p.x()) + float(range_m) * math.cos(gb)
            ly = float(p.y()) + float(range_m) * math.sin(gb)

            self._values_pending.insert(l_key, gtsam.Point2(lx, ly))


    ### OPTIMIZATION UPDATE
    def update(self) -> None:
        """Run the incremental optimizer (iSAM2) on queued factors/values."""
        if self._graph_pending.size() == 0 and self._values_pending.size() == 0:
            return

        self._isam2.update(self._graph_pending, self._values_pending)
        self._graph_pending = gtsam.NonlinearFactorGraph()
        self._values_pending = gtsam.Values()

        # Cache full estimate
        self._estimate_values = self._isam2.calculateEstimate()

        # Latest tuple
        if self._num_nodes > 0 and self._estimate_values.exists(X(self._num_nodes - 1)):
            p: gtsam.Pose2 = self._estimate_values.atPose2(X(self._num_nodes - 1))
            self._latest_estimate = (float(p.x()), float(p.y()), _wrap_angle(float(p.theta())))
            self._last_guess_pose2 = p

    ### OUTPUT FUNCTIONS
    def get_latest(self) -> Tuple[int, float, float, float]:
        """Return (stamp_ns, x, y, yaw) of the latest node."""
        if self._latest_estimate is None or not self._stamps_ns:
            raise RuntimeError("No pose estimate available. Add an anchor and run update().")
        x, y, yaw = self._latest_estimate
        return (self._stamps_ns[-1], x, y, yaw)

    def get_path(self) -> List[Tuple[int, float, float, float]]:
        """Return path as [(stamp_ns, x, y, yaw), ...]."""
        if self._num_nodes == 0:
            return []
        out: List[Tuple[int, float, float, float]] = []
        for i in range(0, self._num_nodes):
            key = X(i)
            if not self._estimate_values.exists(key):
                continue
            pose: gtsam.Pose2 = self._estimate_values.atPose2(key)
            out.append((self._stamps_ns[i], float(pose.x()), float(pose.y()), _wrap_angle(float(pose.theta()))))
        return out

    # def get_landmark(self) -> List[Tuple[int, float, float]]:
    #     out = []
    #     try:
    #         keys = list(self._estimate_values.keys())
    #     except Exception as e:
    #         self.log.error(f"[get_landmark] couldn't read result keys: {e}")
    #         return out

    #     self.log.info(f"[get_landmark] scanning {len(keys)} keys")
    #     found = 0

    #     for key in keys:
    #         s = gtsam.Symbol(key)
    #         tag = chr(s.chr())
    #         idx = s.index()
    #         # Per-key trace (keep while debugging, then downgrade to debug)
    #         self.log.info(f"[get_landmark] key={key} tag={tag} idx={idx}")

    #         if tag.lower() == 'l':   # accept 'L' or 'l'
    #             try:
    #                 pt = self._estimate_values.atPoint2(key)

    #                 # pt can be gtsam.Point2 OR a numpy array; handle both
    #                 try:
    #                     x = float(pt.x()); y = float(pt.y())
    #                 except AttributeError:
    #                     a = np.asarray(pt).reshape(-1)
    #                     x = float(a[0]); y = float(a[1] if a.size > 1 else 0.0)

    #                 out.append((idx, x, y))
    #                 found += 1
    #                 self.log.info(f"[get_landmark] L{idx} -> ({x:.3f}, {y:.3f})  type={type(pt)}")

    #             except Exception as e:
    #                 self.log.error(f"[get_landmark] failed to extract L{idx}: {e}")

    #     if found == 0:
    #         self.log.info("[get_landmark] no L* symbols in current estimate")
    #     else:
    #         self.log.info(f"[get_landmark] returning {found} landmarks")

    #     return out

    def get_landmark(self):
        out = []
        for key in list(self._estimate_values.keys()):
            s = gtsam.Symbol(key)
            if chr(s.chr()).upper() == 'L':
                try:
                    pt = self._estimate_values.atPoint2(key)
                    x, y = float(pt.x()), float(pt.y())
                except AttributeError:
                    a = np.asarray(pt).ravel()
                    if a.size < 2:
                        continue
                    x, y = map(float, a[:2])
                out.append((s.index(), x, y))
        return out


