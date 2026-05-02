from typing import Dict, Optional, Tuple, List
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros

from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import (
    Detection2DArray, Detection2D, ObjectHypothesisWithPose
)
from tf2_geometry_msgs import do_transform_pose_stamped

import threading
from typing import Callable

import numpy as np
import os, yaml

### HELPERS
def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

class IdAllocator:  # thread-safe ID allocator
    def __init__(self, start:int = 0):
        self._n = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            n = self._n
            self._n += 1
            return n

class Track:  # [track_id, Track]
    def __init__(self, track_id: int, pose: PoseStamped, timestamp: rclpy.time.Time) -> None:
        self.track_id = track_id
        self.last_timestamp = timestamp
        self.num_detected = 0

        self.x = np.array([pose.pose.position.x, pose.pose.position.y], dtype=float)  # [x, y]
        self.P = np.diag([1.0, 1.0]).astype(float)  # start a bit uncertain

    def increase_detected(self) -> int:
        self.num_detected += 1
        return self.num_detected

class TrackManager:  # [class_id, TrackManager]
    def __init__(self, drop_seconds, promote_hits, promote_window, id_allocator: Callable[[], int],
                meas_std_xy: Tuple[float, float] = (0.5, 0.5),
                proc_std_xy: Tuple[float, float] = (0.5, 0.5),
                gate_prob: float = 0.95) -> None:
        
        self.tracks: Dict[int, Track] = {}  # track_id -> Track
        self.drop_seconds = drop_seconds  # when to declare as LOST
        self.promote_hits = promote_hits  # when to promote
        self.promote_window = promote_window  # time window for promotion
        self._next_id = id_allocator

        self.R = np.diag(np.array(meas_std_xy, dtype=float)**2)      # measurement noise (x,y)
        self.q = np.array(proc_std_xy, dtype=float)**2               # process noise power (per second) for x,y

        # hardcoded chi-square thresholds
        if gate_prob >= 0.99:   self.gate_thresh = 9.210
        elif gate_prob >= 0.95: self.gate_thresh = 5.991
        else:                   self.gate_thresh = 2.279  # ~80%


    def update(self, pose: PoseStamped, timestamp: rclpy.time.Time) -> Track:  # returns track
        z = np.array([pose.pose.position.x, pose.pose.position.y], dtype=float)  # measurement

        if not self.tracks:  # no existing tracks, create new
            trk = self.create_track(pose, timestamp)
            count = trk.increase_detected()
            return None  # dont send track on first detection
        
        # Predict each track and compute Mahalanobis d^2
        best_tid = None
        best_d2 = self.gate_thresh
        best_stats = None

        for tid, trk in self.tracks.items():
            dt = (stamp_to_ns(timestamp) - stamp_to_ns(trk.last_timestamp)) * 1e-9
            if dt < 0: dt = 0.0

            Q = np.diag(self.q * max(dt, 1e-3))

            x_pred = trk.x  # mean
            P_pred = trk.P + Q  # covariance + time update

            y = z - x_pred  # innovation
            S = P_pred + self.R  # innovation covariance

            # Mahalanobis distance squared
            try:
                d2 = float(y.T @ np.linalg.solve(S, y))
            except np.linalg.LinAlgError:
                S_ = S + 1e-6 * np.eye(2)
                d2 = float(y.T @ np.linalg.solve(S_, y))

            if d2 < best_d2:
                best_d2 = d2
                best_tid = tid
                best_stats = (x_pred, P_pred, y, S)

        # If a gated match exists, do Kalman update; else create new track
        if best_tid is not None:
            trk = self.tracks[best_tid]
            x_pred, P_pred, y, S = best_stats
            # K = P_pred * S^-1  (H=I)
            K = P_pred @ np.linalg.inv(S)
            x_new = x_pred + K @ y
            P_new = (np.eye(2) - K) @ P_pred
            # Symmetrize (numerical hygiene)
            P_new = 0.5 * (P_new + P_new.T)

            # Commit
            trk.x = x_new
            trk.P = P_new
            trk.last_timestamp = timestamp
            count = trk.increase_detected()
            if count >= self.promote_hits:
                return trk
            else:
                return None  # not yet promoted

        # No gated match, start a new track
        trk = self.create_track(pose, timestamp)
        count = trk.increase_detected()
        return None  # dont send track on first detection


    def create_track(self, pose: PoseStamped, timestamp: rclpy.time.Time) -> Track:
        tid = self._next_id()  # get globally unique, monotonic id
        trk = Track(track_id=tid, pose=pose, timestamp=timestamp)
        self.tracks[tid] = trk
        return trk

### TRACKING NODE
class TrackingNode(Node):
    def __init__(self) -> None:
        super().__init__("tracking")

        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        pkg_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
        config_path = os.path.join(pkg_path, 'configs', 'params.yaml')
        
        # Load model parameters
        with open(config_path, "r") as stream:
            try:
                self.params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)


        # parameters
        # self.declare_parameter("detections_topic", "/asv_object_slam/detection/list_raw")
        # self.declare_parameter("meas_topic", "/asv_object_slam/detection/meas_bearing_range")
        # self.declare_parameter("odom_frame", "wamv/odom")
        # self.declare_parameter("base_link_frame", "wamv/wamv/base_link")
        # self.declare_parameter("drop_seconds", 1.5)  # when to declare as LOST
        # self.declare_parameter("promote_hits", 3)    # when to promote
        # self.declare_parameter("promote_window", 5)  # time window for promotion
        # self.declare_parameter("min_score", 0.0)  # min detection score

        # self.detections_topic: str = self.get_parameter("detections_topic").value
        # self.meas_topic: str = self.get_parameter("meas_topic").value
        # self.odom_frame: str = self.get_parameter("odom_frame").value
        # self.base_link_frame: str = self.get_parameter("base_link_frame").value
        # self.drop_seconds: float = float(self.get_parameter("drop_seconds").value)
        # self.promote_hits: int = int(self.get_parameter("promote_hits").value)
        # self.promote_window: int = int(self.get_parameter("promote_window").value)
        # self.min_score: float = float(self.get_parameter("min_score").value)

        # parameters
        self.lm_det_topic = self.params["ros_parameters"]["lm_det_topic"]
        self.meas_topic = self.params["ros_parameters"]["meas_topic"]
        self.odom_frame = self.params["ros_parameters"]["global_frame"]
        self.base_link_frame = self.params["ros_parameters"]["base_link_frame"]
        self.camera_link_frame = self.params["ros_parameters"]["camera_link_frame"]
        self.drop_seconds = float(self.params["Obj_SLAM"]["lm_tracker"]["drop_seconds"])
        self.promote_hits = int(self.params["Obj_SLAM"]["lm_tracker"]["promote_hits"])
        self.promote_window = int(self.params["Obj_SLAM"]["lm_tracker"]["promote_window"])
        self.min_score = float(self.params["Obj_SLAM"]["lm_tracker"]["min_score"])

        # TF listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # publishers / subscribers
        self._sub = self.create_subscription(Detection2DArray, self.lm_det_topic, self._on_dets, 10)
        self._pub_meas = self.create_publisher(Detection2DArray, self.meas_topic, 10)

        # tracking state
        self.track_managers: Dict[str, TrackManager] = {}  # class_id -> TrackManager
        self.track_id_allocator = IdAllocator(start=0)  # global track ID allocator

        self.get_logger().info(
            f"tracking node ready | in: {self.lm_det_topic} | out: {self.meas_topic} | frames: odom={self.odom_frame}, base={self.base_link_frame}"
        )

    # callback
    def _on_dets(self, msg: Detection2DArray) -> None:
        if not msg.detections:
            return # no detections
            
        out = Detection2DArray()
        out.header.frame_id = self.base_link_frame  # bearing and range relative to base_link
        out.header.stamp = msg.header.stamp

        for det in msg.detections:
            if not det.results:
                continue # no classification
            hyp = det.results[0]
            point_count = float(hyp.hypothesis.score)
            if point_count < self.min_score:
                self.get_logger().info(f"Low points count detection ignored: {point_count:.2f} < {self.min_score:.2f}")
                continue # low score

            class_id = hyp.hypothesis.class_id
            pose_cam = hyp.pose.pose

            ps_cam = PoseStamped()
            ps_cam.header = det.header if det.header.frame_id else msg.header
            ps_cam.pose = pose_cam

            try:
                # camera_topical -> base_link
                tf_bl = self.tf_buffer.lookup_transform(
                    self.base_link_frame, self.camera_link_frame, 
                    rclpy.time.Time(),              # fetch the latest available transform
                    timeout=Duration(seconds=0.2),
                )
                ps_bl = do_transform_pose_stamped(ps_cam, tf_bl)  # base_link frame
                bx = float(ps_bl.pose.position.x)
                by = float(ps_bl.pose.position.y)
                bearing = math.atan2(by, bx)  # rad
                range = math.hypot(bx, by)  # m

                tf_odom = self.tf_buffer.lookup_transform(
                    self.odom_frame, self.base_link_frame, 
                    rclpy.time.Time(),              # fetch the latest available transform
                    timeout=Duration(seconds=0.2),
                )
                ps_odom = do_transform_pose_stamped(ps_bl, tf_odom)  # odometry frame
                ox = float(ps_odom.pose.position.x)
                oy = float(ps_odom.pose.position.y)
                # self.get_logger().info(f"TF passes")


            except Exception as e:
                self.get_logger().warn(f"TF error: {e}")
                continue

            # Tracking Logic
            if class_id not in self.track_managers:  # create new TrackManager if not found
                self.track_managers[class_id] = TrackManager(
                    drop_seconds=self.drop_seconds,
                    promote_hits=self.promote_hits,
                    promote_window=self.promote_window,
                    id_allocator=self.track_id_allocator.next,
                )

            trk = self.track_managers[class_id].update(ps_odom, msg.header.stamp)

            if trk is None:
                continue  # not yet promoted

            # output detection
            hyp_child = ObjectHypothesisWithPose()
            hyp_child.hypothesis.class_id = class_id 
            hyp_child.hypothesis.score = point_count
            hyp_child.pose.pose.position.x = bearing  # rad
            hyp_child.pose.pose.position.y = range    # m

            out_child = Detection2D()
            out_child.id = str(trk.track_id)
            out_child.results.append(hyp_child)

            out.detections.append(out_child)

            # self.get_logger().info(f"Track ID {trk.track_id} | class '{class_id}'")
            # self.get_logger().info(f" Track Position: x={trk.x[0]:.2f} m, y={trk.x[1]:.2f} m")
            # self.get_logger().info(f"  Odom Position: x={ox:.2f} m, y={oy:.2f} m")

        if out.detections:  # only publish if we have detections
            self._pub_meas.publish(out)

def main() -> None:
    rclpy.init()
    node = TrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    