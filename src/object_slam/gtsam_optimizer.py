import math
from typing import Optional
import os, yaml

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Quaternion, Point
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time

from utils.gtsam_core import BackendCore

import traceback

### TODO: move to params.yaml
# CLASS_ID_TO_RGBA = {
#     0: (0.0, 0.0, 0.0, 0.95),   # black
#     1: (1.0, 0.55, 0.0, 0.95),  # orange
#     2: (1.0, 1.0, 1.0, 0.95),   # white
#     3: (0.0, 1.0, 0.0, 0.95),   # green
#     4: (1.0, 0.0, 0.0, 0.95),   # red
# }

# COLOR_BY_ID = {
#     "0": (0.0, 0.0, 0.0, 1.0),   # black
#     "1": (1.0, 0.55, 0.0, 1.0),  # orange
#     "2": (1.0, 1.0, 1.0, 1.0),   # white
#     "3": (0.0, 1.0, 0.0, 1.0),   # green
#     "4": (1.0, 0.0, 0.0, 1.0),   # red
# }

### HELPER FUNCTIONS
def yaw_to_quat(yaw: float) -> Quaternion:  # yaw → quaternion (SE(2))
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q

def quat_zw_to_yaw(z: float, w: float) -> float:  # z, w (quat) to yaw (SE(2))
    # atan2(2*w*z, 1 - 2*z^2)
    return math.atan2(2.0 * (w * z), 1.0 - 2.0 * (z * z))

def stamp_to_ns(stamp: Time) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ns_to_time(ns: int) -> Time:
    t = Time()
    t.sec = int(ns // 1_000_000_000)
    t.nanosec = int(ns % 1_000_000_000)
    return t

# def rgba_from_class(cid: int) -> ColorRGBA:
#     r, g, b, a = CLASS_ID_TO_RGBA.get(cid, (0.6, 0.6, 0.6, 0.95))  # default gray
#     c = ColorRGBA()
#     c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
#     return c

# def color_from_cid(cid: Optional[str]) -> ColorRGBA:
#     r,g,b,a = COLOR_BY_ID.get(str(cid), (0.6,0.6,0.6,1.0))  # gray fallback (e.g., "-1")
#     c = ColorRGBA(); c.r,c.g,c.b,c.a = r,g,b,a
#     return c

class OptimizerNode(Node):
    def __init__(self):
        super().__init__("gtsam_backend")

        pkg_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
        config_path = os.path.join(pkg_path, 'configs', 'params.yaml')
        
        # Load model parameters
        with open(config_path, "r") as stream:
            try:
                self.params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        self.ros_params = self.params["ros_parameters"]
        self.rviz_colors = self.params["rviz_colors"]

        # parameters
        # self.declare_parameters(
        #     "",
        #     [
        #         ("anchor_topic", "/asv_object_slam/gtsam/pose_anchor"),
        #         ("keyframe_topic", "/asv_object_slam/gtsam/pose_keyframe"),
        #         ("optimized_pose_topic", "/asv_object_slam/gtsam/optimized_pose"),
        #         ("optimized_path_topic", "/asv_object_slam/gtsam/optimized_path"),
        #         ("optimized_landmarks_topic", "/asv_object_slam/gtsam/optimized_landmarks"),
        #         ("world_frame", "map"),
        #         # Backend tuning
        #         ("robust_loss", ""),
        #         ("relinearize_thresh", 0.1),
        #     ],
        # )

        self.anchor_topic = self.ros_params["anchor_topic"]
        self.keyframe_topic = self.ros_params["keyframe_topic"]
        self.meas_topic = self.ros_params["meas_topic"]
        self.optimized_pose_topic = self.ros_params["opt_pose_topic"]
        self.optimized_path_topic = self.ros_params["opt_path_topci"]
        self.optimized_landmarks_topic = self.ros_params["opt_landmarks_topic"]
        self.world_frame = self.ros_params["global_frame"]

        self.lm_class_by_id = {}  # landmark_id -> class_id (string)

        robust_loss = self.params["Obj_SLAM"]["gtsam_optimizer"]["robust_loss"]
        relin_thresh = self.params["Obj_SLAM"]["gtsam_optimizer"]["relin_thresh"]

        # GTSAM Backend
        core_params = {
            "robust_loss": robust_loss if robust_loss in (None, "Huber", "Cauchy") else None,
            "relinearize_thresh": relin_thresh,
        }
        self.core = BackendCore(core_params)

        # Publisher / Subscribers
        self.sub_anchor = self.create_subscription(
            PoseWithCovarianceStamped, self.anchor_topic, self.cb_anchor, 10
        )
        self.sub_kf = self.create_subscription(
            Odometry, self.keyframe_topic, self.cb_keyframe, 50
        )
        self.sub_meas = self.create_subscription(
            Detection2DArray, self.meas_topic, self.cb_meas, 10
        )

        self.pub_pose = self.create_publisher(PoseStamped, self.optimized_pose_topic, 10)
        self.pub_path = self.create_publisher(Path, self.optimized_path_topic, 10)
        self.pub_landmarks = self.create_publisher(MarkerArray, self.optimized_landmarks_topic, 10)

        self.get_logger().info(
            f"gtsam_backend up. Subscribed to:\n"
            f"  anchor:    {self.anchor_topic}\n"
            f"  keyframe:  {self.keyframe_topic}\n"
            f"Publishing:\n"
            f"  pose:      {self.optimized_pose_topic}\n"
            f"  path:      {self.optimized_path_topic}\n"
            f"  landmarks: {self.optimized_landmarks_topic}\n"
        )

    ### CALLBACKS
    def cb_anchor(self, msg: PoseWithCovarianceStamped):
        try:
            stamp_ns = stamp_to_ns(msg.header.stamp)
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.orientation.z
            w = msg.pose.pose.orientation.w
            yaw = quat_zw_to_yaw(z, w)

            # Extract 3x3 covariance over [x, y, yaw] from 6x6 pose covariance
            cov6 = np.array(msg.pose.covariance, dtype=float).reshape(6, 6)
            cov3 = cov6[np.ix_([0, 1, 5], [0, 1, 5])]

            self.core.add_anchor(stamp_ns, x, y, yaw, cov3)
            self.core.update()

            self._publish_outputs()
        except Exception as e:
            self.get_logger().error(f"anchor callback failed: {e}")

    def cb_keyframe(self, msg: Odometry):
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        # self.get_logger().info(f"[dbg] lin={type(lin)} ang={type(ang)}")
        try:
            stamp_ns = stamp_to_ns(msg.header.stamp)

            # Body-frame deltas (published by drift_bridge)
            dx = float(msg.twist.twist.linear.x)
            dy = float(msg.twist.twist.linear.y)
            dyaw = float(msg.twist.twist.angular.z)

            # unpack dt_ms from twist covariance
            dt_ms: Optional[int] = None
            cov = msg.twist.covariance 
            cov_len = len(cov) if cov is not None else 0
            if cov_len == 36:
                dt_val = float(cov[0])
                if dt_val >= 0.0:
                    dt_ms = int(round(dt_val))

            # No covariance hint: we used those slots for dt/trigger
            # ROOM FOR IMPROVEMENT: better way to fetch covariance from DRIFT?
            cov_hint = None 

            # Feed the backend (delta, ns stamps, ms dt)
            self.core.add_keyframe(stamp_ns, dx, dy, dyaw, dt_ms, cov_hint3x3=cov_hint)
            self.core.update()
            self._publish_outputs()

        except Exception as e:
            self.get_logger().error(f"keyframe callback failed: {e}")

    def cb_meas(self, msg: Detection2DArray):
        try:
            data = msg.detections
            n = len(data)
            if n == 0:
                return

            any_added = False
            for det in data:
                landmark_id = int(det.id)
                cid = det.results[0].hypothesis.class_id  # e.g., "0","1","2","3","4"
                self.lm_class_by_id[landmark_id] = int(cid)

                # Assuming one result per detection
                if len(det.results) == 0:
                    continue
                meas = det.results[0]
                bearing = float(meas.pose.pose.position.x)  # in radians
                rng = float(meas.pose.pose.position.y)      # in meters

                self.core.add_measurement(landmark_id, bearing, rng, None)
                self.get_logger().info(f"Added measurement: id={landmark_id}  bearing={math.degrees(bearing):.1f}°  range={rng:.2f}m")
                any_added = True

            if any_added:
                try:
                    self.core.update()
                except Exception:
                    self.get_logger().error("update() failed:\n" + traceback.format_exc())
                    return
                try:
                    self._publish_outputs()
                except Exception:
                    self.get_logger().error("_publish_outputs() failed:\n" + traceback.format_exc())
                    return
                
        except Exception:
            self.get_logger().error("measurement callback failed:\n" + traceback.format_exc())

    ### PUBLISHING HELPERS
    def _publish_outputs(self):
        # Latest pose
        try:
            stamp_ns, x, y, yaw = self.core.get_latest()
        except Exception as e:
            self.get_logger().warn(f"no latest to publish yet: {e}")
            return

        ps = PoseStamped()
        ps.header.frame_id = self.world_frame
        ps.header.stamp = ns_to_time(stamp_ns)
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = 0.0
        ps.pose.orientation = yaw_to_quat(yaw)
        self.pub_pose.publish(ps)

        # Path
        path_msg = Path()
        path_msg.header.frame_id = self.world_frame
        path_msg.header.stamp = ps.header.stamp

        for (t_ns, px, py, pyaw) in self.core.get_path():
            p = PoseStamped()
            p.header.frame_id = self.world_frame
            p.header.stamp = ns_to_time(t_ns)
            p.pose.position.x = px
            p.pose.position.y = py
            p.pose.position.z = 0.0
            p.pose.orientation = yaw_to_quat(pyaw)
            path_msg.poses.append(p)

        self.pub_path.publish(path_msg)

        # Landmarks (as RViz markers)
        lm_list = self.core.get_landmark()   # [(id, x, y)]
        if lm_list:
            ma = MarkerArray()
            stamp = ps.header.stamp

            m_pts = Marker()
            m_pts.header.frame_id = self.world_frame
            m_pts.header.stamp = stamp
            m_pts.ns = "landmarks"
            m_pts.id = 0
            m_pts.type = Marker.SPHERE_LIST
            m_pts.action = Marker.ADD
            m_pts.scale.x = 0.4
            m_pts.scale.y = 0.4
            m_pts.scale.z = 0.4

            m_pts.color.r = 1.0
            m_pts.color.g = 1.0
            m_pts.color.b = 1.0
            m_pts.color.a = 1.0  

            m_pts.points.clear()
            m_pts.colors.clear()

            # for class_id, lx, ly in lm_list: 
            #     p = Point(x=float(lx), y=float(ly), z=0.0)
            #     m_pts.points.append(p)
            #     c = rgba_from_class(int(class_id))  # map class_id -> ColorRGBA
            #     c.a = 1.0  # fully visible
            #     m_pts.colors.append(c)

            for lm_id, lx, ly in lm_list:
                m_pts.points.append(Point(x=float(lx), y=float(ly), z=0.0))
                cid = self.lm_class_by_id.get(int(lm_id))       # int like 0
                m_pts.colors.append(self.color_from_cid(cid))   # alpha = 1.0

            ma.markers.append(m_pts)
            self.pub_landmarks.publish(ma)
    
    def color_from_cid(self, cid) -> ColorRGBA:
        # r,g,b,a = COLOR_BY_ID.get(cid, (0.6,0.6,0.6,1.0))  # gray fallback (e.g., "-1")
        # c = ColorRGBA(); c.r,c.g,c.b,c.a = r,g,b,a
        # return c
        # Fallback to class 0 if missing
        r, g, b = self.rviz_colors.get(cid, self.rviz_colors[0])

        # Convert 0–255 → 0–1
        c = ColorRGBA()
        c.r = r / 255.0
        c.g = g / 255.0
        c.b = b / 255.0
        c.a = 1.0

        return c

def main(args=None):
    rclpy.init(args=args)
    node = OptimizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
