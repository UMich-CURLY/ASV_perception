import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point, TransformStamped
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time
import tf2_ros
import os, yaml

### HELPER FUNCTIONS
def stamp_to_ns(stamp):  # Integer Nanoseconds
    return stamp.sec * 1_000_000_000 + stamp.nanosec

def quat_is_valid(q):  # Check if quaternion is valid
    xs = (q.x, q.y, q.z, q.w)
    if any(x is None for x in xs): return False
    if any(not math.isfinite(x) for x in xs): return False
    if all(abs(x) < 1e-12 for x in xs): return False
    n2 = sum(x*x for x in xs)
    return n2 > 1e-12

def quat_to_yaw(q):  # Quaternion to Yaw
    return math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

def wrap_to_pi(a):
    return (a + math.pi) % (2.0*math.pi) - math.pi

### NODE
class DriftBridge(Node):
    def __init__(self):
        super().__init__('drift_bridge')

        pkg_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
        config_path = os.path.join(pkg_path, 'configs', 'params.yaml')
        
        # Load model parameters
        with open(config_path, "r") as stream:
            try:
                self.params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)


        # frames
        self.odom_frame = self.params["ros_parameters"]["global_frame"]

        # Timing Gates (ms)
        self.backtrack_tol_ms = self.params["Obj_SLAM"]["drift_bridge"]["backtrack_tol_ms"]   # tolerate small backtracks/dupes
        self.min_dt_floor_ms = self.params["Obj_SLAM"]["drift_bridge"]["min_dt_floor_ms"]     # ignore micro-bursts
        self.dt_thresh_ms = self.params["Obj_SLAM"]["drift_bridge"]["dt_thresh_ms"]           # time-based KF fallback if no motion

        # Motion Thresholds
        self.trans_tresh = self.params["Obj_SLAM"]["drift_bridge"]["trans_thresh_m"]              # meters
        self.yaw_tresh = math.radians(self.params["Obj_SLAM"]["drift_bridge"]["yaw_thresh_deg"])  # rads

        # Δ State
        self.prev = None
        self.hdr_ns_prev = None  # store last KF stamp in ns for ROS headers

        # Subscribers
        self.pose_topic = self.params["ros_parameters"]["pose_topic"]
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.sub = self.create_subscription(PoseStamped, self.pose_topic, self.cb_pose, qos)

        # Publishers
        self.anchor_topic = self.params["ros_parameters"]["anchor_topic"]
        self.kf_topic = self.params["ros_parameters"]["keyframe_topic"]

        anchor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # latch latest anchor
        )
        kf_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pub_anchor = self.create_publisher(PoseWithCovarianceStamped, self.anchor_topic, anchor_qos)
        self.pub_kf = self.create_publisher(Odometry, self.kf_topic, kf_qos)

        # TF Broadcaster
        # self.declare_parameter('odom_frame', 'wamv/odom')
        # self.declare_parameter('base_link_frame', 'wamv/wamv/base_link')
        # self.odom_frame = self.get_parameter('odom_frame').value
        # self.base_link  = self.get_parameter('base_link_frame').value

        # self.tf_br = tf2_ros.TransformBroadcaster(self)

        # Logging
        self.get_logger().info(f"Listening for pose on '{self.pose_topic}'")

    ### INTERNALS
    def _seed_prev(self, p, yaw, hdr_ns):  # Reseeding
        if not (isinstance(yaw, float) and math.isfinite(yaw)):
            yaw = 0.0
        self.prev = {'x': p.x, 'y': p.y, 'z': p.z, 'yaw': yaw}
        self.hdr_ns_prev = hdr_ns

    def _ns_to_time(self, t_ns):
        t = Time()
        t.sec = int(t_ns // 1_000_000_000)
        t.nanosec = int(t_ns % 1_000_000_000)
        return t
    
    def _publish_anchor(self, p, yaw, hdr_ns):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self._ns_to_time(hdr_ns)
        msg.header.frame_id = self.odom_frame

        msg.pose.pose.position = Point(x=float(p.x), y=float(p.y), z=0.0)
        # simple yaw→quat
        msg.pose.pose.orientation.z = math.sin(yaw*0.5)
        msg.pose.pose.orientation.w = math.cos(yaw*0.5)

        # covariance hint (diagonal only)
        # this is a strong covariance hint, as the anchor is "fixed"
        cov = [0.0]*36
        cov[0]  = 0.1**2
        cov[7]  = 0.1**2
        cov[35] = math.radians(5.0)**2

        # out-of-plane: unconstrained
        cov[14] = 1e6                       # z var
        cov[21] = 1e6                       # roll var
        cov[28] = 1e6                       # pitch var

        msg.pose.covariance = cov

        self.pub_anchor.publish(msg)
        self.get_logger().info("Published drift_anchor")

    def _publish_keyframe(self, p, yaw, hdr_ns, dx_b, dy_b, dyaw, dt_ms):
        odom = Odometry()
        odom.header.stamp = self._ns_to_time(hdr_ns)
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = "drift_kf"

        # absolute pose at KF
        odom.pose.pose.position = Point(x=float(p.x), y=float(p.y), z=0.0)
        odom.pose.pose.orientation.z = math.sin(yaw*0.5)
        odom.pose.pose.orientation.w = math.cos(yaw*0.5)

        # pack relative motion in twist
        odom.twist.twist.linear.x  = float(dx_b)
        odom.twist.twist.linear.y  = float(dy_b)
        odom.twist.twist.angular.z = float(dyaw)

        # pack dt into covariance slots
        cov = [0.0]*36
        cov[0] = float(dt_ms)     # ms since prev KF
        odom.twist.covariance = cov

        self.pub_kf.publish(odom)
        # self.get_logger().info("Published drift_kf")

    ### CALLBACK
    def cb_pose(self, msg: PoseStamped):
        # Extract message
        p = msg.pose.position
        q = msg.pose.orientation
        hdr_ns = stamp_to_ns(msg.header.stamp)

        # Validate quaternion / yaw
        if not quat_is_valid(q):
            self.get_logger().warn("Invalid quaternion; reseeding.")
            yaw_seed = self.prev['yaw'] if self.prev else 0.0
            self._seed_prev(p, yaw_seed, hdr_ns)
            return

        yaw = quat_to_yaw(q)

        # # Broadcast TF for odom -> base_link
        # t = TransformStamped()
        # t.header.stamp = msg.header.stamp
        # t.header.frame_id = self.odom_frame
        # t.child_frame_id  = self.base_link
        # t.transform.translation.x = float(msg.pose.position.x)
        # t.transform.translation.y = float(msg.pose.position.y)
        # t.transform.translation.z = 0.0
        # t.transform.rotation = msg.pose.orientation
        # self.tf_br.sendTransform(t)

        # Initialize Pose for Δ
        if self.prev is None:
            self.get_logger().info("Initialized Δ-odometry reference.")
            self._seed_prev(p, yaw, hdr_ns)
            self._publish_anchor(p, yaw, hdr_ns)
            return
        
        # Δt in ms
        dt_ms = (hdr_ns - self.hdr_ns_prev) * 1e-6
        if dt_ms <= 0.0:  # negative or duplicate
            if abs(dt_ms) <= self.backtrack_tol_ms:
                # self.get_logger().warn(f"Small Backtrack {dt_ms:+.2f} ms")
                return  # small backtrack/dupe → drop
            self.get_logger().warn(f"Skipping Δ (hdr backtrack {dt_ms:+.2f} ms).")
            return            
        # if dt_ms < self.min_dt_floor_ms:  # too small to be useful
        #     self.get_logger().warn(f"Skipping {dt_ms:.2f} ms micro-burst.")
        #     return
            
        # World-frame deltas
        dx_w = p.x - self.prev['x']
        dy_w = p.y - self.prev['y']
        dyaw = wrap_to_pi(yaw - self.prev['yaw'])

        # Rotate into body frame
        cy, sy = math.cos(self.prev['yaw']), math.sin(self.prev['yaw'])
        dx_b =  cy*dx_w + sy*dy_w
        dy_b = -sy*dx_w + cy*dy_w

        # self.get_logger().info(
        #     f"Δ_b: dx={dx_b:+.6f} m, dy={dy_b:+.6f} m, dθ={dyaw:+.6f} rad | "
        #     f"dt_arr={dt_ms:.6f} ms"
        # )

        # Keyframing
        if (math.hypot(dx_b, dy_b) >= self.trans_tresh) or (abs(dyaw) >= self.yaw_tresh):
            self._seed_prev(p, yaw, hdr_ns)
            self._publish_keyframe(p, yaw, hdr_ns, dx_b, dy_b, dyaw, dt_ms)  # MOTION
        elif (dt_ms >= self.dt_thresh_ms):
            self._seed_prev(p, yaw, hdr_ns)
            self._publish_keyframe(p, yaw, hdr_ns, dx_b, dy_b, dyaw, dt_ms)  # TIME
            


def main(args=None):
    rclpy.init(args=args)
    node = DriftBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()