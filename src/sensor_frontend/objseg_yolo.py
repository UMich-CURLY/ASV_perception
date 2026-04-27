#!/usr/bin/env python3

import threading
import time

import numpy as np
import cv2
import os, yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO
import torch

class ObjSegNode(Node):
    def __init__(self):
        super().__init__('objdet')

        self.bridge = CvBridge()

        self.latest_msg = None
        self.lock = threading.Lock()
        self.running = True

        pkg_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
        config_path = os.path.join(pkg_path, 'configs', 'params.yaml')
        model_path = os.path.join(pkg_path, 'models', 'best_1.pt')
        
        # Load parameters from YAML file
        with open(config_path, "r") as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # Parameters Inputs
        self.ros_topic = self.config["ros_parameters"]
        self.declare_parameter("conf", 0.25)

        # Subscriber
        self.camera_topic = self.ros_topic["camera_topic"]
        self.conf = self.get_parameter("conf").value

        # Publisher
        self.detection_topic = self.ros_topic["det_mask_topic"]
        self.annot_topic = self.ros_topic["annot_topic"]

        # YOLO model 
        self.get_logger().info(f"Loading YOLO segmentation model: {model_path}")
        self.model = YOLO(model_path)

        if torch.cuda.is_available():
            self.model.to("cuda")
            self.model.fuse()
            self.device = 0
            self.get_logger().info("Using CUDA for YOLO")
        else:
            self.device = "cpu"
            self.get_logger().warn("CUDA not available, running on CPU")

        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        # Publisher
        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.mask_pub  = self.create_publisher(Image, self.detection_topic, pub_qos)
        self.annot_pub = self.create_publisher(Image, self.annot_topic, pub_qos)

        # Worker thread
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        self.cb_count = 0

        self.get_logger().info("Object Segmentation Node Started")


    def image_callback(self, msg: Image):
        with self.lock:
            self.latest_msg = msg

    def worker_loop(self):
        while rclpy.ok() and self.running:
            want_mask = self.mask_pub.get_subscription_count() > 0
            want_annot = self.annot_pub.get_subscription_count() > 0

            if not (want_mask or want_annot):
                time.sleep(0.01)
                continue

            msg = None
            with self.lock:
                if self.latest_msg is not None:
                    msg = self.latest_msg
                    self.latest_msg = None

            if msg is None:
                time.sleep(0.005)
                continue

            self.cb_count += 1

            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                h, w = cv_image.shape[:2]
                
                # YOLOv11 Inference
                # results is a list, results[0] contains the data for one image
                t0 = time.time()
                results = self.model.predict(
                    cv_image,
                    conf=self.conf,
                    device=self.device,
                    verbose=False,
                    retina_masks=True  # Higher quality masks for better LiDAR alignment
                )[0]
                t1 = time.time()

                if want_mask:
                    # 0 = Background
                    label_map = np.zeros((h, w), dtype=np.uint8)
                    
                    if results.masks is not None:
                        # Convert masks to numpy and ensure they match image size
                        masks = results.masks.data.cpu().numpy()              # (N, Hm, Wm)
                        classes = results.boxes.cls.cpu().numpy().astype(int) # (N,)
                        
                        for i, mask in enumerate(masks):
                            # Object ID starts at 1
                            # obj_id = i + 1
                            obj_id = int(classes[i]) + 1
                            
                            # If YOLO internal mask size != image size, resize
                            if mask.shape[0] != h or mask.shape[1] != w:
                                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            
                            # Fill pixels where confidence > 0.5 with the Object ID
                            label_map[mask > 0.5] = obj_id

                    mask_msg = self.bridge.cv2_to_imgmsg(label_map, encoding="mono8")
                    mask_msg.header = msg.header # CRITICAL for time sync
                    self.mask_pub.publish(mask_msg)

                # Publish annotated image (optional)
                if want_annot:
                    annotated_img = results.plot() 
                    annot_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding="bgr8")
                    annot_msg.header = msg.header
                    self.annot_pub.publish(annot_msg)

                infer_ms = (t1 - t0) * 1000.0
                if self.cb_count % 10 == 0:
                    self.get_logger().info(
                        f"YOLO seg inference: {infer_ms:.1f} ms | "
                        f"mask_pub={want_mask} annot_pub={want_annot}"
                    )

            except Exception as e:
                self.get_logger().error(f"Error in worker_loop: {e}")
                time.sleep(0.01)

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObjSegNode()
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