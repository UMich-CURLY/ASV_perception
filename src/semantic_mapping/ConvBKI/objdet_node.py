import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, CameraInfo

from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
import numpy as np
import cv2
import tf2_ros

import torch
from torchvision.ops import box_convert

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "Segmentation"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

from PIL import Image as PILImage

import ros2_numpy
import transforms3d

from sklearn.cluster import DBSCAN, KMeans
import time
### --- TODO: Tranfer these to params.yaml ---

# Camera Intrinsics
CAMERA_INTRINSICS = np.array([[762.722, 0.0, 640.0],  
                              [0.0, 762.722, 360.0],  
                              [0.0, 0.0, 1.0]])

# Camera to LiDAR Transformation
# ROTATION_CAMERA_TO_LIDAR = np.array([0.0, -15.0, 0.0])  
# TRANSLATION_CAMERA_TO_LIDAR = np.array([-0.05, -0.1, 0.30])  
ROTATION_CAMERA_TO_LIDAR = np.array([0.0, 0.0, 0.0])  
TRANSLATION_CAMERA_TO_LIDAR = np.array([0.0, 0.0, 1.0])  

# Compute rotation matrix
R_matrix = transforms3d.euler.euler2mat(
    np.radians(ROTATION_CAMERA_TO_LIDAR[0]),
    np.radians(ROTATION_CAMERA_TO_LIDAR[1]),
    np.radians(ROTATION_CAMERA_TO_LIDAR[2])
)

CAMERA_TO_LIDAR_TRANSFORM = np.eye(4)
CAMERA_TO_LIDAR_TRANSFORM[:3, :3] = R_matrix
CAMERA_TO_LIDAR_TRANSFORM[:3, 3] = TRANSLATION_CAMERA_TO_LIDAR

Cf_TO_Cw_TRANSFORM = np.eye(4)
Cf_TO_Cw_TRANSFORM[:3, :3] = np.array([[0, -1, 0],
                                       [0, 0, -1],
                                       [1, 0, 0]])


fixed_class_id_mapping = {
    "tree": 0,
    "grass - shore": 1,
    "concreteplatform": 2,
    "water": 3,
    "redbuoy": 4,
    "blackbuoy": 5,
    "greenbuoy": 6,
    "orangebuoy": 7,
    "whitebuoy": 8,
}

color_map = {
    0: [200, 275, 200], # tree...very light green
    1: [0, 102, 0], # grass-shore...darkgreen
    2: [160, 160, 160], # concrete_platform...gray
    3: [30, 60, 150], # water...dark-blue
    4: [255, 30, 30], # redbuoy...red
    5: [0, 0, 0], # blackbuoy...black
    6: [150, 240, 80], # greenbuoy...lightgreen
    7: [255, 128, 0], # orangebuoy...orange
    8: [255, 255, 255] # whitebuoy...white
}

# fixed_class_id_mapping = {
#     "blackbuoy": 0,
#     "orangebuoy": 1,
#     "whitebuoy": 2,
#     "greenbuoy": 3,
#     "redbuoy": 4
# }

# color_map = {  # BGR (OpenCV)
#     0: [0,   0,   0],   # blackbuoy -> black
#     1: [0, 165, 255],   # orangebuoy -> orange
#     2: [255,255,255],   # whitebuoy  -> white
#     3: [0, 255,   0],   # greenbuoy  -> green
#     4: [0,   0, 255],   # redbuoy    -> red
# }

class ObjDetNode(Node):
    def __init__(self):
        super().__init__('objdet')

        # Default Parameters (Fall Back)
        self.declare_parameters(
            "",
            [
                ("camera_topic", "/wamv/sensors/cameras/camera_sensor/optical/image_raw"),
                ("camera_info_topic", "/wamv/sensors/cameras/camera_sensor/camera_info"),
                # ("camera_topic", "/wamv/sensors/cameras/front_left_camera_sensor/optical/image_raw"),
                # ("camera_info_topic", "wamv/sensors/cameras/front_left_camera_sensor/optical/camera_info"),
                ("lidar_topic", "/wamv/sensors/lidars/lidar_sensor/points"),
                ("sam2_checkpoint", "Segmentation/checkpoints/sam2.1_hiera_tiny.pt"),
                ("sam2_model_config", "configs/sam2.1/sam2.1_hiera_t.yaml"),
                ("grounding_dino_config", "Segmentation/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                ("grounding_dino_checkpoint", "Segmentation/gdino_checkpoints/groundingdino_swint_ogc.pth"),
                # ("sam2_checkpoint", "/home/multy-surya/ASV-Object-SLAM/Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"),
                # ("sam2_model_config", "configs/sam2.1/sam2.1_hiera_l"),
                ("box_thresh", 0.35),
                ("text_thresh", 0.25),
                ("score_thresh", 0.40),
                
                ("prompts", "tree. grass - shore. concreteplatform. water. redbuoy. blackbuoy. greenbuoy. orangebuoy. whitebuoy."),
                # ("prompts", "blackbuoy. orangebuoy. whitebuoy. greenbuoy. redbuoy."),
                ("visualize", False),    # True
            ],
        )
        # Get Parameters from config/params.yaml
        self.camera_topic = self.get_parameter("camera_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.lidar_topic  = self.get_parameter("lidar_topic").value
        self.prompts      = self.get_parameter("prompts").value
        self.visualize    = bool(self.get_parameter("visualize").value)
        self.sam2_checkpoint = self.get_parameter("sam2_checkpoint").value
        self.sam2_model_config = self.get_parameter("sam2_model_config").value
        self.grounding_dino_config = self.get_parameter("grounding_dino_config").value
        self.grounding_dino_checkpoint = self.get_parameter("grounding_dino_checkpoint").value
        self.box_thresh = self.get_parameter("box_thresh").value
        self.text_thresh = self.get_parameter("text_thresh").value
        self.score_thresh = self.get_parameter("score_thresh").value

        # OpenCV
        self.bridge = CvBridge()

        # Object Detection Model Grounded SAM 2 + Grounding DINO
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)

        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        self.dino_model = load_model(
            model_config_path=self.grounding_dino_config,
            model_checkpoint_path=self.grounding_dino_checkpoint,
            device=self.device
        )

        # Camera intrinsics / Subscriber for Camera Info
        self.fx = self.fy = self.cx = self.cy = None
        self.img_w = self.img_h = None
        self.camera_frame = None
        self.create_subscription(CameraInfo, self.camera_info_topic, self.cb_caminfo, qos_profile_sensor_data)

        # Subscribers for Camera and Lidar
        self.pc_sub = Subscriber(self, PointCloud2, self.lidar_topic)
        self.img_sub = Subscriber(self, Image, self.camera_topic)

        self.ts = ApproximateTimeSynchronizer([self.pc_sub, self.img_sub], queue_size=50, slop=0.03)
        self.ts.registerCallback(self.cb)

        # Publishers
        self.pub_det = self.create_publisher(Detection2DArray, "/asv_object_slam/detection/list_raw", 10)
        self.pub_mask = self.create_publisher(Image, "/asv_object_slam/detection/segmentation_mask", 10)
        self.pub_img = self.create_publisher(Image, "/asv_object_slam/detection/annotated_image", 5)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Logging for debug / info
        self.get_logger().info(
            "Object Detection Node up. Subscribed to:\n"
            f"  camera: {self.camera_topic}\n"
            f"  camera_info: {self.camera_info_topic}\n"
            f"  lidar:  {self.lidar_topic}\n"
            f"  prompts: {self.prompts}\n"
            f"  device: {self.device}"
        )

    def color_for(self, lab):
        # lab can be string or class id
        cid = fixed_class_id_mapping.get(lab, -1) if isinstance(lab, str) else int(lab)
        return tuple(int(c) for c in color_map.get(cid, [200, 200, 200]))

    def put_label(self, img, box, text, color):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        y = max(0, y1 - 6); x = max(0, x1 + 2)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def project_lidar_to_image(self, points_3d, image_width, image_height):
        """Project 3D LiDAR points into 2D camera image plane and filter points outside the FOV."""
        
        points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # [x, y, z, 1]

        # Transform LiDAR points to camera frame
        points_camera = (Cf_TO_Cw_TRANSFORM @ CAMERA_TO_LIDAR_TRANSFORM @ points_3d_h.T).T[:, :3]

        # Keep only points in front of the camera
        valid_camera_indices = points_camera[:, 2] > 0  
        points_camera = points_camera[valid_camera_indices]

        # Project onto image plane
        pixels = (CAMERA_INTRINSICS @ points_camera.T).T
        pixels = pixels[:, :2] / pixels[:, 2:]  # Normalize by depth

        # Filter points within image bounds
        valid_fov_indices = (pixels[:, 0] >= 0) & (pixels[:, 0] < image_width) & \
                            (pixels[:, 1] >= 0) & (pixels[:, 1] < image_height)

        # Apply final filtering
        valid_indices = np.where(valid_camera_indices)[0][valid_fov_indices]  # Indices w.r.t original input
        pixels = pixels[valid_fov_indices].astype(int)
        
        return pixels, points_camera[valid_fov_indices], valid_indices
    
    def generate_segmentation(self, pil_image):
        image_source, image = load_image(pil_image)
        self.sam2_predictor.set_image(image_source)

        boxes, confidences, labels = predict(
            model=self.dino_model,
            image=image,
            caption=self.prompts,
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh,
            remove_combined=True,
        )

        # scale boxes to pixels
        h, w, _ = image_source.shape
        boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

        # --- score filter + sorting ---
        scores = confidences.detach().float().cpu().numpy().reshape(-1)
        keep = scores >= self.score_thresh
        xyxy = xyxy[keep]
        scores = scores[keep]
        labels = [l for l, k in zip(labels, keep) if k]

        # sort high→low for nicer overlays
        if len(scores):
            order = np.argsort(-scores)
            xyxy = xyxy[order]
            scores = scores[order]
            labels = [labels[i] for i in order]

        # run SAM2 only on kept boxes
        input_boxes = xyxy.cpu().numpy()
        if input_boxes.size == 0:
            return np.zeros((0, h, w), dtype=np.uint8), [], np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        masks, _, _ = self.sam2_predictor.predict(
            point_coords=None, point_labels=None, box=input_boxes, multimask_output=False
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        return masks, labels, input_boxes, scores
    
    # DBSCAN
    def fg_mask_dbscan_Z(self, Z, eps=0.30, min_samples=10):
        if Z.size == 0:
            return np.zeros_like(Z, dtype=bool)
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit_predict(Z[:, None])
        # If all noise, fallback: keep all (you can choose a different fallback)
        if (labels < 0).all():
            return np.ones_like(Z, dtype=bool)
        # Pick cluster with smallest mean depth as foreground
        clusters = [c for c in np.unique(labels) if c >= 0]
        fg_label = min(clusters, key=lambda c: Z[labels == c].mean())
        return labels == fg_label

    # Camera Intrinsics Callback
    def cb_caminfo(self, msg: CameraInfo):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]
        self.img_w, self.img_h = msg.width, msg.height
        self.camera_frame = msg.header.frame_id 

    def cb(self, pc_msg, img_msg):
        timer0 = time.time()
        pc_np = ros2_numpy.point_cloud2.point_cloud2_to_array(pc_msg)

        xyz = pc_np['xyz']  
        intensity = pc_np['intensity']  # not used for now

        lidar_raw = np.zeros((xyz.shape[0], 4))
        lidar_raw[:, :3] = xyz
        lidar_raw[:, 3] = intensity[:, 0]

        width = int(img_msg.width)
        height = int(img_msg.height)
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        projected_pixels, cam_pts, valid_camera_indices = self.project_lidar_to_image(lidar_raw[:, :3], width, height)

        # Check if projected pixels are empty
        if projected_pixels.shape[0] == 0:
            self.get_logger().warn("No valid LiDAR points in camera FOV. Skipping this callback.")
            return  # Exit early and wait for the next callback

        lidar = lidar_raw[valid_camera_indices]  # only keep LiDAR points in the image frame

        # Convert OpenCV image to PIL Image
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))


        timer1 = time.time()
        print(f"timer1: {timer1 - timer0}")
        with torch.no_grad():
            masks, labels, input_boxes, scores = self.generate_segmentation(pil_image)
        # torch.set_float32_matmul_precision("high")  # optional
        # with torch.autocast(device_type="cuda", dtype=torch.float16):
        #     masks, labels, input_boxes, scores = self.generate_segmentation(pil_image)  # 0.16 sec
        
        timer2 = time.time()
        print(f"timer2: {timer2 - timer1}")

        # Publish depth for tracking node
        det_arr = Detection2DArray()
        det_arr.header = img_msg.header

        H, W = cv_image.shape[:2]
        masks_msg = Image()
        masks_msg.header = img_msg.header
        masks_msg.height = H
        masks_msg.width  = W
        masks_msg.encoding = "mono8"
        masks_msg.is_bigendian = 0
        masks_msg.step = W                     # bytes per row for mono8

        uu = projected_pixels[:, 0]
        vv = projected_pixels[:, 1]
        
        inst_mask = np.zeros((H, W), dtype=np.uint8)    # instance map for all masks, 0:bg, 1,...N:class
        per_det_overlays = [] # for visualization segmentation points

        if len(labels) != 0:
            self.get_logger().info(f"new detection frame with {len(labels)} objects.")
        else:
            self.get_logger().info("no objects detected in this frame.")
        
        timer3 = time.time()
        print(f"timer3: {timer3 - timer2}")

        for m, lab, box, score in zip(masks, labels, input_boxes, scores):
            # normalize mask shape and type
            m = m.squeeze()
            if hasattr(m, "cpu"):  # torch tensor
                m = m.cpu().numpy()
            if m.ndim == 3:
                m = m[...,0]
            if m.shape[:2] != (H,W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            m = m.astype(bool)

            # ## yui: publish masks here using highest score if overlapping masks exist, 
            # ## masks is sorted in descending order by score
            # ## do not overwrite existing class ids for pixels
            inst_mask[(m > 0) & (inst_mask == 0)] = fixed_class_id_mapping.get(lab, 255) + 1  # +1 to avoid bg=0

            # pick LiDAR points whose projected pixels lie inside mask
            hit = m[vv, uu]               # boolean over projected points
            if not np.any(hit):
                continue            

            pts_cam = cam_pts[hit]        # (K,3) in camera frame

            # DBSCAN to filter foreground points based on depth (Z in camera frame)
            fg_mask = self.fg_mask_dbscan_Z(pts_cam[:, 2], eps=0.30, min_samples=10)
            if not np.any(fg_mask):
                continue
            pts_cam = pts_cam[fg_mask]

            K = pts_cam.shape[0]  # number of points after mask + dbscan

            # indices of projected points that hit the mask
            idx_all = np.arange(len(cam_pts))
            hit_idx = idx_all[hit]

            # keep only those that survived DBSCAN
            fg_idx = hit_idx[fg_mask]

            # pixel coords + depths for survivors
            uu_db = uu[fg_idx]
            vv_db = vv[fg_idx]
            z_db  = cam_pts[fg_idx, 2]

            # safety: keep only points inside image bounds
            in_img_db = (uu_db >= 0) & (uu_db < W) & (vv_db >= 0) & (vv_db < H)
            uu_db = uu_db[in_img_db].astype(int)
            vv_db = vv_db[in_img_db].astype(int)
            z_db  = z_db[in_img_db]

            # stash for overlay drawing
            per_det_overlays.append((lab, uu_db, vv_db, z_db))

            # robust centroid in camera frame
            xc = np.median(pts_cam[:, 0])
            yc = np.median(pts_cam[:, 1])
            zc = np.median(pts_cam[:, 2])

            # message
            det = Detection2D()
            det.header = img_msg.header

            # class + score
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(fixed_class_id_mapping.get(lab, -1))
            hyp.hypothesis.score = float(K)

            # pose in camera frame
            hyp.pose.pose.position.x = float(xc)
            hyp.pose.pose.position.y = float(yc)
            hyp.pose.pose.position.z = float(zc)

            # simple diag covariance scaled by sample count
            # tune later
            scale = float(max(1.0, K/50.0))
            cov = [0.0] * 36
            cov[0]  = 0.04 / scale   # var x
            cov[7]  = 0.04 / scale   # var y
            cov[14] = 0.25 / scale   # var z
            cov[21] = 1e6                  # var roll
            cov[28] = 1e6                  # var pitch
            cov[35] = 1e6                  # var yaw
            hyp.pose.covariance = cov

            det.results.append(hyp)
            det_arr.detections.append(det)

            self.get_logger().info(f"{lab} detected at ({xc:.2f}, {yc:.2f}, {zc:.2f}) with {K} points.")
        
        timer4 = time.time()
        print(f"timer4: {timer4 - timer3}")
        masks_msg.data = np.ascontiguousarray(inst_mask).tobytes()      # 0.03 to 0.04 s

        timer5 = time.time()
        print(f"timer5: {timer5 - timer4}")

        if self.pub_det.get_subscription_count() > 0:
            if len(det_arr.detections):
                self.pub_det.publish(det_arr)
        if self.pub_mask.get_subscription_count() > 0:
                self.pub_mask.publish(masks_msg)
        
        timer6 = time.time()
        print(f"timer6: {timer6 - timer5}")
        print(f"Subtotal: {timer6 - timer0}")
        
        # Publish annotated (only if visualize)
        if self.visualize and self.pub_img.get_subscription_count() > 0:
            try:
                overlay = cv_image.copy()

                # Masking overlay
                alpha = 0.35
                H, W = overlay.shape[:2]

                for m, lab, box in zip(masks, labels, input_boxes):
                    m = np.array(m)                    # to ndarray
                    if m.ndim == 3:                    # drop channel if present
                        m = m[..., 0]
                    if m.shape[:2] != (H, W):          # resize if needed
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    m = m > 0                          # boolean mask

                    col = np.array(self.color_for(lab), dtype=np.uint8)  # shape (3,)
                    # alpha blend only where mask is true
                    region = overlay[m]                                   # Nx3
                    overlay[m] = (region * (1 - alpha) + col * alpha).astype(np.uint8)

                    # optional outline + label
                    cnts, _ = cv2.findContours((m.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, cnts, -1, tuple(int(c) for c in col.tolist()), 2)
                    self.put_label(overlay, box, str(lab), tuple(int(c) for c in col.tolist()))

                # LiDAR image overlay
                # try:
                #     # safety: keep only points inside image bounds
                #     in_img = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                #     if np.any(in_img):
                #         uu_in = uu[in_img].astype(int)
                #         vv_in = vv[in_img].astype(int)
                #         z_in  = cam_pts[in_img, 2]  # depth in camera frame (meters)

                #         # depth → grayscale (near = bright)
                #         z_clip = np.clip(z_in, 0.0, np.percentile(z_in, 95))
                #         z_norm = (z_clip - z_clip.min()) / (z_clip.ptp() + 1e-6)
                #         gray   = (255 * (1.0 - z_norm)).astype(np.uint8)

                #         # draw small filled circles
                #         pt_radius  = 2
                #         thickness  = -1
                #         for x, y, g in zip(uu_in, vv_in, gray):
                #             cv2.circle(overlay, (int(x), int(y)), pt_radius, (int(g), int(g), int(g)), thickness)

                #         # Overlay Lidar inside masks
                #         Hov, Wov = overlay.shape[:2]  # use current overlay size

                #         for m, lab in zip(masks, labels):
                #             mm = np.array(m)
                #             if mm.ndim == 3:
                #                 mm = mm[..., 0]
                #             if mm.shape[:2] != (Hov, Wov):
                #                 mm = cv2.resize(mm, (Wov, Hov), interpolation=cv2.INTER_NEAREST)
                #             mm = mm.astype(bool)

                #             # points (uu_in, vv_in) already filtered to be inside the image
                #             hit = mm[vv_in, uu_in]
                #             if not np.any(hit):
                #                 continue

                #             uu_hit = uu_in[hit].astype(int)
                #             vv_hit = vv_in[hit].astype(int)

                #             # distinct color for cam_pts inside mask (use per-label color; example: thicker dots)
                #             col_hit = tuple([0, 255, 255])
                #             for xh, yh in zip(uu_hit, vv_hit):
                #                 cv2.circle(overlay, (int(xh), int(yh)), 3, col_hit, -1)  # radius 3, filled

                # except Exception as e:
                #     self.get_logger().warning(f"LiDAR point overlay failed: {e}")       

                try:
                    Hov, Wov = overlay.shape[:2]  # current overlay size

                    # --- 1) BACKGROUND: draw ALL projected points (faint) ---
                    in_img_all = (uu >= 0) & (uu < Wov) & (vv >= 0) & (vv < Hov)
                    if np.any(in_img_all):
                        uu_bg = uu[in_img_all].astype(int)
                        vv_bg = vv[in_img_all].astype(int)
                        z_bg  = cam_pts[in_img_all, 2]

                        # depth → grayscale (near = bright) with zero-range guard
                        zc = np.clip(z_bg, 0.0, np.percentile(z_bg, 95))
                        zr = zc.ptp()
                        if zr < 1e-6:
                            gray_bg = np.full(zc.shape, 160, dtype=np.uint8)  # constant mid-gray
                        else:
                            zn = (zc - zc.min()) / (zr + 1e-6)
                            gray_bg = (255 * (1.0 - zn)).astype(np.uint8)

                        # small, faint dots for all points
                        for x, y, g in zip(uu_bg, vv_bg, gray_bg):
                            cv2.circle(overlay, (int(x), int(y)), 1, (int(g), int(g), int(g)), -1)

                    # --- 2) FOREGROUND: draw survivors (post-DBSCAN) on top ---
                    for lab, uu_db, vv_db, z_db in per_det_overlays:
                        if len(uu_db) == 0:
                            continue

                        zc = np.clip(z_db, 0.0, np.percentile(z_db, 95))
                        zr = zc.ptp()
                        if zr < 1e-6:
                            gray_fg = np.full(zc.shape, 220, dtype=np.uint8)
                        else:
                            zn = (zc - zc.min()) / (zr + 1e-6)
                            gray_fg = (255 * (1.0 - zn)).astype(np.uint8)

                        # survivors as brighter/larger gray dots
                        for x, y, g in zip(uu_db, vv_db, gray_fg):
                            cv2.circle(overlay, (int(x), int(y)), 2, (int(g), int(g), int(g)), -1)

                        # optional colored halo per class
                        col = tuple([0, 255, 255])
                        for x, y in zip(uu_db, vv_db):
                            cv2.circle(overlay, (int(x), int(y)), 3, col, -1)

                except Exception as e:
                    self.get_logger().warning(f"LiDAR point overlay failed: {e}")      
                
                timer7 = time.time()
                print(f"timer7: {timer7 - timer6}")

                msg = self.bridge.cv2_to_imgmsg(np.ascontiguousarray(overlay), encoding="bgr8")
                msg.header = img_msg.header
                self.pub_img.publish(msg)
                
                timer8 = time.time()
                print(f"timer8: {timer8 - timer7}")
                print(f"Total: {timer8 - timer0}")
            except Exception as e:
                self.get_logger().warning(f"Failed to publish annotated image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjDetNode()
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