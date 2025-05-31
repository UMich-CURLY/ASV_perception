# FROM https://github.com/mit-han-lab/spvnas
import torch
import torchsparse.nn.functional as F
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets

from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate
import numpy as np
import transforms3d

import os
import cv2
import json
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "Segmentation"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import time
from typing import Union
from PIL import Image
from supervision.detection.core import Detections



__all__ = ['initial_voxelize', 'point_to_voxel', 'voxel_to_point', 'generate_seg_in', 'project_lidar_to_image', 'generate_point_labels']


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1)

    pc_hash = F.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    counts = F.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = F.spvoxelize(torch.floor(new_float_coord), idx_query,
                                   counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get(
            'idx_query') is None or z.additional_features['idx_query'].get(
                x.s) is None:
        pc_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = F.sphash(x.C.to(z.F.device))
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = F.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor


def generate_seg_in(lidar, res):
    # Create input data
    coords = np.round(lidar[:, :3] / res)
    coords -= coords.min(0, keepdims=1)
    feats = lidar
    # Filter out duplicate points
    coords, indices, inverse = sparse_quantize(coords, return_index=True, return_inverse=True)
    coords = torch.tensor(coords, dtype=torch.int)
    feats = torch.tensor(feats[indices], dtype=torch.float)

    inputs = SparseTensor(coords=coords, feats=feats)
    inputs = sparse_collate([inputs]).cuda()
    return inputs, inverse

# Camera Intrinsics
CAMERA_INTRINSICS = np.array([[762.722, 0.0, 640.0],  
                               [0.0, 762.722, 360.0],  
                               [0.0, 0.0, 1.0]])

# Camera to LiDAR Transformation
ROTATION_CAMERA_TO_LIDAR = np.array([0.0, -15.0, 0.0])  
TRANSLATION_CAMERA_TO_LIDAR = np.array([-0.05, -0.1, 0.30])  

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

def project_lidar_to_image(points_3d, camera_intrinsics=CAMERA_INTRINSICS, image_width=1280, image_height=720):
    """Project 3D LiDAR points into 2D camera image plane and filter points outside the FOV."""
    
    points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # [x, y, z, 1]

    # Transform LiDAR points to camera frame
    points_camera = (Cf_TO_Cw_TRANSFORM @ CAMERA_TO_LIDAR_TRANSFORM @ points_3d_h.T).T[:, :3]

    # Keep only points in front of the camera
    valid_camera_indices = points_camera[:, 2] > 0  
    points_camera = points_camera[valid_camera_indices]

    # Project onto image plane
    pixels = (camera_intrinsics @ points_camera.T).T
    pixels = pixels[:, :2] / pixels[:, 2:]  # Normalize by depth

    # Filter points within image bounds
    valid_fov_indices = (pixels[:, 0] >= 0) & (pixels[:, 0] < image_width) & \
                        (pixels[:, 1] >= 0) & (pixels[:, 1] < image_height)

    # Apply final filtering
    valid_indices = np.where(valid_camera_indices)[0][valid_fov_indices]  # Indices w.r.t original input
    pixels = pixels[valid_fov_indices].astype(int)
    
    return pixels, points_camera[valid_fov_indices, 2], valid_indices

"""
Hyper parameters
"""
# TEXT_PROMPT = "grass-shore. tree. building. tent. concreteplatform. water. redbuoy. blackbuoy. greenbuoy. orangebuoy. whitebuoy."
TEXT_PROMPT = "grass-shore. tree. concreteplatform. water. redbuoy. blackbuoy. greenbuoy. orangebuoy. whitebuoy."
# IMG_PATH = "notebooks/images_left/1735055673502660793.png"
SAM2_CHECKPOINT = "Segmentation/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "Segmentation/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "Segmentation/gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.30 #0.30
TEXT_THRESHOLD = 0.05 #0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import torch
import numpy as np
from torchvision.ops import box_convert
from supervision.detection.core import Detections

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

def generate_point_labels(lidar, res, image, img_msg):
    """
    Generates input for the segmentation model with assigned point labels.
    
    Args:
        lidar (np.array): LiDAR point cloud (N, 4) [x, y, z, intensity].
        res (float): Voxel resolution.
        image (np.array): RGB image from the camera.
        img_msg (sensor_msgs.msg.Image): ROS2 Image message (for timestamp-based naming).

    Returns:
        SparseTensor: Sparse representation of the LiDAR data.
        inverse (np.array): Mapping for back-projection.
        point_labels (np.array): One-hot encoded labels for each LiDAR point (N, num_classes).
    """
    # Extract timestamp for unique naming
    timestamp = img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9  # Convert to seconds
    timestamp_str = f"{timestamp:.6f}".replace('.', '_')  # Convert to string with microsecond precision

    # Step 1: Run Grounded SAM2 on `image` to get segmentation masks
    text = TEXT_PROMPT
    image_source, image = load_image(image)  # Convert OpenCV image to format for model
    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        remove_combined=True,
    )

    # Process bounding boxes
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # Get segmentation masks
    masks, _, _ = sam2_predictor.predict(
        point_coords=None, point_labels=None, box=input_boxes, multimask_output=False
    )

    # Convert masks to (num_classes, H, W) format
    if masks.ndim == 4:
        masks = masks.squeeze(1)  # Shape (num_classes, H, W)

    # Create a fixed class ID mapping
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
    num_classes = len(fixed_class_id_mapping)

    class_ids = np.array([
        fixed_class_id_mapping.get(class_name, num_classes)  # Use an "unknown" class ID if not found
        for class_name in labels
    ])
    
    ## Optional
    # Step 1.5: Save the combined mask for each image
    #save_dir = "saved_masks"
    #os.makedirs(save_dir, exist_ok=True)

    # Initialize a combined mask (same size as image)
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # Assign unique class IDs to the combined mask
    for class_idx, mask in zip(class_ids, masks):
        combined_mask[mask > 0] = class_idx + 1  # Avoid class ID 0 (reserved for background)

    # Convert mask to colored version for visualization
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

    # Ensure color values are in uint8 format
    for k in color_map:
        color_map[k] = np.array(color_map[k], dtype=np.uint8)

    # Initialize a colored mask (H, W, 3)
    #combined_mask_colored = np.zeros((h, w, 3), dtype=np.uint8)

    # Assign colors based on class index
    #for class_idx, color in color_map.items():
    #    combined_mask_colored[combined_mask == class_idx] = color

    # Save the combined mask using the timestamp
    #mask_filename = os.path.join(save_dir, f"mask_{timestamp_str}.png")
    #cv2.imwrite(mask_filename, combined_mask_colored)
    #print(f"Saved mask: {mask_filename}")

    # Step 2: Project LiDAR points onto the segmentation mask
    projected_pixels, depths, valid_camera_indices = project_lidar_to_image(lidar[:, :3], CAMERA_INTRINSICS)
    lidar = lidar[valid_camera_indices]  # Keep only valid points
    projected_pixels = projected_pixels[valid_camera_indices]

    # Step 3: Assign class labels based on projection
    img_h, img_w, _ = image.shape
    valid_indices = (
        (0 <= projected_pixels[:, 0]) & (projected_pixels[:, 0] < img_w) &
        (0 <= projected_pixels[:, 1]) & (projected_pixels[:, 1] < img_h)
    )

    # Initialize point labels (N, num_classes) → One-hot encoding
    point_labels = np.zeros((lidar.shape[0], num_classes), dtype=np.float32)

    for i, (u, v) in enumerate(projected_pixels):
        if combined_mask[int(v), int(u)] > 0:  # Check if point falls inside any mask
            class_idx = combined_mask[int(v), int(u)] - 1  # Subtract 1 to get correct class index
            point_labels[i, class_idx] = 1.0

    # Step 4: Convert LiDAR to SparseTensor format
    coords = np.round(lidar[:, :3] / res)
    coords -= coords.min(0, keepdims=1)

    coords, indices, inverse = sparse_quantize(coords, return_index=True, return_inverse=True)
    coords = torch.tensor(coords, dtype=torch.int)
    feats = torch.tensor(lidar[indices], dtype=torch.float)

    inputs = SparseTensor(coords=coords, feats=feats)
    inputs = sparse_collate([inputs]).cuda()

    return inputs, inverse, point_labels
