import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import time
from typing import Union
from PIL import Image
from supervision.detection.core import Detections
"""
Hyper parameters
"""
# TEXT_PROMPT = "grass-shore. tree. building. tent. concreteplatform. water. redbuoy. blackbuoy. greenbuoy. orangebuoy. whitebuoy."
TEXT_PROMPT = "grass-shore. floating green patches on the water surface. water. blue-boat."
IMG_PATH = "notebooks/images_left/1735055673502660793.png"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.25#0.30
TEXT_THRESHOLD = 0.01#0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo_algae")
DUMP_JSON_RESULTS = True

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16

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

# def annotate_with_custom_colors(
#     scene: Union[np.ndarray, Image.Image],  # Supports both numpy and PIL images
#     detections: Detections,
#     custom_color_map: dict,
#     opacity: float = 0.5
# ) -> Union[np.ndarray, Image.Image]:  # Returns the same type as input
#     assert isinstance(scene, np.ndarray)
#     if detections.mask is None:
#         return scene

#     colored_mask = np.array(scene, copy=True, dtype=np.uint8)

#     for detection_idx in np.flip(np.argsort(detections.area)):
#         class_id = detections.class_id[detection_idx]
#         class_name = detections.class_names[class_id]
#         color = custom_color_map.get(class_name, (255, 255, 255))  # Default to white if not found
#         mask = detections.mask[detection_idx]
#         colored_mask[mask] = color  # Assign the specific color

#     cv2.addWeighted(colored_mask, opacity, scene, 1 - opacity, 0, dst=scene)
#     return scene


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
def process_image(img_path):
    text = TEXT_PROMPT
    img_path = img_path
    start_time = time.time()
    image_source, image = load_image(img_path)

    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        remove_combined=True,
    )
    end_time = time.time()
    print("gdino time: ", end_time - start_time)
    start_time = time.time()
    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


    # # FIXME: figure how does this influence the G-DINO model
    # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    # if torch.cuda.get_device_properties(0).major >= 8:
    #     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    end_time = time.time()
    # print("logits.shape: ", logits.shape)
    print("sam2 time: ", end_time - start_time)
    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)


    confidences = confidences.numpy().tolist()
    class_names = labels

    # class_ids = np.array(list(range(len(class_names))))
    # Define a fixed mapping of class names to IDs
    # fixed_class_id_mapping = {
    #     "grass - shore": 0,
    #     "tree": 1,
    #     "building": 2,
    #     "tent": 3,
    #     "concreteplatform": 4,
    #     "water": 5,
    #     "redbuoy": 6,
    #     "blackbuoy": 7,
    #     "greenbuoy": 8,
    #     "orangebuoy": 9,
    #     "whitebuoy": 10,
    # }
    fixed_class_id_mapping = {
        "grass - shore": 0,
        "floating green patches on the water surface": 1,
        "water": 2,
        "blue - boat": 3,
    }

    # Assign fixed IDs based on the mapping
    # class_ids = np.array([fixed_class_id_mapping[class_name] for class_name in class_names])
    class_ids = np.array([
    fixed_class_id_mapping.get(class_name, 4)  # Use 11 or any other default value for unknown classes
    for class_name in class_names
    ])

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]
    # print("class_names: ", class_names)
    """
    Visualize image with supervision useful API
    """
    img = cv2.imread(img_path)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )
    # print("class ids: ", class_ids)
    # print("detection.shape: ", detections)
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator(opacity=1.0)
    # annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)
    annotated_frame = mask_annotator.annotate(annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(img_path)), annotated_frame)
    # annotated_frame = annotate_with_custom_colors(
    # scene=img.copy(),
    # detections=detections,
    # custom_color_map=CLASS_COLOR_MAP,
    # opacity=0.5
    # )
    # cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(img_path)), annotated_frame)

    """
    Dump the results in standard format and save as json files
    """

    def single_mask_to_rle(mask):
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    if DUMP_JSON_RESULTS:
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes = input_boxes.tolist()
        scores = scores.tolist()
        # save the results in standard format
        results = {
            "image_path": img_path,
            "annotations" : [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }
        
        # with open(os.path.join(OUTPUT_DIR, "grounded_sam2_local_image_demo_results.json"), "w") as f:
        #     json.dump(results, f, indent=4)

# INPUT_DIR = "notebooks/images_left"
INPUT_DIR = "notebooks/algae_frames"
# Process all images in the input directory
for image_file in os.listdir(INPUT_DIR):
    if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(INPUT_DIR, image_file)
        print(img_path)
        process_image(img_path)
        # break