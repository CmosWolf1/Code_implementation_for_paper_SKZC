# Copyright (c) Facebook, Inc. and its affiliates.
from PIL import Image
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from diffusiondet.predictor import VisualizationDemo
from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def crop_images(img, coordinates):
    """
    Crop images based on given coordinates.

    Parameters:
    image_path (str): The path to the image file.
    coordinates (list of tuples): A list of coordinates where each tuple
                                  consists of (left, upper, right, lower) pixel coordinates.

    Returns:
    list: A list of cropped image objects.
    """

    # Initialize output list for cropped image objects
    cropped_images = []

    # Loop over each set of coordinates and crop the image.
    # Append the resulting image object to our list.
    for (left, upper, right, lower) in coordinates:
        if left < 0 or upper < 0 or right > img.shape[1] or lower > img.shape[0]:
            raise ValueError("Invalid coordinates")
        cropped_images.append(img[upper:lower, left:right])

    # Return the list of cropped images
    return cropped_images


mp.set_start_method("spawn", force=True)
args = get_parser().parse_args()
setup_logger(name="fvcore")
logger = setup_logger()
logger.info("Arguments: " + str(args))

cfg = setup_cfg(args)

demo = VisualizationDemo(cfg)
cropped_images = []
for filename in os.listdir(args.input[0]):
    if filename.endswith(('.png', '.jpg', 'jpeg')):
        img_path = os.path.join(args.input[0], filename)
        # use PIL, to be consistent with evaluation
        img = read_image(img_path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        coordinates = predictions['instances']._fields['pred_boxes'].tensor.to(dtype=int).tolist()
        coordinates = [tuple(inner_list) for inner_list in coordinates]
        cropped_images.extend(crop_images(img, coordinates))

for i, cropped_image in enumerate(cropped_images):
    cv2.imwrite(f"crop_result/cropped_{i}.jpg", cropped_image)
