"""
Common experiment profile and paths.

Every train / evaluate script of every detector reads its settings
from here. Only the algorithm (detectors/<version>/) differs between runs.

Paths can be moved with one environment variable:
    export PHYSICAL_AI_ROOT=~/Physical_AI
"""

import os
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_AI_ROOT = Path(os.environ.get("PHYSICAL_AI_ROOT", "~/Physical_AI")).expanduser()
DATASETS_ROOT = PHYSICAL_AI_ROOT / "datasets"

VOC_ROOT = DATASETS_ROOT / "VOC" / "VOCdevkit" / "VOC2007"
RUNS_ROOT = FRAMEWORK_ROOT / "runs"

# ------------------------------------------------------------------
# OD-Small Benchmark Profile v1.0  (Mode A: controlled comparison)
# ------------------------------------------------------------------
PROFILE = {
    "name": "OD-Small v1.1",       # v1.1: Image training augmentation of the YOLO papers (image_aug)
    "seed": 777,
    "epochs": 30,   # [Release] FashionLetters36 validation: 30 epochs
    "optimizer": "Adam",           # decided 2026-09-28 (legacy Complex/YOLO3D optimizer)
    "lr": 1e-4,   # [Release] FashionLetters36 validation: Adam 1e-4
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "lr_schedule": "constant",     # legacy StepLR(10) never fires within 10 epochs
    "effective_batch": 8,          # micro batch x gradient accumulation
    "hflip_prob": 0.5,             # common augmentation for every task
    # [Unified] profile v1.1 (2026-09-30): basic augmentation of the YOLOv1 / v2 papers for the Image
    # track (10 epochs from scratch with flip only memorised the training set: train cls acc 99.6 %,
    # test mAP50 < 0.04 in the 1/4-VOC convergence study). KITTI keeps its point-cloud augmentation.
    "image_aug": {"jitter": 0.2, "hue": 0.1, "saturation": 1.5, "exposure": 1.5},
    "eval_iou": 0.5,               # AP50 / mAP50
    "eval_conf": 0.01,             # low threshold -> full PR curve
    "nms_iou": 0.5,                # same NMS threshold for every task
    "pre_nms_topk": 300,
    "max_det": 100,
}

# Mode B (reference recipe): legacy optimizer of each project.
REFERENCE_RECIPE = {
    "image":   {"optimizer": "SGD",  "lr": 2.5e-4, "lr_schedule": "constant"},
}

# ------------------------------------------------------------------
# Task definitions (dataset side only - no algorithm here)
# ------------------------------------------------------------------
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
# [Unified] Image dataset selection (2026-10-01). Algorithm validation runs on FashionLetters36
# (data_processing_fl36.py: Fashion-MNIST 10 + EMNIST letters 26 classes, 4,096 / 1,024 images,
# VOC layout); benchmarking runs on VOC2007. Select with UNIFIED_IMAGE_DATASET=fl36 (default voc2007).
IMAGE_DATASET = os.environ.get("UNIFIED_IMAGE_DATASET", "fl36")
FL36_CLASSES = ("tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot",
                *[chr(ord("A") + i) for i in range(26)])
FL36_ROOT = DATASETS_ROOT / "FashionLetters36"
if IMAGE_DATASET == "fl36":
    VOC_ROOT, VOC_CLASSES = FL36_ROOT, FL36_CLASSES
elif IMAGE_DATASET != "voc2007":
    raise ValueError(f"UNIFIED_IMAGE_DATASET must be voc2007 or fl36, not {IMAGE_DATASET!r}")


TASKS = {
    "image": {
        "dataset": IMAGE_DATASET,
        "classes": VOC_CLASSES,
        "img_size": 608,             # decided 2026-09-28 (same as the BEV map)
        "micro_batch": 2,            # x accum 4 = effective 8 (fits 8 GB)
        "train_split": "trainval",   # 5,011 images
        "test_split": "test",        # 4,952 images
    },
}


def run_dir(task, version):
    return RUNS_ROOT / f"{task}_{version}"


def checkpoint_path(task, version, epochs):
    return run_dir(task, version) / "checkpoints" / f"{task}_{version}_{TASKS[task]['dataset']}_e{epochs:03d}.pth"
