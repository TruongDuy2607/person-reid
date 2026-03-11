"""
Standalone benchmark script for SOLIDER-REID.

Evaluates a trained model on a custom dataset directory that follows the layout:
    <dataset_dir>/
        bounding_box_test/   (gallery)
        query/

Filename format (auto-detected):
    {pid}_{camid}_{...}.jpg / .png
    e.g.  00000_c021s0_549866.jpg
          0042_c1s1_1772593032625_04.jpg
          0003_c1_22.png

Outputs:
    mAP and Rank-1 (also Rank-5, Rank-10) printed to stdout.
"""

import argparse
import os
import re
import glob
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Allow imports from the SOLIDER-REID project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import cfg
from model import make_model
from utils.metrics import R1_mAP_eval



# Filename parser
# Matches the *first two* underscore-separated fields that look like
#   <digits>  _  c<digits>[optional-suffix]
# Examples:
#   00000_c021s0_549866.jpg   -> pid=0,    camid=21
#   0042_c1s1_1772593032625_04.jpg -> pid=42, camid=1
#   0003_c1_22.png            -> pid=3,    camid=1
_FILENAME_PATTERN = re.compile(r'^([-\d]+)_c(\d+)')


def parse_filename(filename: str):
    """
    Parse person-id and camera-id from a ReID image filename.

    Parameters
    ----------
    filename : str
        Basename of the image file (without directory).

    Returns
    -------
    pid : int
    camid : int  (0-based)

    Raises
    ------
    ValueError if the filename does not match the expected pattern.
    """
    m = _FILENAME_PATTERN.match(os.path.splitext(filename)[0])
    if m is None:
        raise ValueError(
            f"Cannot parse pid/camid from filename: '{filename}'. "
            "Expected format: <pid>_c<camid>[...].jpg"
        )
    pid = int(m.group(1))
    camid = int(m.group(2)) - 1  # convert to 0-based
    return pid, camid


# Dataset helpers
def load_split(directory: str, relabel: bool = False):
    """
    Load all .jpg / .png images from *directory* into a list of tuples.

    Parameters
    ----------
    directory : str
        Path to ``query/`` or ``bounding_box_test/``.
    relabel : bool
        When True, remap raw pids to contiguous 0-based labels (used for
        gallery / query — not needed here, but kept for consistency).

    Returns
    -------
    list of (img_path, pid, camid, trackid=0)
    """
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(directory, ext)))
    img_paths = sorted(img_paths)

    if not img_paths:
        raise RuntimeError(f"No images found in '{directory}'")

    pid_set = set()
    for p in img_paths:
        pid, _ = parse_filename(os.path.basename(p))
        if pid >= 0:
            pid_set.add(pid)

    pid2label = {pid: label for label, pid in enumerate(sorted(pid_set))}

    data = []
    for p in img_paths:
        pid, camid = parse_filename(os.path.basename(p))
        if pid < 0:           # junk images (pid == -1 in Market-style datasets)
            continue
        mapped_pid = pid2label[pid] if relabel else pid
        data.append((p, mapped_pid, camid, 0))  # trackid = 0

    return data


class ReidImageDataset(Dataset):
    """Thin wrapper around a list of (path, pid, camid, trackid) tuples."""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, torch.tensor(camid, dtype=torch.int64), \
               torch.tensor(0, dtype=torch.int64), img_path


# Evaluation
def build_transforms(cfg):
    """Return inference-time image transforms from config."""
    return T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])


def collate_fn(batch):
    imgs, pids, camids, camids_batch, target_view, img_paths = zip(*batch)
    camids_batch = torch.stack(camids_batch)
    target_view = torch.stack(target_view)
    return torch.stack(imgs), pids, camids, camids_batch, target_view, img_paths


def run_evaluation(model, val_loader, num_query, feat_norm, reranking, device):
    """
    Extract features and compute mAP / CMC.

    Parameters
    ----------
    model       : nn.Module (already on *device*)
    val_loader  : DataLoader  (query + gallery concatenated)
    num_query   : int
    feat_norm   : bool
    reranking   : bool
    device      : str  ('cuda' / 'cpu')

    Returns
    -------
    mAP   : float
    rank1 : float
    rank5 : float
    rank10: float
    """
    evaluator = R1_mAP_eval(num_query, max_rank=50,
                            feat_norm=feat_norm, reranking=reranking)
    evaluator.reset()
    model.eval()

    with torch.no_grad():
        for img, pid, camid, camids_batch, target_view, _ in val_loader:
            img = img.to(device)
            camids_batch = camids_batch.to(device)
            target_view = target_view.to(device)
            feat, _ = model(img, cam_label=camids_batch, view_label=target_view)
            evaluator.update((feat, pid, camid))

    cmc, mAP, *_ = evaluator.compute()
    return mAP, cmc[0], cmc[4], cmc[9]


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a SOLIDER-REID checkpoint on a custom dataset."
    )
    parser.add_argument(
        "--config_file", required=True, type=str,
        help="Path to the YACS config .yml file used during training."
    )
    parser.add_argument(
        "--weight", required=True, type=str,
        help="Path to the model checkpoint (.pth)."
    )
    parser.add_argument(
        "--dataset_dir", required=True, type=str,
        help="Root of the benchmark dataset. Must contain 'bounding_box_test/' "
             "and 'query/' sub-directories."
    )
    parser.add_argument(
        "--device", default="cuda:1", type=str, choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)."
    )
    parser.add_argument(
        "--batch_size", default=256, type=int,
        help="Batch size for feature extraction (default: 256)."
    )
    parser.add_argument(
        "--num_workers", default=4, type=int,
        help="Number of DataLoader workers (default: 4)."
    )
    parser.add_argument(
        "--reranking", action="store_true",
        help="Apply re-ranking post-processing."
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER,
        help="Optional config overrides, e.g. MODEL.SEMANTIC_WEIGHT 0.2"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # load config file
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.TEST.RE_RANKING = args.reranking
    # Prevent make_model from calling init_weights() with the training
    # checkpoint path — the full model is loaded via load_param() below.
    cfg.MODEL.PRETRAIN_PATH = ''
    cfg.freeze()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    gallery_dir = os.path.join(args.dataset_dir, "bounding_box_test")
    query_dir   = os.path.join(args.dataset_dir, "query")

    for d in (gallery_dir, query_dir):
        if not os.path.isdir(d):
            sys.exit(f"Directory not found: '{d}'")

    print(f"Loading query   : {query_dir}")
    print(f"Loading gallery : {gallery_dir}")

    query_data   = load_split(query_dir,   relabel=False)
    gallery_data = load_split(gallery_dir, relabel=False)
    num_query    = len(query_data)

    transforms = build_transforms(cfg)
    val_dataset = ReidImageDataset(query_data + gallery_data, transforms)
    val_loader  = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    # compute unique camera count for model init
    all_camids = set(c for _, _, c, _ in query_data + gallery_data)
    camera_num = len(all_camids)

    print(f"\nDataset summary:")
    print(f"  query   : {num_query} images")
    print(f"  gallery : {len(gallery_data)} images")
    print(f"  cameras : {camera_num}")

    num_classes = len(set(pid for _, pid, _, _ in gallery_data))
    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=1,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    model.load_param(args.weight)
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    print(f"\nLoaded weights : {args.weight}")
    
    print("\nRunning evaluation ...")
    mAP, rank1, rank5, rank10 = run_evaluation(
        model, val_loader, num_query,
        feat_norm=(cfg.TEST.FEAT_NORM == "yes"),
        reranking=cfg.TEST.RE_RANKING,
        device=device,
    )

    print("\n----------- Results -----------")
    print(f"  mAP    : {mAP:.2%}")
    print(f"  Rank-1 : {rank1:.2%}")
    print(f"  Rank-5 : {rank5:.2%}")
    print(f"  Rank-10: {rank10:.2%}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
