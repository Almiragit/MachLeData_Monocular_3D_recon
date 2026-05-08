"""
src/data/dataset.py
-------------------
PyTorch Dataset for loading (RGB image, depth map) pairs.
Works with processed data created by preprocess.py.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ─── Transforms ───────────────────────────────────────────────────────────────
def get_transforms(split: str, img_size: tuple[int, int] = (480, 640)):
    """
    Return torchvision transforms for *split* (train | val | test).
    No random augmentation during val/test.
    """
    H, W = img_size

    rgb_train = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    rgb_eval = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    depth_transform = transforms.Compose([
        transforms.Resize((H, W), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),   # → (1, H, W) float32 in [0,1]
    ])

    if split == "train":
        return rgb_train, depth_transform
    return rgb_eval, depth_transform


# ─── Dataset ──────────────────────────────────────────────────────────────────
class DepthDataset(Dataset):
    """
    Loads (RGB, depth) pairs from the processed data directory.

    Expected directory layout:
        processed_dir/
            {split}/
                rgb/   *.jpg or *.png
                depth/ *.png  (same filename as rgb)

    Alternatively pass a split JSON (from preprocess.py) via *split_json*.
    """

    def __init__(
        self,
        processed_dir: str,
        split: str,
        img_size: tuple[int, int] = (480, 640),
        split_json: str | None = None,
        depth_scale: float = 1000.0,   # NYU: depth stored in mm → metres
    ):
        self.split = split
        self.depth_scale = depth_scale
        self.rgb_transform, self.depth_transform = get_transforms(split, img_size)

        if split_json and os.path.exists(split_json):
            with open(split_json, "r") as f:
                all_splits = json.load(f)
            self.pairs = all_splits[split]
        else:
            rgb_dir = Path(processed_dir) / split / "rgb"
            depth_dir = Path(processed_dir) / split / "depth"
            self.pairs = [
                {"rgb": str(p), "depth": str(depth_dir / p.name)}
                for p in sorted(rgb_dir.glob("*"))
                if (depth_dir / p.name).exists()
            ]

        print(f"[Dataset] {split}: {len(self.pairs)} samples loaded")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]

        rgb = Image.open(pair["rgb"]).convert("RGB")
        depth = Image.open(pair["depth"])

        rgb_t = self.rgb_transform(rgb)           # (3, H, W)
        depth_t = self.depth_transform(depth)     # (1, H, W) in [0,1]

        # Convert to metres (assuming stored as uint16 mm values mapped to float)
        depth_t = depth_t.float() / self.depth_scale

        return rgb_t, depth_t


# ─── DataLoader factory ───────────────────────────────────────────────────────
def build_dataloaders(
    processed_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: tuple[int, int] = (480, 640),
    split_json: str | None = None,
) -> dict[str, DataLoader]:
    """Return {'train': ..., 'val': ..., 'test': ...} DataLoaders."""
    loaders = {}
    for split in ("train", "val", "test"):
        ds = DepthDataset(
            processed_dir=processed_dir,
            split=split,
            img_size=img_size,
            split_json=split_json,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
