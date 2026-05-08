"""
src/data/preprocess.py
----------------------
Stage 1 – Data Preparation: Clean, resize, and split dataset into
train / val / test sets. Outputs are DVC-tracked.

Usage:
    python src/data/preprocess.py --config configs/paths.yaml
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


# ─── Constants ────────────────────────────────────────────────────────────────
SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
TARGET_SIZE = (480, 640)   # (H, W) — standard for NYU Depth V2


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def collect_image_pairs(raw_dir: str) -> list[dict]:
    """
    Walk *raw_dir* and collect (rgb, depth) file pairs.
    Expects layout:
        raw_dir/
            rgb/   *.jpg or *.png
            depth/ *.png  (same stem as rgb)
    """
    rgb_dir = Path(raw_dir) / "rgb"
    depth_dir = Path(raw_dir) / "depth"

    if not rgb_dir.exists() or not depth_dir.exists():
        raise FileNotFoundError(
            f"Expected sub-dirs 'rgb/' and 'depth/' inside {raw_dir}.\n"
            "Adapt this script to match your dataset layout."
        )

    pairs = []
    for rgb_path in sorted(rgb_dir.glob("*")):
        if rgb_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        depth_path = depth_dir / (rgb_path.stem + ".png")
        if depth_path.exists():
            pairs.append({"rgb": str(rgb_path), "depth": str(depth_path)})

    print(f"[INFO] Found {len(pairs)} valid (rgb, depth) pairs in {raw_dir}")
    return pairs


def split_pairs(pairs: list[dict], seed: int = SEED) -> dict:
    """Shuffle and split into train / val / test."""
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train: n_train + n_val],
        "test": pairs[n_train + n_val:],
    }


def process_and_save(pairs: list[dict], out_dir: str, split_name: str) -> None:
    """Resize images and save into out_dir/split_name/{rgb,depth}/."""
    rgb_out = Path(out_dir) / split_name / "rgb"
    depth_out = Path(out_dir) / split_name / "depth"
    rgb_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    for i, pair in enumerate(pairs):
        # RGB
        img = Image.open(pair["rgb"]).convert("RGB")
        img = img.resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.BILINEAR)
        img.save(rgb_out / Path(pair["rgb"]).name)

        # Depth
        depth = Image.open(pair["depth"])
        depth = depth.resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.NEAREST)
        depth.save(depth_out / Path(pair["depth"]).name)

        if (i + 1) % 200 == 0 or (i + 1) == len(pairs):
            print(f"  [{split_name}] {i+1}/{len(pairs)} processed")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess depth dataset")
    parser.add_argument("--config", type=str, default="configs/paths.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg["data"]["raw"]
    processed_dir = cfg["data"]["processed"]
    split_json = cfg["splits"]["json_path"]

    # 1. Collect pairs
    pairs = collect_image_pairs(raw_dir)

    # 2. Split
    splits = split_pairs(pairs)
    for name, subset in splits.items():
        print(f"  {name}: {len(subset)} samples")

    # 3. Save split JSON (create once, never regenerate)
    os.makedirs(os.path.dirname(split_json), exist_ok=True)
    if os.path.exists(split_json):
        print(f"[INFO] Split file already exists — reusing: {split_json}")
    else:
        with open(split_json, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"[INFO] Saved split file: {split_json}")

    # 4. Process and save
    print("[INFO] Processing and saving images…")
    for split_name, subset in splits.items():
        process_and_save(subset, processed_dir, split_name)

    print(f"\n[DONE] Processed data saved to: {processed_dir}")


if __name__ == "__main__":
    main()
