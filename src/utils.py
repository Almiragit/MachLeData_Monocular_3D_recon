"""
src/utils.py
------------
Shared utilities used across all pipeline stages.
"""

import os
import random
import time

import numpy as np
import torch
import yaml


# ─── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[utils] Seed fixed to {seed}")


# ─── Config ───────────────────────────────────────────────────────────────────
def load_yaml(path: str) -> dict:
    """Load a YAML config file and return as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_configs(paths_yaml: str = "configs/paths.yaml",
                 train_yaml: str = "configs/train.yaml") -> dict:
    """Merge paths + training configs into one dict."""
    cfg = load_yaml(paths_yaml)
    cfg.update(load_yaml(train_yaml))
    return cfg


# ─── Device ───────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Return CUDA if available, else MPS (Apple Silicon), else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[utils] Using device: {device}")
    return device


# ─── Checkpointing ────────────────────────────────────────────────────────────
def save_checkpoint(
    state: dict,
    filepath: str,
    is_best: bool = False,
    best_filepath: str | None = None,
) -> None:
    """Save a training checkpoint. Optionally copy to best_filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    if is_best and best_filepath:
        os.makedirs(os.path.dirname(best_filepath), exist_ok=True)
        import shutil
        shutil.copyfile(filepath, best_filepath)
        print(f"[utils] ✓ Best checkpoint saved → {best_filepath}")


def load_checkpoint(filepath: str, device: torch.device) -> dict:
    """Load a checkpoint dict from *filepath*."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    print(f"[utils] Loaded checkpoint: {filepath}")
    return checkpoint


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """
    Compute standard monocular depth evaluation metrics.
    Both *pred* and *target* should be float tensors with shape (B, 1, H, W)
    and values in metres (or a consistent unit).
    """
    # Avoid division by zero
    mask = target > 0

    pred_m = pred[mask]
    target_m = target[mask]

    # Threshold accuracy δ1: % of pixels where max(pred/gt, gt/pred) < 1.25
    ratio = torch.max(pred_m / target_m, target_m / pred_m)
    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    abs_diff = torch.abs(pred_m - target_m)
    rmse = torch.sqrt((abs_diff ** 2).mean()).item()
    mae = abs_diff.mean().item()
    abs_rel = (abs_diff / target_m).mean().item()

    return {
        "rmse": rmse,
        "mae": mae,
        "abs_rel": abs_rel,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


# ─── Timing ───────────────────────────────────────────────────────────────────
class Timer:
    """Simple context-manager timer."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        return f"{self.elapsed:.3f}s"
