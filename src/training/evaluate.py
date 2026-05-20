"""
src/training/evaluate.py
------------------------
Standalone evaluation script — run on val or test split after training.

Usage:
    python src/training/evaluate.py
    python src/training/evaluate.py --checkpoint artifacts/checkpoints/best_model.pth
    python src/training/evaluate.py --split test --log_wandb
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.model import build_hybrid_model
from src.models.losses import SILogLoss
from src.training.train import NYUDataset
from src.utils import get_device, load_checkpoint, load_configs, set_seed


# ─── Metrics ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute standard monocular depth evaluation metrics.

    Args:
        pred: (B, 1, H, W) predicted depth
        target: (B, 1, H, W) ground truth depth

    Returns:
        dict with rmse, mae, abs_rel, delta1, delta2, delta3
    """
    mask = (target > 0) & (target <= 10.0)
    if not mask.any():
        return {}

    p = pred[mask]
    t = target[mask]

    # Threshold accuracy
    ratio = torch.max(p / t, t / p)
    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    # Error metrics
    abs_diff = torch.abs(p - t)
    rmse = torch.sqrt((abs_diff ** 2).mean()).item()
    mae = abs_diff.mean().item()
    abs_rel = (abs_diff / t).mean().item()

    return {
        "rmse": rmse,
        "mae": mae,
        "abs_rel": abs_rel,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, debug: bool = False) -> dict:
    """Run evaluation on a DataLoader, return averaged metrics."""
    model.eval()
    all_metrics = []
    total_loss = 0

    n_batches = 5 if debug else len(loader)
    loader_iter = tqdm(loader, desc="Evaluating", total=n_batches)

    for i, (images, depths) in enumerate(loader_iter):
        if i >= n_batches:
            break

        images, depths = images.to(device), depths.to(device)
        outputs = model(images)
        outputs = torch.clamp(outputs, min=0.1, max=10.0)

        loss = criterion(outputs, depths)
        total_loss += loss.item()

        metrics = compute_metrics(outputs, depths)
        if metrics:
            all_metrics.append(metrics)

    # Average metrics
    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    avg["val_loss"] = total_loss / n_batches
    return avg


# ─── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned depth model")
    p.add_argument("--checkpoint", type=str,
                   default="artifacts/checkpoints/best_model.pth")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--paths_config", default="configs/paths.yaml")
    p.add_argument("--train_config", default="configs/train.yaml")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--log_wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_configs(args.paths_config, args.train_config)

    set_seed(42)
    device = get_device()

    # ── Data ─────────────────────────────────────────────────────────────────
    data_root = cfg["data"].get("nyu_processed", "data/nyu/processed")
    split_dir = os.path.join(data_root, args.split)
    print(f"[INFO] Evaluating on: {split_dir}")

    ds = NYUDataset(split_dir)
    if args.debug:
        ds.files = ds.files[:20]

    loader = DataLoader(
        ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=0
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_hybrid_model(
        encoder=cfg["model"]["encoder"],
        checkpoint_dir=cfg["model"].get("checkpoint_dir"),
        device=device,
        freeze_encoder=False,  # load full model (encoder + decoder)
    )
    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    print(f"[INFO] Loaded weights from: {args.checkpoint}")

    criterion = SILogLoss()

    # ── Evaluate ─────────────────────────────────────────────────────────────
    metrics = evaluate(model, loader, criterion, device, debug=args.debug)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Evaluation on [{args.split}] split — {len(ds)} samples")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:<12} : {v:.4f}")
    print(f"{'='*55}\n")

    # ── Optional W&B log ─────────────────────────────────────────────────────
    if args.log_wandb:
        mode = "debug" if args.debug else "full"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        encoder = cfg["model"].get("encoder", "enc")
        run_name = (
            f"eval-{cfg['experiment']['name']}-{encoder}-{args.split}-{mode}-{ts}"
        )
        run = wandb.init(
            entity=cfg["experiment"].get("entity"),
            project=cfg["experiment"]["project"],
            job_type="evaluation",
            group=cfg["experiment"]["name"],
            name=run_name,
        )
        wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        wandb.finish()
        print("[INFO] Results logged to W&B")


if __name__ == "__main__":
    main()