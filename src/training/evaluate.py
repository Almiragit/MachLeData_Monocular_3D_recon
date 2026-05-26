<<<<<<< HEAD
import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb

# ==========================================================
# 1. RESOLVE ABSOLUTE SYSTEM PATHS
# ==========================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

REPO_PATH = os.path.join(BASE_DIR, 'src', 'models', 'Depth-Anything-V2')
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)

from ipynb.fs.full.DAV2_Hybrid import load_hybrid_model
from train import NYUDataset, SILogLoss 

# ==========================================================
# 2. W&B CONFIGURATION (CRITICAL FOR MERGING FILES)
# ==========================================================
# ⚠️ REPLACE THIS STRING WITH YOUR ACTUAL 8-CHARACTER RUN ID FROM YOUR W&B OVERVIEW PAGE!
YOUR_RUN_ID = "YOUR_ACTUAL_RUN_ID_HERE" 

VAL_DATA_PATH = os.path.join(BASE_DIR, "src", "data", "data", "processed", "val")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running evaluation on device target: {device}")

# RESUME THE PREVIOUS RUN SAFELY
print(f"Connecting and merging into W&B run folder: hybrid-decoder-v1-run (ID: {YOUR_RUN_ID})")
run = wandb.init(
    project="Monocular-3D-Reconstruction",
    id=YOUR_RUN_ID,
    resume="allow"
)
=======
"""
src/training/evaluate.py
------------------------
Standalone evaluation script — run on val or test split after training.

Usage:
    python src/training/evaluate.py
    python src/training/evaluate.py --checkpoint artifacts/checkpoints/best_model.pth
    python src/training/evaluate.py --split test --log_wandb
"""

import sys
from pathlib import Path

# Ensure project root is importable when run as script: python src/training/evaluate.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import get_device, load_checkpoint, load_configs, set_seed  # noqa: E402
from src.training.train import NYUDataset  # noqa: E402
from src.models.losses import SILogLoss  # noqa: E402
from src.models.model import build_hybrid_model  # noqa: E402
import argparse
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


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
>>>>>>> 83dab2a3a58fb0b30202b6ff52cba402c35217c8

# ==========================================================
# 3. LOAD LOCAL WEIGHTS DIRECTLY (SMART SEARCH)
# ==========================================================
FILENAME = "latest_hybrid_model.pth"

<<<<<<< HEAD
# List of potential places the training loop might have dropped the file
possible_paths = [
    os.path.join(BASE_DIR, FILENAME),                          # Root folder
    os.path.join(BASE_DIR, "src", "training", FILENAME),       # src/training/
    os.path.join(os.path.dirname(__file__), FILENAME),         # Current directory
    os.path.join(BASE_DIR, "checkpoints", FILENAME)            # checkpoints folder
]

weights_path = None
for path in possible_paths:
    if os.path.exists(path):
        weights_path = path
        break

if weights_path is None:
    raise FileNotFoundError(
        f"Could not locate '{FILENAME}' automatically.\n"
        f"Please manually check your folders and move it to your root project folder:\n"
        f"-> C:\\Users\\kanha\\Documents\\MachLeData\\"
    )

print(f"🎯 Found weights file successfully at: {weights_path}")

# Load model structure and local weights
model = load_hybrid_model(encoder='vitb', device=device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval() 

# ==========================================================
# 4. PREPARE VALIDATION LOADER
# ==========================================================
val_dataset = NYUDataset(VAL_DATA_PATH)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
=======
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
    print(f"\n{'=' * 55}")
    print(f"  Evaluation on [{args.split}] split — {len(ds)} samples")
    print(f"{'=' * 55}")
    for k, v in metrics.items():
        print(f"  {k:<12} : {v:.4f}")
    print(f"{'=' * 55}\n")

    # ── Optional W&B log ─────────────────────────────────────────────────────
    if args.log_wandb:
        mode = "debug" if args.debug else "full"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        encoder = cfg["model"].get("encoder", "enc")
        run_name = (
            f"eval-{cfg['experiment']['name']}-{encoder}-{args.split}-{mode}-{ts}"
        )
        wandb.init(
            entity=cfg["experiment"].get("entity"),
            project=cfg["experiment"]["project"],
            job_type="evaluation",
            group=cfg["experiment"]["name"],
            name=run_name,
        )
        wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        wandb.finish()
        print("[INFO] Results logged to W&B")
>>>>>>> 83dab2a3a58fb0b30202b6ff52cba402c35217c8

criterion = SILogLoss()

# Metrics storage arrays
total_val_loss = 0
all_mae, all_rmse, all_abs_rel = [], [], []
all_delta1, all_delta2, all_delta3 = [], [], []

print(f"\n--- Starting evaluation over {len(val_dataset)} validation samples ---")

# ==========================================================
# 5. EVALUATION LOOP (With Live Line Trajectory Tracking)
# ==========================================================
with torch.no_grad():
    for batch_idx, (images, depths) in enumerate(tqdm(val_loader, desc="Evaluating")):
        images, depths = images.to(device), depths.to(device)
        
        # Predict pass
        outputs = model(images)
        outputs = torch.clamp(outputs, min=0.1, max=10.0)
        
        # 1. Compute validation loss
        loss = criterion(outputs, depths)
        total_val_loss += loss.item()
        
        # 2. Compute quantitative metrics pixel-by-pixel
        valid_mask = (depths > 0) & (depths <= 10.0)
        if not valid_mask.any():
            continue
            
        pred_valid = outputs[valid_mask]
        gt_valid = depths[valid_mask]
        
        # Metric formulas implementation
        all_mae.append(torch.mean(torch.abs(pred_valid - gt_valid)).item())
        all_rmse.append(torch.sqrt(torch.mean((pred_valid - gt_valid) ** 2)).item())
        all_abs_rel.append(torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid).item())
        
        # Threshold Accuracies (delta brackets)
        ratios = torch.max(pred_valid / gt_valid, gt_valid / pred_valid)
        all_delta1.append((ratios < 1.25).float().mean().item())
        all_delta2.append((ratios < 1.25 ** 2).float().mean().item())
        all_delta3.append((ratios < 1.25 ** 3).float().mean().item())

        # 🚀 LIVE LINE LOGGING: Every 5 batches, calculate the running average 
        # and push it to W&B to build a moving trajectory line on your charts.
        if batch_idx % 5 == 0:
            wandb.log({
                "val_loss_trajectory": total_val_loss / (batch_idx + 1),
                "val_RMSE_trajectory": np.mean(all_rmse),
                "val_MAE_trajectory": np.mean(all_mae),
                "val_delta2_trajectory": np.mean(all_delta2),
                "val_delta3_trajectory": np.mean(all_delta3),
            }, step=batch_idx)

# ==========================================================
# 6. FINAL SUMMARY REPORT
# ==========================================================
metrics = {
    "final_val_loss_SILog": total_val_loss / len(val_loader),
    "final_val_MAE": np.mean(all_mae),
    "final_val_RMSE": np.mean(all_rmse),
    "final_val_abs_rel": np.mean(all_abs_rel),
    "final_val_delta1": np.mean(all_delta1),
    "final_val_delta2": np.mean(all_delta2),
    "final_val_delta3": np.mean(all_delta3),
}

print("\n================ FINAL EVALUATION REPORT ================")
for name, value in metrics.items():
    print(f"📈 {name:<20}: {value:.4f}")
print("===========================================================")

# Log final overall summary markers
wandb.log(metrics)
run.finish()
print("\n🎉 Success! Evaluation complete. Check W&B for your new trajectory charts.")