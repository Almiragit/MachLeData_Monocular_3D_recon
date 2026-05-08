"""
src/training/evaluate.py
------------------------
Standalone evaluation script — run on val or test split.

Usage:
    python src/training/evaluate.py --checkpoint artifacts/checkpoints/best_model.pth
    python src/training/evaluate.py --checkpoint artifacts/checkpoints/best_model.pth --split test
"""

import argparse
import sys
from pathlib import Path

import torch
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.dataset import build_dataloaders
from src.models.model import build_model
from src.utils import (
    compute_depth_metrics,
    get_device,
    load_checkpoint,
    load_configs,
    set_seed,
)


def evaluate(model, loader, device, debug: bool = False) -> dict:
    model.eval()
    all_metrics = []
    n_batches = 5 if debug else len(loader)

    with torch.no_grad():
        for i, (rgb, depth) in enumerate(loader):
            if i >= n_batches:
                break
            rgb = rgb.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)
            pred = model(rgb)
            all_metrics.append(compute_depth_metrics(pred, depth))

    avg = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
    return avg


def main():
    parser = argparse.ArgumentParser(description="Evaluate depth model")
    parser.add_argument("--checkpoint", type=str,
                        default="artifacts/checkpoints/best_model.pth")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"])
    parser.add_argument("--paths_config", type=str, default="configs/paths.yaml")
    parser.add_argument("--train_config", type=str, default="configs/train.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log_wandb", action="store_true",
                        help="Log results to W&B (requires active run or will create one)")
    args = parser.parse_args()

    cfg = load_configs(args.paths_config, args.train_config)
    set_seed(cfg["training"]["seed"])
    device = get_device()

    # Load model
    model = build_model(cfg["model"]["architecture"]).to(device)
    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[INFO] Loaded weights from: {args.checkpoint}")

    # Build loader for desired split
    loaders = build_dataloaders(
        processed_dir=cfg["data"]["processed"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        img_size=(cfg["model"]["img_height"], cfg["model"]["img_width"]),
        split_json=cfg["splits"]["json_path"],
    )

    metrics = evaluate(model, loaders[args.split], device, debug=args.debug)

    # Print results
    print(f"\n{'='*50}")
    print(f"  Evaluation on [{args.split}] split")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<12} : {v:.4f}")
    print(f"{'='*50}\n")

    # Optional W&B log
    if args.log_wandb:
        run = wandb.init(
            project=cfg["experiment"]["project"],
            job_type="evaluation",
            name=f"eval_{args.split}",
        )
        wandb.log({f"eval_{args.split}/{k}": v for k, v in metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
