"""
src/training/train.py
--------------------
Stage 2 – Model Training: Fine-tune DaV2 Hybrid Decoder on NYU Depth V2.

Architecture:
  - DINOv2 Encoder (frozen) – extracts visual features
  - DPTHead Decoder (trainable) – predicts depth map

Usage:
    python src/training/train.py
    python src/training/train.py --epochs 20 --lr 5e-5
    python src/training/train.py --debug        # 2 epochs, 50 samples
"""

from src.utils import get_device, load_configs, set_seed
from src.models.losses import SILogLoss
from src.models.model import build_hybrid_model
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ─── NYU Dataset (portable, no ipynb dependency) ─────────────────────────────
class NYUDataset(torch.utils.data.Dataset):
    """
    Loads preprocessed NYU Depth V2 data saved as .pt files.
    Expected structure:
        data_dir/
            file_000.pt
            file_001.pt
            ...
    Each .pt file contains:
        {'image': np.ndarray (H, W, 3), 'depth': np.ndarray (H, W)}
    """

    def __init__(self, data_dir: str, img_size: int = 518):
        self.data_dir = data_dir
        self.img_size = img_size
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        self.files = sorted([
            f for f in os.listdir(data_dir) if f.endswith('.pt')
        ])
        print(f"[NYUDataset] Loaded {len(self.files)} samples from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(path, map_location='cpu', weights_only=False)

        # Image: (H, W, C) → (C, H, W), normalize to [0,1]
        image = torch.from_numpy(data['image']).permute(
            2, 0, 1).float() / 255.0

        # Normalize using ImageNet stats (DaV2 requirement)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        # Resize to model input size
        image = nn.functional.interpolate(
            image.unsqueeze(0), size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)

        # Depth: (H, W) → (1, H, W), resize to match
        depth = torch.from_numpy(data['depth']).float().unsqueeze(0)
        depth = nn.functional.interpolate(
            depth.unsqueeze(0), size=(self.img_size, self.img_size),
            mode='nearest'
        ).squeeze(0)

        return image, depth


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")

    for images, depths in pbar:
        images, depths = images.to(device), depths.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Clamp to prevent log(0) in SILogLoss
        outputs = torch.clamp(outputs, min=0.1, max=10.0)

        loss = criterion(outputs, depths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        wandb.log({"batch_loss": loss.item()})
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    return avg_loss


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for images, depths in loader:
        images, depths = images.to(device), depths.to(device)
        outputs = model(images)
        outputs = torch.clamp(outputs, min=0.1, max=10.0)
        loss = criterion(outputs, depths)
        total_loss += loss.item()
    return total_loss / len(loader)


# ─── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune DaV2 Hybrid Decoder")
    p.add_argument("--paths_config", default="configs/paths.yaml")
    p.add_argument("--train_config", default="configs/train.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--debug", action="store_true",
                   help="2 epochs, 50 samples max")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_configs(args.paths_config, args.train_config)

    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr

    set_seed(42)
    device = get_device()

    # ── W&B run naming/grouping ───────────────────────────────────────────────
    mode = "debug" if args.debug else "full"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    encoder = cfg["model"].get("encoder", "enc")
    run_name = (
        f"train-{cfg['experiment']['name']}-{encoder}"
        f"-e{cfg['training']['epochs']}-bs{cfg['training']['batch_size']}"
        f"-lr{cfg['training']['learning_rate']}-{mode}-{ts}"
    )
    run_group = cfg["experiment"]["name"]

    # ── W&B ──────────────────────────────────────────────────────────────────
    run = wandb.init(
        entity=cfg["experiment"].get("entity"),
        project=cfg["experiment"]["project"],
        job_type="training",
        group=run_group,
        name=run_name,
        tags=cfg["experiment"]["tags"] + ["finetune", "hybrid"],
        config={
            "epochs": cfg["training"]["epochs"],
            "batch_size": cfg["training"]["batch_size"],
            "learning_rate": cfg["training"]["learning_rate"],
            "freeze_encoder": cfg["training"].get("freeze_encoder", True),
            "debug": args.debug,
        },
    )
    print(f"[W&B] Run: {run.url}")

    # ── Data ─────────────────────────────────────────────────────────────────
    data_root = cfg["data"].get("nyu_processed", "data/nyu/processed")
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_ds = NYUDataset(train_dir)
    val_ds = NYUDataset(val_dir)

    if args.debug:
        import random
        train_ds.files = random.sample(
            train_ds.files, min(50, len(train_ds.files)))
        val_ds.files = val_ds.files[:20]
        cfg["training"]["epochs"] = 2
        print(
            f"[DEBUG] Training on {len(train_ds)} samples, {cfg['training']['epochs']} epochs")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=0
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    freeze_encoder = cfg["training"].get("freeze_encoder", True)
    model = build_hybrid_model(
        encoder=cfg["model"]["encoder"],
        checkpoint_dir=cfg["model"]["checkpoint_dir"],
        device=device,
        freeze_encoder=freeze_encoder,
    )

    optimizer = optim.Adam(
        model.custom_decoder.parameters(),
        lr=cfg["training"]["learning_rate"]
    )
    criterion = SILogLoss()

    # ── Training Loop ──────────────────────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    best_val_loss = float("inf")
    checkpoint_dir = Path(cfg["artifacts"]["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Starting training for {epochs} epochs")
    print(f"[INFO] Encoder frozen: {freeze_encoder}")
    print(
        f"[INFO] Trainable params: {sum(p.numel() for p in model.custom_decoder.parameters()):,}")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, epochs
        )
        val_loss = validate(model, val_loader, criterion, device)

        print(f"  → Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, best_path)
            print(f"  ✓ Best model saved: {best_path}")

        # Save latest checkpoint every epoch
        latest_path = checkpoint_dir / "latest_model.pth"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, latest_path)

    # ── Log model artifact to W&B ──────────────────────────────────────────────
    artifact = wandb.Artifact(
        name=f"{cfg['experiment']['name']}-finetuned",
        type="model",
        description="Fine-tuned DaV2 Hybrid Decoder on NYU Depth V2",
        metadata={"best_val_loss": best_val_loss, "epochs": epochs},
    )
    artifact.add_file(str(best_path))
    run.log_artifact(artifact)

    wandb.finish()
    print(f"\n[DONE] Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
