"""
src/training/train.py
---------------------
Stage 2 – Model Development & Experimentation.

Since Depth-Anything-V2 is a pretrained model used for INFERENCE (not trained
from scratch), this script:
  1. Loads the DaV2-vitb model
  2. Runs inference on the full dataset split
  3. Logs metrics (depth stats, inference speed) to W&B
  4. Saves depth maps and generated point clouds as W&B artifacts
  5. Writes a CSV metrics log

Think of this as the "experiment run" stage — you can swap configs
(vits / vitb / vitl, different input_sizes) and compare them in W&B.

Usage:
    python src/training/train.py
    python src/training/train.py --encoder vitl --input_size 518
    python src/training/train.py --debug      # 5 images only
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.model import build_model
from src.utils import Timer, get_device, load_configs, set_seed


# ─── Point cloud builder (from DaV2_PtCloud.ipynb) ───────────────────────────
def create_point_cloud(depth_norm: np.ndarray, rgb_image: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Build an Open3D PointCloud from a normalised depth map and RGB image.
    Matches the implementation in DaV2_PtCloud.ipynb.
    """
    h, w = depth_norm.shape
    fx = fy = 500   # focal-length approximation (no camera intrinsics available)
    cx, cy = w / 2, h / 2

    x = np.arange(0, w)
    y = np.arange(0, h)
    xx, yy = np.meshgrid(x, y)

    z = depth_norm / 255.0
    valid = z > 0

    X = (xx - cx) * z / fx
    Y = (yy - cy) * z / fy
    Z = z

    points = np.stack([X, Y, Z], axis=-1)[valid]
    colors = (rgb_image / 255.0)[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# ─── Run inference on a folder of images ─────────────────────────────────────
def run_inference_on_split(
    model,
    image_dir: str,
    output_dir: str,
    cfg: dict,
    debug: bool = False,
    debug_n: int = 5,
) -> list[dict]:
    """
    Iterate over all images in *image_dir*, run DaV2 inference,
    save depth maps and point clouds, return per-image metrics.
    """
    input_size = cfg["model"]["input_size"]
    encoder    = cfg["model"]["encoder"]

    depth_dir = Path(output_dir) / "depth_maps"
    pcd_dir   = Path(output_dir) / "point_clouds"
    depth_dir.mkdir(parents=True, exist_ok=True)
    pcd_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([
        p for p in Path(image_dir).rglob("*") if p.suffix.lower() in exts
    ])

    if not image_paths:
        print(f"[WARN] No images found in {image_dir}")
        return []

    if debug:
        image_paths = image_paths[:debug_n]
        print(f"[DEBUG] Running on {len(image_paths)} images only")

    print(f"[INFO] Processing {len(image_paths)} images with DaV2-{encoder}")

    results = []
    for i, img_path in enumerate(image_paths):
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            print(f"  [SKIP] Could not read: {img_path}")
            continue

        t0 = time.perf_counter()
        depth = model.infer_image(raw_bgr, input_size=input_size)
        inference_ms = (time.perf_counter() - t0) * 1000

        # Normalize depth to [0, 255]
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)

        # Save depth map as PNG
        stem = img_path.stem
        cv2.imwrite(str(depth_dir / f"{stem}_depth.png"), depth_norm)

        # Save colourised depth
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(depth_dir / f"{stem}_depth_color.png"), depth_color)

        # Build & save point cloud
        rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        pcd = create_point_cloud(depth_norm, rgb)
        pcd_path = str(pcd_dir / f"{stem}.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)

        result = {
            "image": img_path.name,
            "depth_min": float(d_min),
            "depth_max": float(d_max),
            "depth_mean": float(depth.mean()),
            "depth_std": float(depth.std()),
            "inference_ms": round(inference_ms, 1),
            "n_points": int(np.sum((depth_norm / 255.0) > 0)),
        }
        results.append(result)

        if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {img_path.name} — {inference_ms:.0f}ms")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paths_config", default="configs/paths.yaml")
    p.add_argument("--train_config", default="configs/train.yaml")
    p.add_argument("--encoder", type=str, default=None,
                   help="Override encoder: vits | vitb | vitl")
    p.add_argument("--input_size", type=int, default=None,
                   help="Override DaV2 inference input size (default 518)")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val", "test"],
                   help="Which data split to run inference on")
    p.add_argument("--debug", action="store_true",
                   help="Run on only 5 images for quick sanity check")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_configs(args.paths_config, args.train_config)

    # CLI overrides
    if args.encoder:
        cfg["model"]["encoder"] = args.encoder
    if args.input_size:
        cfg["model"]["input_size"] = args.input_size

    set_seed(42)
    device = get_device()

    # ── W&B ──────────────────────────────────────────────────────────────────
    run = wandb.init(
        project=cfg["experiment"]["project"],
        name=f"dav2-{cfg['model']['encoder']}-{args.split}" + ("-DEBUG" if args.debug else ""),
        tags=cfg["experiment"]["tags"] + [f"encoder:{cfg['model']['encoder']}", args.split],
        config={
            "encoder": cfg["model"]["encoder"],
            "input_size": cfg["model"]["input_size"],
            "split": args.split,
            "debug": args.debug,
        },
    )
    print(f"[W&B] Run: {run.url}")

    # ── Load DaV2 model ───────────────────────────────────────────────────────
    model = build_model(
        architecture=f"dav2_{cfg['model']['encoder']}",
        checkpoint_dir=cfg["model"]["checkpoint_dir"],
        device=device,
    )

    # ── Run inference ──────────────────────────────────────────────────────────
    image_dir = str(Path(cfg["data"]["processed"]) / args.split / "rgb")
    output_dir = str(Path(cfg["artifacts"]["outputs"]) / args.split)

    with Timer() as t:
        results = run_inference_on_split(
            model=model,
            image_dir=image_dir,
            output_dir=output_dir,
            cfg=cfg,
            debug=args.debug,
        )

    if not results:
        print("[WARN] No results to log.")
        wandb.finish()
        return

    # ── Compute summary metrics ───────────────────────────────────────────────
    avg_inference_ms = np.mean([r["inference_ms"] for r in results])
    avg_depth_mean   = np.mean([r["depth_mean"] for r in results])
    avg_n_points     = np.mean([r["n_points"] for r in results])

    summary = {
        "n_images": len(results),
        "avg_inference_ms": round(avg_inference_ms, 1),
        "avg_depth_mean": round(avg_depth_mean, 3),
        "avg_n_points": int(avg_n_points),
        "total_time_s": round(t.elapsed, 1),
    }

    wandb.log(summary)
    print(f"\n[Summary] {summary}")

    # ── Save CSV log ──────────────────────────────────────────────────────────
    os.makedirs(cfg["artifacts"]["logs"], exist_ok=True)
    csv_path = os.path.join(cfg["artifacts"]["logs"], f"inference_{args.split}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] CSV log saved: {csv_path}")

    # ── Log sample depth maps to W&B ─────────────────────────────────────────
    depth_dir = Path(output_dir) / "depth_maps"
    sample_imgs = list(depth_dir.glob("*_depth_color.png"))[:8]
    if sample_imgs:
        wandb.log({
            "depth_samples": [
                wandb.Image(str(p), caption=p.stem) for p in sample_imgs
            ]
        })

    # ── Log output folder as W&B artifact ────────────────────────────────────
    artifact = wandb.Artifact(
        name=f"depth-outputs-{args.split}",
        type="depth-maps",
        description=f"DaV2-{cfg['model']['encoder']} inference on {args.split} split",
        metadata=summary,
    )
    artifact.add_dir(output_dir)
    run.log_artifact(artifact)

    wandb.finish()
    print(f"\n[DONE] Inference complete in {t.elapsed:.1f}s")


if __name__ == "__main__":
    main()
