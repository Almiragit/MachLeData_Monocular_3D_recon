"""
src/monitoring/compute_baseline.py
-----------------------------------
Compute reference baseline statistics from the validation split for drift detection.

The baseline is used by the FastAPI DriftDetector to compare incoming images
against the training distribution. It captures:
  - Brightness, contrast, blur (image-level features)
  - Depth statistics (if ground truth is available)

Usage:
    python src/monitoring/compute_baseline.py
    python src/monitoring/compute_baseline.py --data_dir data/nyu/processed/val --output artifacts/logs/baseline.json

Output:
    JSON file containing arrays of ImageStats (brightness, contrast, blur, depth_mean, depth_std, invalid_ratio)
    and aggregate statistics (mean, std, percentiles).
"""

from monitoring.drift_detector import extract_image_stats
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import from drift_detector to ensure compatibility
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "app"))


def load_nyu_file(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load a single NYU .pt file and return (bgr_image, depth_norm).
    Returns (bgr_image, None) if no depth is available.
    """
    data = torch.load(path, map_location='cpu', weights_only=False)

    # Image: (H, W, C) numpy float [0,1] or uint8
    image = data['image']
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    # Convert RGB to BGR (OpenCV format for extract_image_stats)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    depth = None
    if 'depth' in data:
        d = data['depth']
        if d.max() > 0:
            # Normalize to [0, 255] uint8 for extract_image_stats
            d_norm = (d - d.min()) / (d.max() - d.min()) * 255
            depth = d_norm.astype(np.uint8)

    return bgr, depth


def compute_baseline(
    data_dir: str,
    output_path: str,
    max_samples: int = 500,
    debug: bool = False,
) -> list[dict]:
    """
    Iterate over .pt files in data_dir, extract image stats,
    save as JSON baseline, and print summary statistics.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pt_files = sorted(data_dir.glob("*.pt"))
    if not pt_files:
        # Try recursive: maybe files are in subdirectories
        pt_files = sorted(data_dir.rglob("*.pt"))

    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    if debug:
        pt_files = pt_files[:20]
        print(f"[DEBUG] Using {len(pt_files)} files (debug mode)")

    if max_samples and len(pt_files) > max_samples:
        import random
        random.seed(42)
        pt_files = random.sample(pt_files, max_samples)
        print(
            f"[Baseline] Sampled {len(pt_files)} files from {len(pt_files)} total")

    stats_list = []
    skipped = 0

    print(f"[Baseline] Computing baseline from {len(pt_files)} files...")
    for pt_path in tqdm(pt_files, desc="Extracting stats"):
        try:
            bgr, depth = load_nyu_file(str(pt_path))
            stats = extract_image_stats(bgr, depth_norm=depth)
            stats_list.append({
                "brightness": stats.brightness,
                "contrast": stats.contrast,
                "blur": stats.blur,
                "depth_mean": stats.depth_mean,
                "depth_std": stats.depth_std,
                "invalid_ratio": stats.invalid_ratio,
            })
        except Exception as e:
            skipped += 1
            if debug:
                print(f"  [SKIP] {pt_path.name}: {e}")

    if not stats_list:
        raise RuntimeError("No valid stats could be extracted from any file")

    # Compute aggregate statistics for the report
    brightness = np.array([s["brightness"] for s in stats_list])
    contrast = np.array([s["contrast"] for s in stats_list])
    blur = np.array([s["blur"] for s in stats_list])

    baseline_report = {
        "n_samples": len(stats_list),
        "n_skipped": skipped,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        "stats": stats_list,
        "aggregates": {
            "brightness": {
                "mean": float(brightness.mean()),
                "std": float(brightness.std()),
                "p5": float(np.percentile(brightness, 5)),
                "p95": float(np.percentile(brightness, 95)),
            },
            "contrast": {
                "mean": float(contrast.mean()),
                "std": float(contrast.std()),
                "p5": float(np.percentile(contrast, 5)),
                "p95": float(np.percentile(contrast, 95)),
            },
            "blur": {
                "mean": float(blur.mean()),
                "std": float(blur.std()),
                "p5": float(np.percentile(blur, 5)),
                "p95": float(np.percentile(blur, 95)),
            },
        },
    }

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(baseline_report, f, indent=2)
    print(f"[Baseline] ✓ Saved baseline to {output_path}")

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"  Baseline Summary ({len(stats_list)} samples)")
    print(f"{'=' * 50}")
    for metric in ["brightness", "contrast", "blur"]:
        agg = baseline_report["aggregates"][metric]
        print(f"  {metric:<12} mean={agg['mean']:.2f}  "
              f"std={agg['std']:.2f}  "
              f"p5={agg['p5']:.2f}  p95={agg['p95']:.2f}")
    print(f"{'=' * 50}\n")

    return stats_list


def main():
    parser = argparse.ArgumentParser(
        description="Compute drift baseline from validation data")
    parser.add_argument("--data_dir", type=str, default="data/nyu/processed/val",
                        help="Directory with .pt files (default: data/nyu/processed/val)")
    parser.add_argument("--output", type=str, default="artifacts/logs/baseline.json",
                        help="Output JSON path (default: artifacts/logs/baseline.json)")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Maximum number of files to process (default: 500)")
    parser.add_argument("--debug", action="store_true",
                        help="Use only 20 files for quick test")
    args = parser.parse_args()

    compute_baseline(
        data_dir=args.data_dir,
        output_path=args.output,
        max_samples=args.max_samples if not args.debug else None,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
