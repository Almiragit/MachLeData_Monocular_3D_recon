"""
src/registry/push_model.py
--------------------------
Stage 3 – Model Registry: Push best checkpoint to W&B Model Registry.

This script is called automatically by GitHub Actions after a successful
training run, but can also be run manually.

Usage:
    python src/registry/push_model.py
    python src/registry/push_model.py --checkpoint artifacts/checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_configs


def push_to_registry(
    checkpoint_path: str,
    project: str,
    model_name: str,
    metadata: dict | None = None,
) -> str:
    """
    Upload *checkpoint_path* to W&B Model Registry as a versioned artifact.
    Returns the artifact version string (e.g. 'v3').
    """
    run = wandb.init(
        project=project,
        job_type="model-registry",
        name=f"push-{model_name}",
    )

    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description="Best depth estimation checkpoint",
        metadata=metadata or {},
    )
    artifact.add_file(checkpoint_path, name="best_model.pth")
    logged = run.log_artifact(artifact)
    logged.wait()   # ensure upload completes before finishing run

    version = logged.version
    print(f"[Registry] ✓ Artifact logged: {model_name}:{version}")
    print(f"[Registry] View at: https://wandb.ai/{run.entity}/{project}/artifacts/model/{model_name}/{version}")

    wandb.finish()
    return version


def main():
    parser = argparse.ArgumentParser(description="Push model to W&B Model Registry")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/checkpoints/best_model.pth",
    )
    parser.add_argument("--paths_config", type=str, default="configs/paths.yaml")
    parser.add_argument("--train_config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    cfg = load_configs(args.paths_config, args.train_config)

    push_to_registry(
        checkpoint_path=args.checkpoint,
        project=cfg["experiment"]["project"],
        model_name=cfg["experiment"]["name"],
        metadata={
            "architecture": cfg["model"]["architecture"],
            "img_height": cfg["model"]["img_height"],
            "img_width": cfg["model"]["img_width"],
        },
    )


if __name__ == "__main__":
    main()
