"""
src/registry/push_model.py
--------------------------
Push best checkpoint to W&B Model Registry.

Usage:
    python src/registry/push_model.py
    python src/registry/push_model.py --checkpoint artifacts/checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_configs


def push_to_registry(
    checkpoint_path: str,
    entity: str | None,
    project: str,
    model_name: str,
    metadata: dict | None = None,
) -> str:
    """Upload checkpoint to W&B Model Registry. Returns version string."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"registry-{model_name}-{ts}"
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="model-registry",
        group=model_name,
        name=run_name,
    )

    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description="Fine-tuned DaV2 Hybrid Decoder checkpoint",
        metadata=metadata or {},
    )
    artifact.add_file(checkpoint_path, name="best_model.pth")
    logged = run.log_artifact(artifact)
    logged.wait()

    version = logged.version
    print(f"[Registry] Artifact logged: {model_name}:{version}")
    wandb.finish()
    return version


def main():
    parser = argparse.ArgumentParser(description="Push model to W&B Model Registry")
    parser.add_argument("--checkpoint", type=str,
                        default="artifacts/checkpoints/best_model.pth")
    parser.add_argument("--paths_config", default="configs/paths.yaml")
    parser.add_argument("--train_config", default="configs/train.yaml")
    args = parser.parse_args()

    cfg = load_configs(args.paths_config, args.train_config)

    push_to_registry(
        checkpoint_path=args.checkpoint,
        entity=cfg["experiment"].get("entity"),
        project=cfg["experiment"]["project"],
        model_name=cfg["experiment"]["name"] + "-finetuned",
        metadata={
            "encoder": cfg["model"]["encoder"],
            "architecture": cfg["model"]["architecture"],
            "epochs": cfg["training"]["epochs"],
            "learning_rate": cfg["training"]["learning_rate"],
            "freeze_encoder": cfg["training"].get("freeze_encoder", True),
        },
    )


if __name__ == "__main__":
    main()