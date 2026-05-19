"""
src/models/model.py
-------------------
Wrapper around Depth-Anything-V2 (ViT) for the MLOps pipeline.
Supports both inference and fine-tuning.

Two factory functions:
  - build_model()         → inference-only DaV2 (evaluate, FastAPI)
  - build_hybrid_model()  → trainable DaV2 (fine-tune decoder only)

The actual DaV2 source lives in:
    src/models/Depth-Anything-V2/depth_anything_v2/dpt.py

The pretrained checkpoint:
    src/models/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# ── Locate the DaV2 package ───────────────────────────────────────────────────
_DAV2_ROOT = Path(__file__).resolve().parent / "Depth-Anything-V2"
if str(_DAV2_ROOT) not in sys.path:
    sys.path.append(str(_DAV2_ROOT))

try:
    from depth_anything_v2.dpt import DepthAnythingV2 as _DaV2
    _DAV2_AVAILABLE = True
except ImportError:
    _DAV2_AVAILABLE = False
    print("[model] WARNING: depth_anything_v2 package not found. "
          f"Expected at: {_DAV2_ROOT}")


# ── Model configs per encoder size ───────────────────────────────────────────
_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

_CHECKPOINT_NAMES = {
    "vits": "depth_anything_v2_vits.pth",
    "vitb": "depth_anything_v2_vitb.pth",
    "vitl": "depth_anything_v2_vitl.pth",
}


# ── Inference Wrapper (unchanged from dev) ────────────────────────────────────
class DepthAnythingWrapper(nn.Module):
    """
    Thin wrapper around DepthAnythingV2 for INFERENCE ONLY.
    Used by FastAPI, evaluate.py, and the original DVC pipeline.
    """

    def __init__(
        self,
        encoder: str = "vitb",
        checkpoint_dir: str | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()

        if not _DAV2_AVAILABLE:
            raise ImportError(
                "depth_anything_v2 is not importable. "
                f"Make sure {_DAV2_ROOT} exists and contains the DaV2 source."
            )

        if encoder not in _MODEL_CONFIGS:
            raise ValueError(f"encoder must be one of {list(_MODEL_CONFIGS.keys())}")

        self.encoder = encoder
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Build model
        cfg = _MODEL_CONFIGS[encoder]
        self._model = _DaV2(**cfg)

        # Load checkpoint
        ckpt_dir = checkpoint_dir or str(_DAV2_ROOT / "checkpoints")
        ckpt_path = os.path.join(ckpt_dir, _CHECKPOINT_NAMES[encoder])

        if os.path.exists(ckpt_path):
            self._model.load_state_dict(
                torch.load(ckpt_path, map_location="cpu")
            )
            print(f"[Model] ✓ Loaded DaV2-{encoder} checkpoint: {ckpt_path}")
        else:
            print(
                f"[Model] WARNING: checkpoint not found at {ckpt_path}\n"
                f"  Download from: https://huggingface.co/depth-anything/Depth-Anything-V2-{encoder.upper()}\n"
                f"  Place at: {ckpt_path}"
            )

        self._model = self._model.to(self.device).eval()

        n_params = sum(p.numel() for p in self._model.parameters())
        print(f"[Model] DaV2-{encoder} — {n_params:,} parameters (inference only)")

    def infer_image(self, raw_bgr_image: np.ndarray, input_size: int = 518) -> np.ndarray:
        with torch.no_grad():
            return self._model.infer_image(raw_bgr_image, input_size=input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            depth = self._model(x)
        return depth.unsqueeze(1)


# ── Hybrid Model for Fine-Tuning ──────────────────────────────────────────────
class HybridDepthModel(nn.Module):
    """
    DaV2 with a TRAINABLE decoder and FROZEN encoder.

    Architecture:
        DINOv2 (frozen) → DPTHead (trainable) → depth map

    The 'custom_decoder' attribute references the trainable DPTHead,
    so optimizers can target only these parameters:
        optimizer = Adam(model.custom_decoder.parameters(), lr=1e-4)
    """

    def __init__(
        self,
        encoder: str = "vitb",
        checkpoint_dir: str | None = None,
        device: torch.device | None = None,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        if not _DAV2_AVAILABLE:
            raise ImportError("depth_anything_v2 not importable. Check _DAV2_ROOT.")

        if encoder not in _MODEL_CONFIGS:
            raise ValueError(f"encoder must be one of {list(_MODEL_CONFIGS.keys())}")

        self.encoder = encoder
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.freeze_encoder = freeze_encoder

        # Build base DaV2 model
        cfg = _MODEL_CONFIGS[encoder]
        self._dav2 = _DaV2(**cfg)

        # Load pretrained checkpoint
        ckpt_dir = checkpoint_dir or str(_DAV2_ROOT / "checkpoints")
        ckpt_path = os.path.join(ckpt_dir, _CHECKPOINT_NAMES[encoder])

        if os.path.exists(ckpt_path):
            self._dav2.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            print(f"[HybridModel] ✓ Loaded DaV2-{encoder} pretrained weights")
        else:
            print(f"[HybridModel] WARNING: checkpoint not found at {ckpt_path}")

        # Reference the DPTHead as 'custom_decoder' (used by train.py optimizer)
        self.custom_decoder = self._dav2.depth_head

        # Freeze DINOv2 encoder
        if freeze_encoder:
            for param in self._dav2.pretrained.parameters():
                param.requires_grad = False
            print(f"[HybridModel] DINOv2 encoder FROZEN ({freeze_encoder})")

        # Count trainable vs total params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[HybridModel] {total:,} total params | {trainable:,} trainable")

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: DINOv2 (maybe frozen) → DPTHead (trainable).
        Returns depth map as (B, 1, H, W).
        """
        return self._dav2(x)


# ── Factory Functions ─────────────────────────────────────────────────────────
def build_model(
    architecture: str = "dav2_vitb",
    checkpoint_dir: str | None = None,
    device: torch.device | None = None,
) -> DepthAnythingWrapper:
    """
    Build inference-only DaV2 model (no gradients).
    Used by: evaluate.py, FastAPI, original DVC pipeline.
    """
    encoder = architecture.replace("dav2_", "")
    return DepthAnythingWrapper(
        encoder=encoder,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )


def build_hybrid_model(
    encoder: str = "vitb",
    checkpoint_dir: str | None = None,
    device: torch.device | None = None,
    freeze_encoder: bool = True,
) -> HybridDepthModel:
    """
    Build trainable DaV2 model with frozen DINOv2 encoder.
    Used by: train.py (fine-tuning).

    Only custom_decoder (DPTHead) parameters are trainable.
    """
    return HybridDepthModel(
        encoder=encoder,
        checkpoint_dir=checkpoint_dir,
        device=device,
        freeze_encoder=freeze_encoder,
    )