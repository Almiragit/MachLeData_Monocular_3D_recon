"""
src/models/model.py
-------------------
Wrapper around Depth-Anything-V2 (ViT-B) for the MLOps pipeline.

The actual DaV2 source lives in:
    src/models/Depth-Anything-V2/depth_anything_v2/dpt.py

The pretrained checkpoint lives (or should be downloaded to):
    src/models/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth

This wrapper keeps the rest of the pipeline (FastAPI, evaluate.py, etc.)
model-agnostic — they just call build_model() and model.infer_image().
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
    sys.path.append(str(_DAV2_ROOT))  # append, not insert(0) — avoids shadowing our app/ package

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


# ── Wrapper ───────────────────────────────────────────────────────────────────
class DepthAnythingWrapper(nn.Module):
    """
    Thin wrapper around DepthAnythingV2 that:
      - loads the pretrained checkpoint automatically
      - exposes infer_image() matching the notebook API
      - exposes forward() for pipeline compatibility
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
        """
        Run depth inference on a single BGR image (OpenCV format).
        Returns depth map as float32 numpy array, same H×W as input.
        Matches the API used in DaV2_PtCloud.ipynb.
        """
        with torch.no_grad():
            return self._model.infer_image(raw_bgr_image, input_size=input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for batch tensors (used by FastAPI / DataLoader pipeline).
        x: (B, 3, H, W) float32 normalized tensor
        Returns: (B, 1, H, W) depth map
        """
        # DaV2 internal forward works on preprocessed tensors directly
        with torch.no_grad():
            depth = self._model(x)          # (B, H, W)
        return depth.unsqueeze(1)           # (B, 1, H, W)


# ── Factory ───────────────────────────────────────────────────────────────────
def build_model(
    architecture: str = "dav2_vitb",
    checkpoint_dir: str | None = None,
    device: torch.device | None = None,
) -> DepthAnythingWrapper:
    """
    Build and return the depth model.

    architecture options:
      dav2_vits  — ViT-Small  (fastest, lighter)
      dav2_vitb  — ViT-Base   (default, used in notebook)
      dav2_vitl  — ViT-Large  (best quality, needs more VRAM)
    """
    encoder = architecture.replace("dav2_", "")
    return DepthAnythingWrapper(
        encoder=encoder,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )