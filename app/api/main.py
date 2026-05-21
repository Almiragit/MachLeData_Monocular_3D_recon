"""
app/api/main.py
---------------
Stage 4 & 5 – FastAPI inference server + Drift Detection.

Endpoints:
  GET  /health          — liveness / readiness check
  GET  /drift           — current drift report (Stage 5)
  POST /predict         — accepts image → returns depth map PNG
  POST /predict/json    — accepts image → returns JSON with base64 depth + point cloud
  GET  /metrics         — Prometheus scrape endpoint (drift + API metrics)

Run locally:
    uvicorn app.api.main:app --reload --port 8000
"""

from app.monitoring.drift_detector import (
    ImageStats,
    drift_detector,
    extract_image_stats,
)
from src.utils import get_device, load_configs
from src.models.model import build_model
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Monocular 3D Reconstruction API",
    description=(
        "Depth estimation and 3D point cloud generation using Depth-Anything-V2 (ViT-B). "
        "Upload a single RGB image and receive depth map + 3D point cloud data."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

# ─── Globals ──────────────────────────────────────────────────────────────────
cfg = load_configs("configs/paths.yaml", "configs/train.yaml")
DEVICE = get_device()
MODEL = None
INPUT_SIZE = int(os.getenv("DAV2_INPUT_SIZE", cfg["model"]["input_size"]))


def _load_model():
    global MODEL
    MODEL = build_model(
        architecture=cfg["model"]["architecture"],
        checkpoint_dir=cfg["model"]["checkpoint_dir"],
        device=DEVICE,
    )
    print("[API] ✓ DaV2 model loaded and ready")


def _load_baseline():
    """Load drift baseline from artifacts/logs/baseline.json (created by compute_baseline.py)."""
    baseline_path = os.path.join(
        os.path.dirname(cfg["artifacts"]["logs"]), "baseline.json"
    )
    if not os.path.exists(baseline_path):
        print(f"[API] WARNING: No baseline found at {baseline_path}")
        print("[API] Drift detection will be inactive until baseline is computed.")
        print("[API] Run: python src/monitoring/compute_baseline.py")
        return

    try:
        with open(baseline_path) as f:
            data = json.load(f)
        stats_list = [ImageStats(**s) for s in data["stats"]]
        drift_detector.set_reference(stats_list)
        print(f"[API] ✓ Drift baseline loaded: {len(stats_list)} samples")
    except Exception as e:
        print(f"[API] ERROR loading baseline: {e}")


@app.on_event("startup")
async def startup():
    _load_model()
    _load_baseline()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR numpy array."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _depth_to_colormap_png(depth_norm: np.ndarray) -> bytes:
    """Apply INFERNO colormap and encode as PNG bytes."""
    color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    _, buf = cv2.imencode(".png", color)
    return buf.tobytes()


def _build_point_cloud_json(depth_norm: np.ndarray, rgb: np.ndarray,
                            subsample: int = 4) -> dict:
    """
    Build a lightweight point cloud dict suitable for JSON / Plotly.
    Returns {"x": [...], "y": [...], "z": [...], "colors": [...]}
    """
    h, w = depth_norm.shape
    fx = fy = 500
    cx, cy = w / 2, h / 2

    ys, xs = np.mgrid[0:h:subsample, 0:w:subsample]
    zs = depth_norm[ys, xs].astype(np.float32) / 255.0
    mask = zs > 0.05

    X = ((xs - cx) * zs / fx)[mask]
    Y = (-(ys - cy) * zs / fy)[mask]
    Z = zs[mask]

    r = rgb[ys, xs, 0][mask]
    g = rgb[ys, xs, 1][mask]
    b = rgb[ys, xs, 2][mask]

    return {
        "x": X.tolist(),
        "y": Y.tolist(),
        "z": Z.tolist(),
        "colors": [f"rgb({int(ri)},{int(gi)},{int(bi)})"
                   for ri, gi, bi in zip(r, g, b)],
        "n_points": int(mask.sum()),
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Monocular 3D Reconstruction API",
        "model": f"Depth-Anything-V2 ({cfg['model']['encoder']})",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model": cfg["model"]["architecture"],
        "device": str(DEVICE),
    }


@app.get("/drift", tags=["Monitoring"],
         summary="Current drift detection report (Stage 5)")
def drift_report():
    """Returns the latest drift scores computed by the background monitor."""
    return drift_detector.latest_report.summary


@app.post("/predict", tags=["Inference"],
          summary="Returns depth map as colourised PNG image")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(415, f"Unsupported type: {file.content_type}")

    raw = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Cannot decode image: {e}")

    bgr = _pil_to_bgr(pil_img)

    t0 = time.perf_counter()
    depth = MODEL.infer_image(bgr, input_size=INPUT_SIZE)
    ms = (time.perf_counter() - t0) * 1000

    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max > d_min:
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)

    png_bytes = _depth_to_colormap_png(depth_norm)

    # ── Stage 5: update drift detector ──────────────────────────────────────
    stats = extract_image_stats(bgr, depth_norm)
    drift_detector.update(stats)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Depth-Min": f"{d_min:.4f}",
            "X-Depth-Max": f"{d_max:.4f}",
            "X-Inference-Ms": f"{ms:.1f}",
        },
    )


@app.post("/predict/json", tags=["Inference"],
          summary="Returns depth map + point cloud data as JSON")
async def predict_json(file: UploadFile = File(...), subsample: int = 4):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(415, "Use JPEG or PNG")

    raw = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, str(e))

    bgr = _pil_to_bgr(pil_img)
    rgb = np.array(pil_img)

    t0 = time.perf_counter()
    depth = MODEL.infer_image(bgr, input_size=INPUT_SIZE)
    ms = (time.perf_counter() - t0) * 1000

    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max > d_min:
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)

    # Colourised depth → base64 PNG
    color_png = _depth_to_colormap_png(depth_norm)
    depth_b64 = base64.b64encode(color_png).decode()

    # Grayscale depth for Streamlit display
    _, gray_buf = cv2.imencode(".png", depth_norm)
    gray_b64 = base64.b64encode(gray_buf.tobytes()).decode()

    # ── Stage 5: update drift detector ──────────────────────────────────────
    stats = extract_image_stats(bgr, depth_norm)
    drift_detector.update(stats)

    # Point cloud data for Plotly
    pc = _build_point_cloud_json(depth_norm, rgb, subsample=subsample)

    return JSONResponse({
        "depth_colormap_b64": depth_b64,
        "depth_gray_b64": gray_b64,
        "point_cloud": pc,
        "depth_min": d_min,
        "depth_max": d_max,
        "inference_ms": round(ms, 1),
        "image_size": {"width": pil_img.width, "height": pil_img.height},
        "model": cfg["model"]["architecture"],
    })
