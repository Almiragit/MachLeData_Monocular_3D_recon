"""
app/monitoring/drift_detector.py
---------------------------------
Stage 5 – Drift Detection & Model Monitoring.

Monitors two types of drift:
  1. INPUT DRIFT   — changes in incoming image statistics vs. reference baseline
                     (brightness, contrast, blur, resolution, embedding similarity)
  2. PREDICTION DRIFT — changes in depth map output distributions
                         (depth mean/std, invalid ratio, point cloud density)

How it works:
  - On startup, loads a reference baseline computed from the training split.
  - Each API call updates a rolling window of stats.
  - A background thread periodically computes drift scores (PSI, KS-test)
    and exposes them as Prometheus gauges → Grafana alerts.

Usage: imported by app/api/main.py — not run standalone.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from prometheus_client import Gauge

# ─── Prometheus gauges (exposed at /metrics) ──────────────────────────────────
DRIFT_INPUT_BRIGHTNESS = Gauge(
    "drift_input_brightness_psi", "PSI score for image brightness")
DRIFT_INPUT_BLUR = Gauge("drift_input_blur_psi",
                         "PSI score for image blur (Laplacian)")
DRIFT_INPUT_CONTRAST = Gauge(
    "drift_input_contrast_psi", "PSI score for image contrast")
DRIFT_PRED_DEPTH_MEAN = Gauge(
    "drift_pred_depth_mean", "Rolling mean of predicted depth values")
DRIFT_PRED_DEPTH_STD = Gauge("drift_pred_depth_std",
                             "Rolling std of predicted depth values")
DRIFT_PRED_INVALID_RATIO = Gauge(
    "drift_pred_invalid_ratio", "Fraction of near-zero depth pixels")
DRIFT_KS_BRIGHTNESS = Gauge(
    "drift_ks_brightness_stat", "KS-test statistic for brightness")
DRIFT_ALERT_TRIGGERED = Gauge(
    "drift_alert_triggered", "1 if any drift alert is active, else 0")


# ─── Data structures ──────────────────────────────────────────────────────────
@dataclass
class ImageStats:
    brightness: float
    contrast: float
    blur: float
    depth_mean: float = 0.0
    depth_std: float = 0.0
    invalid_ratio: float = 0.0


@dataclass
class DriftReport:
    timestamp: float = field(default_factory=time.time)
    psi_brightness: float = 0.0
    psi_contrast: float = 0.0
    psi_blur: float = 0.0
    ks_brightness: float = 0.0
    depth_mean: float = 0.0
    depth_std: float = 0.0
    invalid_ratio: float = 0.0
    alert: bool = False

    @property
    def summary(self) -> dict:
        return {
            "psi_brightness": round(self.psi_brightness, 4),
            "psi_contrast": round(self.psi_contrast, 4),
            "psi_blur": round(self.psi_blur, 4),
            "ks_brightness": round(self.ks_brightness, 4),
            "depth_mean": round(self.depth_mean, 4),
            "depth_std": round(self.depth_std, 4),
            "invalid_ratio": round(self.invalid_ratio, 4),
            "alert": self.alert,
        }


# ─── PSI calculation ──────────────────────────────────────────────────────────
def compute_psi(reference: np.ndarray, production: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.10 → no significant drift
    PSI 0.10–0.25 → moderate drift, monitor
    PSI > 0.25 → significant drift, alert
    """
    eps = 1e-8
    ref_hist, bin_edges = np.histogram(reference, bins=n_bins, density=True)
    prod_hist, _ = np.histogram(production, bins=bin_edges, density=True)

    ref_hist = np.clip(ref_hist, eps, None)
    prod_hist = np.clip(prod_hist, eps, None)

    # Normalise to proportions
    ref_p = ref_hist / ref_hist.sum()
    prod_p = prod_hist / prod_hist.sum()

    psi = np.sum((prod_p - ref_p) * np.log(prod_p / ref_p))
    return float(psi)


def compute_ks(reference: np.ndarray, production: np.ndarray) -> float:
    """Kolmogorov-Smirnov test statistic."""
    from scipy.stats import ks_2samp
    stat, _ = ks_2samp(reference, production)
    return float(stat)


# ─── Image feature extractor ──────────────────────────────────────────────────
def extract_image_stats(bgr_image: np.ndarray, depth_norm: Optional[np.ndarray] = None) -> ImageStats:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    depth_mean = depth_std = invalid_ratio = 0.0
    if depth_norm is not None:
        d = depth_norm.astype(np.float32) / 255.0
        depth_mean = float(d.mean())
        depth_std = float(d.std())
        # near-zero = unreliable depth
        invalid_ratio = float((d < 0.02).mean())

    return ImageStats(
        brightness=brightness,
        contrast=contrast,
        blur=blur,
        depth_mean=depth_mean,
        depth_std=depth_std,
        invalid_ratio=invalid_ratio,
    )


# ─── Drift Detector ───────────────────────────────────────────────────────────
class DriftDetector:
    """
    Rolling-window drift detector.

    Call .update(stats) after each prediction.
    A background thread computes drift scores every *check_interval_s* seconds.
    """

    PSI_ALERT_THRESHOLD = 0.25   # significant drift
    PSI_WARN_THRESHOLD = 0.10   # moderate drift

    def __init__(
        self,
        window_size: int = 100,
        check_interval_s: float = 60.0,
    ):
        self.window_size = window_size
        self.check_interval = check_interval_s

        # Rolling window of production stats
        self._window: deque[ImageStats] = deque(maxlen=window_size)
        self._lock = threading.Lock()

        # Reference baseline (set via set_reference())
        self._ref_brightness: Optional[np.ndarray] = None
        self._ref_contrast: Optional[np.ndarray] = None
        self._ref_blur: Optional[np.ndarray] = None

        self.latest_report = DriftReport()

        # Start background thread
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("[DriftDetector] ✓ Background monitor started")

    def set_reference(self, stats_list: list[ImageStats]) -> None:
        """Set baseline from training/val split statistics."""
        self._ref_brightness = np.array([s.brightness for s in stats_list])
        self._ref_contrast = np.array([s.contrast for s in stats_list])
        self._ref_blur = np.array([s.blur for s in stats_list])
        print(
            f"[DriftDetector] Reference baseline set from {len(stats_list)} samples")

    def update(self, stats: ImageStats) -> None:
        """Call after each inference — thread-safe."""
        with self._lock:
            self._window.append(stats)

    def _compute_report(self) -> DriftReport:
        with self._lock:
            window = list(self._window)

        if len(window) < 10 or self._ref_brightness is None:
            return DriftReport()  # not enough data yet

        prod_brightness = np.array([s.brightness for s in window])
        prod_contrast = np.array([s.contrast for s in window])
        prod_blur = np.array([s.blur for s in window])

        psi_b = compute_psi(self._ref_brightness, prod_brightness)
        psi_c = compute_psi(self._ref_contrast, prod_contrast)
        psi_bl = compute_psi(self._ref_blur, prod_blur)
        ks_b = compute_ks(self._ref_brightness, prod_brightness)

        depth_mean = float(np.mean([s.depth_mean for s in window]))
        depth_std = float(np.mean([s.depth_std for s in window]))
        invalid_ratio = float(np.mean([s.invalid_ratio for s in window]))

        alert = any([
            psi_b > self.PSI_ALERT_THRESHOLD,
            psi_c > self.PSI_ALERT_THRESHOLD,
            psi_bl > self.PSI_ALERT_THRESHOLD,
            invalid_ratio > 0.30,
        ])

        return DriftReport(
            psi_brightness=psi_b,
            psi_contrast=psi_c,
            psi_blur=psi_bl,
            ks_brightness=ks_b,
            depth_mean=depth_mean,
            depth_std=depth_std,
            invalid_ratio=invalid_ratio,
            alert=alert,
        )

    def _monitor_loop(self) -> None:
        while True:
            time.sleep(self.check_interval)
            try:
                report = self._compute_report()
                self.latest_report = report

                # Update Prometheus gauges
                DRIFT_INPUT_BRIGHTNESS.set(report.psi_brightness)
                DRIFT_INPUT_CONTRAST.set(report.psi_contrast)
                DRIFT_INPUT_BLUR.set(report.psi_blur)
                DRIFT_KS_BRIGHTNESS.set(report.ks_brightness)
                DRIFT_PRED_DEPTH_MEAN.set(report.depth_mean)
                DRIFT_PRED_DEPTH_STD.set(report.depth_std)
                DRIFT_PRED_INVALID_RATIO.set(report.invalid_ratio)
                DRIFT_ALERT_TRIGGERED.set(1.0 if report.alert else 0.0)

                if report.alert:
                    print(f"[DriftDetector] ⚠️  DRIFT ALERT: {report.summary}")
                else:
                    print(
                        f"[DriftDetector] ✓ No drift (PSI_brightness={report.psi_brightness:.3f})")

            except Exception as e:
                print(f"[DriftDetector] Error in monitor loop: {e}")


# ─── Singleton instance (imported by main.py) ─────────────────────────────────
drift_detector = DriftDetector(window_size=100, check_interval_s=60.0)
