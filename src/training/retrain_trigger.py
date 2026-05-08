"""
src/training/retrain_trigger.py
--------------------------------
Stage 5 – Automated Retraining Loop.

Checks drift metrics exposed by the monitoring stack (via Prometheus API)
and triggers a DVC-based retraining run when sustained drift is detected.

Trigger conditions (any one is sufficient):
  - PSI score for brightness/contrast/blur > 0.25 for N consecutive checks
  - Invalid depth ratio > 0.30
  - New validated data exceeds threshold in S3 (future: active learning)

Usage:
    # Run as a scheduled job (e.g. cron or GitHub Actions schedule)
    python src/training/retrain_trigger.py

    # Or run once and exit:
    python src/training/retrain_trigger.py --once
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ─── Config ───────────────────────────────────────────────────────────────────
PROMETHEUS_URL = "http://localhost:9090"
CHECK_INTERVAL_S = 300           # check every 5 minutes
DRIFT_ALERT_METRIC = "drift_alert_triggered"
CONSECUTIVE_ALERTS_NEEDED = 3   # require 3 consecutive alerts before retraining


# ─── Prometheus query helper ──────────────────────────────────────────────────
def query_prometheus(metric: str) -> float | None:
    """Query current value of a Prometheus instant metric."""
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": metric},
            timeout=5,
        )
        data = resp.json()
        results = data.get("data", {}).get("result", [])
        if results:
            return float(results[0]["value"][1])
    except Exception as e:
        print(f"[RetainTrigger] Prometheus query failed: {e}")
    return None


# ─── Retraining trigger ───────────────────────────────────────────────────────
def trigger_retraining() -> bool:
    """
    Trigger the DVC inference pipeline re-run.
    In a real pipeline this would kick off full retraining if the model
    supports fine-tuning. For DaV2 (inference-only), we re-run the
    data pipeline to refresh outputs with the latest data.
    """
    print("[RetainTrigger] 🚀 Triggering DVC pipeline re-run...")
    try:
        result = subprocess.run(
            ["dvc", "repro", "--force", "preprocess", "inference_val", "inference_test"],
            capture_output=False,
            timeout=3600,   # 1 hour max
        )
        if result.returncode == 0:
            print("[RetainTrigger] ✓ DVC pipeline completed successfully")
            return True
        else:
            print(f"[RetainTrigger] ✗ DVC pipeline failed (exit {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print("[RetainTrigger] ✗ Pipeline timed out after 1 hour")
        return False
    except FileNotFoundError:
        print("[RetainTrigger] ✗ DVC not found — is it installed?")
        return False


def log_retrain_event(reason: str, success: bool) -> None:
    """Append retrain event to log file."""
    import csv, os
    from datetime import datetime

    log_path = "artifacts/logs/retrain_log.csv"
    os.makedirs("artifacts/logs", exist_ok=True)
    write_header = not Path(log_path).exists()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "reason", "success"])
        writer.writerow([datetime.utcnow().isoformat(), reason, success])


# ─── Main loop ────────────────────────────────────────────────────────────────
def run(once: bool = False) -> None:
    consecutive_alerts = 0
    print(f"[RetainTrigger] Started. Checking Prometheus every {CHECK_INTERVAL_S}s")
    print(f"[RetainTrigger] Will trigger after {CONSECUTIVE_ALERTS_NEEDED} consecutive alerts")

    while True:
        alert_value = query_prometheus(DRIFT_ALERT_METRIC)

        if alert_value is None:
            print("[RetainTrigger] ⚠️  Could not reach Prometheus — skipping check")
            consecutive_alerts = 0

        elif alert_value >= 1.0:
            consecutive_alerts += 1
            print(f"[RetainTrigger] 🔴 Drift alert active "
                  f"({consecutive_alerts}/{CONSECUTIVE_ALERTS_NEEDED})")

            if consecutive_alerts >= CONSECUTIVE_ALERTS_NEEDED:
                print("[RetainTrigger] Sustained drift confirmed → triggering pipeline")
                success = trigger_retraining()
                log_retrain_event(
                    reason=f"sustained_drift_{CONSECUTIVE_ALERTS_NEEDED}_checks",
                    success=success,
                )
                consecutive_alerts = 0   # reset after trigger

        else:
            if consecutive_alerts > 0:
                print(f"[RetainTrigger] ✅ Alert cleared (was {consecutive_alerts} consecutive)")
            consecutive_alerts = 0
            print("[RetainTrigger] ✅ No drift — system healthy")

        if once:
            break

        time.sleep(CHECK_INTERVAL_S)


def main():
    parser = argparse.ArgumentParser(description="Automated retraining trigger")
    parser.add_argument("--once", action="store_true",
                        help="Check once and exit (for CI/cron)")
    parser.add_argument("--prometheus", type=str, default=PROMETHEUS_URL,
                        help="Prometheus base URL")
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL_S,
                        help="Check interval in seconds")
    args = parser.parse_args()

    global PROMETHEUS_URL, CHECK_INTERVAL_S
    PROMETHEUS_URL = args.prometheus
    CHECK_INTERVAL_S = args.interval

    run(once=args.once)


if __name__ == "__main__":
    main()
