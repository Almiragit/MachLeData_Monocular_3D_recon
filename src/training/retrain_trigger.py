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
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ─── Config (env vars override defaults) ──────────────────────────────────────
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
CHECK_INTERVAL_S = int(os.getenv("CHECK_INTERVAL_S", "300"))
DVC_REPRO_ENABLED = os.getenv("DVC_REPRO_ENABLED", "true").lower() in ("1", "true", "yes")
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
        # Metric not present yet in Prometheus -> treat as no active alert
        return 0.0
    except Exception as e:
        print(f"[RetainTrigger] Prometheus query failed: {e}")
    return None


# ─── Retraining trigger ───────────────────────────────────────────────────────
def trigger_retraining() -> bool:
    """
    Trigger the current DVC training pipeline re-run.
    Uses the stage names defined in dvc.yaml:
      prepare_data -> train -> compute_baseline -> evaluate -> push_registry
    """
    print("[RetainTrigger] Triggering DVC pipeline re-run...")
    try:
        stages = ["prepare_data", "train",
                  "compute_baseline", "evaluate", "push_registry"]
        print(f"[RetainTrigger] Running: dvc repro --force {' '.join(stages)}")
        result = subprocess.run(
            ["dvc", "repro", "--force", *stages],
            capture_output=False,
            timeout=3600,   # 1 hour max
        )
        if result.returncode == 0:
            print("[RetainTrigger] OK: DVC pipeline completed successfully")
            return True
        else:
            print(
                f"[RetainTrigger] ERROR: DVC pipeline failed (exit {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print("[RetainTrigger] ERROR: Pipeline timed out after 1 hour")
        return False
    except FileNotFoundError:
        print("[RetainTrigger] ERROR: DVC not found - is it installed?")
        return False


def log_retrain_event(reason: str, success: bool) -> None:
    """Append retrain event to log file."""
    import csv
    import os
    from datetime import datetime, timezone

    log_path = "artifacts/logs/retrain_log.csv"
    os.makedirs("artifacts/logs", exist_ok=True)
    write_header = not Path(log_path).exists()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "reason", "success"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), reason, success])


# ─── Main loop ────────────────────────────────────────────────────────────────
def run(once: bool = False, dry_run: bool = False, force_alert: bool = False) -> None:
    consecutive_alerts = 0
    print(
        f"[RetainTrigger] Started. Checking Prometheus every {CHECK_INTERVAL_S}s")
    print(
        f"[RetainTrigger] Will trigger after {CONSECUTIVE_ALERTS_NEEDED} consecutive alerts")

    while True:
        if force_alert:
            alert_value = 1.0
        else:
            alert_value = query_prometheus(DRIFT_ALERT_METRIC)

        if alert_value is None:
            print("[RetainTrigger] WARN: Could not reach Prometheus - skipping check")
            consecutive_alerts = 0

        elif alert_value >= 1.0:
            consecutive_alerts += 1
            print(f"[RetainTrigger] ALERT: Drift alert active "
                  f"({consecutive_alerts}/{CONSECUTIVE_ALERTS_NEEDED})")

            if consecutive_alerts >= CONSECUTIVE_ALERTS_NEEDED:
                print(
                    "[RetainTrigger] Sustained drift confirmed -> triggering pipeline")
                if dry_run or not DVC_REPRO_ENABLED:
                    if not DVC_REPRO_ENABLED:
                        print("[RetainTrigger] [DVC_REPRO_DISABLED] Would trigger retraining "
                              "(set DVC_REPRO_ENABLED=true to activate)")
                    else:
                        print("[RetainTrigger] [DRY-RUN] Retraining would be triggered now")
                    success = True
                else:
                    success = trigger_retraining()
                log_retrain_event(
                    reason=f"sustained_drift_{CONSECUTIVE_ALERTS_NEEDED}_checks",
                    success=success,
                )
                consecutive_alerts = 0   # reset after trigger

        else:
            if consecutive_alerts > 0:
                print(
                    f"[RetainTrigger] OK: Alert cleared (was {consecutive_alerts} consecutive)")
            consecutive_alerts = 0
            print("[RetainTrigger] OK: No drift - system healthy")

        if once:
            break

        time.sleep(CHECK_INTERVAL_S)


def main():
    global PROMETHEUS_URL, CHECK_INTERVAL_S, CONSECUTIVE_ALERTS_NEEDED
    parser = argparse.ArgumentParser(
        description="Automated retraining trigger")
    parser.add_argument("--once", action="store_true",
                        help="Check once and exit (for CI/cron)")
    parser.add_argument("--prometheus", type=str, default=PROMETHEUS_URL,
                        help="Prometheus base URL")
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL_S,
                        help="Check interval in seconds")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate retrain trigger without running dvc repro")
    parser.add_argument("--force-alert", action="store_true",
                        help="Force alert_value=1.0 for trigger-path testing")
    parser.add_argument("--alerts-needed", type=int, default=CONSECUTIVE_ALERTS_NEEDED,
                        help="Override consecutive alerts needed before trigger")
    args = parser.parse_args()

    PROMETHEUS_URL = args.prometheus
    CHECK_INTERVAL_S = args.interval
    CONSECUTIVE_ALERTS_NEEDED = args.alerts_needed

    run(once=args.once, dry_run=args.dry_run, force_alert=args.force_alert)


if __name__ == "__main__":
    main()
