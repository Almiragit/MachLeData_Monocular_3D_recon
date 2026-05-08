"""
src/data/download.py
--------------------
Stage 1 – Data Preparation: sync raw dataset from AWS S3.

Usage:
    python src/data/download.py
    python src/data/download.py --bucket s3://my-bucket/dataset/ --output data/raw
    python src/data/download.py --dry-run    # preview without downloading
"""

import argparse
import os
import subprocess
import sys


def sync_from_s3(s3_uri: str, output_dir: str, profile: str | None = None,
                 dry_run: bool = False) -> None:
    """Sync *s3_uri* → *output_dir* using the AWS CLI."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = ["aws", "s3", "sync", s3_uri, output_dir, "--no-progress"]
    if profile:
        cmd += ["--profile", profile]
    if dry_run:
        cmd += ["--dryrun"]

    print(f"[INFO] {'DRY RUN: ' if dry_run else ''}Syncing {s3_uri} → {output_dir}")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(
            "[ERROR] AWS CLI not found. Install it:\n"
            "  pip install awscli\n"
            "  or: brew install awscli\n"
            "Then configure: aws configure"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] aws s3 sync failed with exit code {e.returncode}")
        sys.exit(e.returncode)

    if not dry_run:
        files = sum(len(fs) for _, _, fs in os.walk(output_dir))
        print(f"[INFO] ✓ Sync complete. {files} files in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset from AWS S3")
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="S3 URI (e.g. s3://my-bucket/dataset/). "
             "Falls back to configs/paths.yaml → aws.s3_bucket",
    )
    parser.add_argument(
        "--output", type=str, default="data/raw",
        help="Local destination directory",
    )
    parser.add_argument(
        "--profile", type=str, default=None,
        help="AWS CLI profile name (optional)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview sync without downloading",
    )
    args = parser.parse_args()

    # Fall back to config if no CLI bucket given
    s3_uri = args.bucket
    if not s3_uri:
        try:
            import yaml
            with open("configs/paths.yaml") as f:
                cfg = yaml.safe_load(f)
            s3_uri = cfg["aws"]["s3_bucket"]
        except Exception:
            print("[ERROR] No --bucket given and configs/paths.yaml not readable.")
            sys.exit(1)

    sync_from_s3(s3_uri, args.output, args.profile, args.dry_run)


if __name__ == "__main__":
    main()
