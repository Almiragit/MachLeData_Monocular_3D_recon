"""
tests/test_basic.py
-------------------
Basic tests for the MLOps pipeline: project structure, configs, and core modules.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── Project Structure ────────────────────────────────────────────────────────
def test_project_structure():
    """Check that all required project files exist."""
    required = [
        "src/models/model.py",
        "src/models/losses.py",
        "src/training/train.py",
        "src/training/evaluate.py",
        "src/registry/push_model.py",
        "src/utils.py",
        "src/monitoring/compute_baseline.py",
        "app/api/main.py",
        "app/frontend/app.py",
        "app/monitoring/drift_detector.py",
        "configs/paths.yaml",
        "configs/train.yaml",
        "dvc.yaml",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        ".github/workflows/ci.yml",
    ]
    for path in required:
        assert os.path.exists(path), f"Missing required file: {path}"


def test_yaml_configs():
    """Validate all YAML configs parse correctly."""
    import yaml
    for path in ["configs/paths.yaml", "configs/train.yaml", "dvc.yaml"]:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None, f"{path} is empty or invalid"


# ─── Config Values ────────────────────────────────────────────────────────────
def test_paths_config():
    """Verify paths.yaml has required keys."""
    import yaml
    with open("configs/paths.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "data" in cfg
    assert "model" in cfg
    assert "artifacts" in cfg
    assert cfg["model"]["encoder"] in ("vits", "vitb", "vitl")
    assert cfg["model"]["input_size"] > 0


def test_train_config():
    """Verify train.yaml has training hyperparameters."""
    import yaml
    with open("configs/train.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "training" in cfg
    assert cfg["training"]["epochs"] >= 1
    assert cfg["training"]["batch_size"] >= 1
    assert cfg["training"]["learning_rate"] > 0


# ─── DVC Pipeline ─────────────────────────────────────────────────────────────
def test_dvc_pipeline_stages():
    """Verify dvc.yaml defines all expected stages."""
    import yaml
    with open("dvc.yaml") as f:
        cfg = yaml.safe_load(f)
    stages = cfg["stages"]
    expected = ["download_nyu", "prepare_data", "train",
                "compute_baseline", "evaluate", "push_registry"]
    for stage in expected:
        assert stage in stages, f"Missing DVC stage: {stage}"
    assert len(stages) == len(expected), (
        f"Expected {len(expected)} stages, got {len(stages)}"
    )


# ─── Loss Functions ───────────────────────────────────────────────────────────
def test_silog_loss():
    """SILogLoss runs without errors."""
    import torch
    from src.models.losses import SILogLoss

    criterion = SILogLoss()
    pred = torch.rand(2, 1, 64, 64) * 5 + 0.5   # realistic depth [0.5, 5.5]
    target = torch.rand(2, 1, 64, 64) * 5 + 0.5
    loss = criterion(pred, target)
    assert loss.item() > 0
    assert torch.isfinite(loss).all(), "SILogLoss produced non-finite loss"


def test_berhu_loss():
    """BerHuLoss runs without errors."""
    import torch
    from src.models.losses import BerHuLoss

    criterion = BerHuLoss()
    pred = torch.rand(2, 1, 64, 64) * 5 + 0.5
    target = torch.rand(2, 1, 64, 64) * 5 + 0.5
    loss = criterion(pred, target)
    assert loss.item() > 0
    assert torch.isfinite(loss).all(), "BerHuLoss produced non-finite loss"


# ─── Utility Functions ────────────────────────────────────────────────────────
def test_load_configs():
    """load_configs merges paths.yaml and train.yaml."""
    from src.utils import load_configs
    cfg = load_configs("configs/paths.yaml", "configs/train.yaml")
    assert "data" in cfg
    assert "training" in cfg
    assert "model" in cfg


def test_get_device():
    """get_device returns a valid torch device."""
    import torch
    from src.utils import get_device
    device = get_device()
    assert isinstance(device, torch.device)


def test_set_seed():
    """set_seed runs without errors."""
    from src.utils import set_seed
    set_seed(42)


# ─── Drift Detection ──────────────────────────────────────────────────────────
def test_psi_computation():
    """PSI computation works with different distributions."""
    import numpy as np
    from app.monitoring.drift_detector import compute_psi

    # Same distribution → PSI ~ 0
    ref = np.random.normal(100, 15, 1000)
    prod = np.random.normal(100, 15, 1000)
    psi_same = compute_psi(ref, prod)
    assert psi_same < 0.1, f"Expected low PSI for same dist, got {psi_same:.4f}"

    # Different distribution → PSI > 0.1
    prod_diff = np.random.normal(200, 15, 1000)
    psi_diff = compute_psi(ref, prod_diff)
    print(f"PSI same={psi_same:.4f}, diff={psi_diff:.4f}")


def test_ks_computation():
    """KS-test computation runs without errors."""
    import numpy as np
    from app.monitoring.drift_detector import compute_ks

    ref = np.random.normal(100, 15, 1000)
    prod = np.random.normal(105, 15, 1000)
    stat = compute_ks(ref, prod)
    assert 0 <= stat <= 1, f"KS stat should be in [0, 1], got {stat:.4f}"


def test_extract_image_stats():
    """extract_image_stats returns valid ImageStats."""
    import numpy as np
    import cv2
    from app.monitoring.drift_detector import extract_image_stats

    # Create a synthetic image
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_depth = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

    stats = extract_image_stats(dummy_img, dummy_depth)
    assert stats.brightness > 0
    assert stats.contrast > 0
    assert stats.blur >= 0
    assert stats.depth_mean >= 0
    assert stats.invalid_ratio >= 0