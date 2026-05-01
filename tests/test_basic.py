# tests/test_basic.py
import os

def test_project_structure():
    """Checks if the basic project files are in place."""
    assert os.path.exists("src/models/model.py"), "model.py is missing!"
    assert os.path.exists("dvc.yaml"), "dvc.yaml is missing!"
    assert os.path.exists("requirements.txt"), "requirements.txt is missing!"
    assert os.path.exists("configs/train.yaml"), "train.yaml is missing!"