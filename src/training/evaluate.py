import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb

# ==========================================================
# 1. RESOLVE ABSOLUTE SYSTEM PATHS
# ==========================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

REPO_PATH = os.path.join(BASE_DIR, 'src', 'models', 'Depth-Anything-V2')
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)

from ipynb.fs.full.DAV2_Hybrid import load_hybrid_model
from train import NYUDataset, SILogLoss 

# ==========================================================
# 2. W&B CONFIGURATION (CRITICAL FOR MERGING FILES)
# ==========================================================
# ⚠️ REPLACE THIS STRING WITH YOUR ACTUAL 8-CHARACTER RUN ID FROM YOUR W&B OVERVIEW PAGE!
YOUR_RUN_ID = "YOUR_ACTUAL_RUN_ID_HERE" 

VAL_DATA_PATH = os.path.join(BASE_DIR, "src", "data", "data", "processed", "val")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running evaluation on device target: {device}")

# RESUME THE PREVIOUS RUN SAFELY
print(f"Connecting and merging into W&B run folder: hybrid-decoder-v1-run (ID: {YOUR_RUN_ID})")
run = wandb.init(
    project="Monocular-3D-Reconstruction",
    id=YOUR_RUN_ID,
    resume="allow"
)

# ==========================================================
# 3. LOAD LOCAL WEIGHTS DIRECTLY (SMART SEARCH)
# ==========================================================
FILENAME = "latest_hybrid_model.pth"

# List of potential places the training loop might have dropped the file
possible_paths = [
    os.path.join(BASE_DIR, FILENAME),                          # Root folder
    os.path.join(BASE_DIR, "src", "training", FILENAME),       # src/training/
    os.path.join(os.path.dirname(__file__), FILENAME),         # Current directory
    os.path.join(BASE_DIR, "checkpoints", FILENAME)            # checkpoints folder
]

weights_path = None
for path in possible_paths:
    if os.path.exists(path):
        weights_path = path
        break

if weights_path is None:
    raise FileNotFoundError(
        f"Could not locate '{FILENAME}' automatically.\n"
        f"Please manually check your folders and move it to your root project folder:\n"
        f"-> C:\\Users\\kanha\\Documents\\MachLeData\\"
    )

print(f"🎯 Found weights file successfully at: {weights_path}")

# Load model structure and local weights
model = load_hybrid_model(encoder='vitb', device=device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval() 

# ==========================================================
# 4. PREPARE VALIDATION LOADER
# ==========================================================
val_dataset = NYUDataset(VAL_DATA_PATH)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

criterion = SILogLoss()

# Metrics storage arrays
total_val_loss = 0
all_mae, all_rmse, all_abs_rel = [], [], []
all_delta1, all_delta2, all_delta3 = [], [], []

print(f"\n--- Starting evaluation over {len(val_dataset)} validation samples ---")

# ==========================================================
# 5. EVALUATION LOOP (With Live Line Trajectory Tracking)
# ==========================================================
with torch.no_grad():
    for batch_idx, (images, depths) in enumerate(tqdm(val_loader, desc="Evaluating")):
        images, depths = images.to(device), depths.to(device)
        
        # Predict pass
        outputs = model(images)
        outputs = torch.clamp(outputs, min=0.1, max=10.0)
        
        # 1. Compute validation loss
        loss = criterion(outputs, depths)
        total_val_loss += loss.item()
        
        # 2. Compute quantitative metrics pixel-by-pixel
        valid_mask = (depths > 0) & (depths <= 10.0)
        if not valid_mask.any():
            continue
            
        pred_valid = outputs[valid_mask]
        gt_valid = depths[valid_mask]
        
        # Metric formulas implementation
        all_mae.append(torch.mean(torch.abs(pred_valid - gt_valid)).item())
        all_rmse.append(torch.sqrt(torch.mean((pred_valid - gt_valid) ** 2)).item())
        all_abs_rel.append(torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid).item())
        
        # Threshold Accuracies (delta brackets)
        ratios = torch.max(pred_valid / gt_valid, gt_valid / pred_valid)
        all_delta1.append((ratios < 1.25).float().mean().item())
        all_delta2.append((ratios < 1.25 ** 2).float().mean().item())
        all_delta3.append((ratios < 1.25 ** 3).float().mean().item())

        # 🚀 LIVE LINE LOGGING: Every 5 batches, calculate the running average 
        # and push it to W&B to build a moving trajectory line on your charts.
        if batch_idx % 5 == 0:
            wandb.log({
                "val_loss_trajectory": total_val_loss / (batch_idx + 1),
                "val_RMSE_trajectory": np.mean(all_rmse),
                "val_MAE_trajectory": np.mean(all_mae),
                "val_delta2_trajectory": np.mean(all_delta2),
                "val_delta3_trajectory": np.mean(all_delta3),
            }, step=batch_idx)

# ==========================================================
# 6. FINAL SUMMARY REPORT
# ==========================================================
metrics = {
    "final_val_loss_SILog": total_val_loss / len(val_loader),
    "final_val_MAE": np.mean(all_mae),
    "final_val_RMSE": np.mean(all_rmse),
    "final_val_abs_rel": np.mean(all_abs_rel),
    "final_val_delta1": np.mean(all_delta1),
    "final_val_delta2": np.mean(all_delta2),
    "final_val_delta3": np.mean(all_delta3),
}

print("\n================ FINAL EVALUATION REPORT ================")
for name, value in metrics.items():
    print(f"📈 {name:<20}: {value:.4f}")
print("===========================================================")

# Log final overall summary markers
wandb.log(metrics)
run.finish()
print("\n🎉 Success! Evaluation complete. Check W&B for your new trajectory charts.")