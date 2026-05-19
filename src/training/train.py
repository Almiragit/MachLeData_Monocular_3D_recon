import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import wandb
from tqdm import tqdm

# ==========================================================
# 1. RESOLVE ABSOLUTE PATHS & NOTEBOOK IMPORTER
# ==========================================================
# This script is located in 'src/training'. Go up 2 levels to reach 'MachLeData' root.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Inject Depth-Anything-V2 repository path directly into Python's search path
REPO_PATH = os.path.join(BASE_DIR, 'src', 'models', 'Depth-Anything-V2')
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

# Inject the notebooks directory so Python can locate your .ipynb file
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)

try:
    # Reads directly from your DAV2_Hybrid.ipynb notebook file
    from ipynb.fs.full.DAV2_Hybrid import load_hybrid_model
    print("--- Console: Successfully imported Hybrid Model directly from DAV2_Hybrid.ipynb! ---")
except ImportError as e:
    print(f"\n[CRITICAL ERROR]: Could not parse notebook features. Error detail: {e}")
    print("👉 Make sure you ran 'pip install ipynb' in your terminal environment.")
    print(f"👉 Verify that 'DAV2_Hybrid.ipynb' is inside: {NOTEBOOKS_DIR}\n")
    sys.exit(1)

# ==========================================================
# 2. MATCH DATA SET STRUCTURE PATHS (From your VS Code structure)
# ==========================================================
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "src", "data", "data", "processed", "train")
VAL_DATA_PATH = os.path.join(BASE_DIR, "src", "data", "data", "processed", "val")

# ==========================================================
# 3. CUSTOM NYU DATASET CLASS WITH ON-THE-FLY PREPROCESSING
# ==========================================================
class NYUDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Target directory missing: {data_dir}")
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        
        # Preprocessing pipeline required by the DA-V2 ViT Backbone
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),  # Rigid resolution matching ViT-B expectations
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        
        # Safe checkpoint loading for PyTorch 2.6+ environments containing NumPy structures
        data = torch.load(path, map_location='cpu', weights_only=False)
        
        # Images were stored as (H, W, C) from NumPy -> Convert to PyTorch (C, H, W)
        image = torch.from_numpy(data['image']).permute(2, 0, 1).float() / 255.0
        image = self.transform(image)
        
        # Depths were stored as (H, W) -> Add explicit channel dimension
        depth = torch.from_numpy(data['depth']).float().unsqueeze(0)
        
        # Interpolate the ground truth target to match the 518x518 processing space
        depth = torch.nn.functional.interpolate(depth.unsqueeze(0), 
                                                size=(518, 518), 
                                                mode='nearest').squeeze(0)
        return image, depth

# ==========================================================
# 4. SCALE-INVARIANT LOGARITHMIC LOSS (SILog)
# ==========================================================
class SILogLoss(torch.nn.Module):
    def __init__(self, lambd=0.5):
        super(SILogLoss, self).__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        # Small epsilon value (1e-6) injected to prevent invalid log(0) occurrences
        diff = torch.log(pred[valid_mask] + 1e-6) - torch.log(target[valid_mask] + 1e-6)
        loss = torch.sqrt(torch.mean(diff**2) - self.lambd * (torch.mean(diff)**2))
        return loss

# ==========================================================
# 5. MAIN ENGINE LOOP EXECUTOR
# ==========================================================
def run_training():
    # Initialize real-time tracking session on Weights & Biases cloud
    wandb.init(project="Monocular-3D-Reconstruction", name="hybrid-decoder-v1-run")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Console: Training session executing on device target: {device} ---")
    
    # Instantiate the custom model architecture constructed in your notebook
    model = load_hybrid_model(encoder='vitb', device=device)
    
    # Create Data Loaders (Keep low batch size for local memory stability)
    train_dataset = NYUDataset(TRAIN_DATA_PATH)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # CRITICAL PIPELINE DESIGN: Only optimize parameters bound to the trainable CNN decoder!
    optimizer = optim.Adam(model.custom_decoder.parameters(), lr=1e-4)
    criterion = SILogLoss()
    
    # Set up checkpoints output directory path string
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.train()
    epochs = 10
    
    print(f"--- Console: Kicking off optimization over {epochs} Epochs ---")
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, depths) in enumerate(pbar):
            images, depths = images.to(device), depths.to(device)
            
            optimizer.zero_grad()
            output = model(images)
            
            # Enforce spatial clamp boundaries to stop logarithmic training divergence
            output = torch.clamp(output, min=0.1, max=10.0)
            
            loss = criterion(output, depths)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Record individual batch steps inside W&B tracking charts
            wandb.log({"batch_loss": loss.item()})
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Completed successfully | Average SILog Loss: {avg_epoch_loss:.4f} ---")
        wandb.log({"epoch_loss": avg_epoch_loss})
        
        # Save historical model checkpoints state weights locally
        checkpoint_path = os.path.join(checkpoint_dir, "latest_hybrid_model.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint state successfully dumped locally to: {checkpoint_path}")

if __name__ == "__main__":
    run_training()