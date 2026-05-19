import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# ==========================================================
# 1. PATH SETUP
# ==========================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO_PATH = os.path.join(BASE_DIR, 'src', 'models', 'Depth-Anything-V2')

if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

from depth_anything_v2.dpt import DepthAnythingV2

TRAIN_DIR = os.path.dirname(__file__)
if TRAIN_DIR not in sys.path:
    sys.path.append(TRAIN_DIR)

try:
    from train import load_hybrid_model
except ImportError:
    sys.path.append(os.path.join(BASE_DIR, "src", "training"))
    from train import load_hybrid_model

# ==========================================================
# 2. FILE SELECTION
# ==========================================================
def select_image_file():
    root = tk.Tk()
    root.withdraw()      
    root.attributes("-topmost", True) 
    file_path = filedialog.askopenfilename(
        title="Select an Image for Blended Depth Analysis",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    root.destroy()       
    return file_path

# ==========================================================
# 3. MAIN EXECUTION PIPELINE
# ==========================================================
def main():
    # 🎛️ TUNING KNOB: Change this to control the blending ratio!
    # 0.0 = Pure Hybrid Model | 1.0 = Pure Base DA-V2 Model
    ALPHA = 0.65 

    selected_path = select_image_file()
    if not selected_path: return
        
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    weights_path = next((p for p in [
        os.path.join(BASE_DIR, "latest_hybrid_model.pth"),
        os.path.join(BASE_DIR, "src", "training", "latest_hybrid_model.pth"),
        os.path.join(BASE_DIR, "checkpoints", "latest_hybrid_model.pth")
    ] if os.path.exists(p)), None)

    # Load Base DA-V2
    model_config = {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
    vanilla_model = DepthAnythingV2(**model_config)
    vanilla_backbone_path = os.path.join(REPO_PATH, 'checkpoints', 'depth_anything_v2_vitb.pth')
    vanilla_model.load_state_dict(torch.load(vanilla_backbone_path, map_location='cpu'))
    vanilla_model = vanilla_model.to(DEVICE).eval()

    # Load Custom Hybrid Network
    hybrid_model = load_hybrid_model(encoder='vitb', device=DEVICE)
    hybrid_model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    hybrid_model.eval()

    # Process Images
    raw_image = cv2.imread(selected_path)
    h, w, _ = raw_image.shape
    
    input_size = 518
    img_input = cv2.resize(raw_image, (input_size, input_size))
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_input = (img_input - mean) / std
    tensor_input = torch.from_numpy(img_input).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

    print("🔮 Extracting predictions from both models...")
    with torch.no_grad():
        # Map 1: Native Base Model Output
        dav2_depth = vanilla_model.infer_image(raw_image, input_size=518)
        
        # Map 2: Hybrid Custom Output
        hybrid_raw = hybrid_model(tensor_input)
        hybrid_up = torch.nn.functional.interpolate(hybrid_raw, size=(h, w), mode="bilinear", align_corners=False)
        hybrid_depth = torch.squeeze(hybrid_up).cpu().numpy()

    if hybrid_depth.ndim > 2: hybrid_depth = hybrid_depth[0]

    # Normalize both cleanly to a shared 0.0 - 1.0 floating point baseline
    dav2_scaled = (dav2_depth - dav2_depth.min()) / (dav2_depth.max() - dav2_depth.min() + 1e-8)
    hybrid_scaled = (hybrid_depth - hybrid_depth.min()) / (hybrid_depth.max() - hybrid_depth.min() + 1e-8)

    # 🚀 THE ENSEMBLE BLEND MATRIX MATH
    # We mix the structural sharpness of DA-V2 with the contextual positioning of your Hybrid model
    blended_depth = (ALPHA * dav2_scaled) + ((1.0 - ALPHA) * hybrid_scaled)
    
    # Scale back to 8-bit image spectrum
    final_output_map = (blended_depth * 255.0).astype(np.uint8)

    # ==========================================================
    # 4. RENDER VISUAL DISPLAY FOR PROJECT PRESENTATION
    # ==========================================================
    print("🖥️ Rendering side-by-side comparison screen...")
    plt.figure(figsize=(16, 8))
    
    # Left View: Baseline reference
    plt.subplot(1, 2, 1)
    plt.title("Base DA-V2 Crisp Baseline", fontsize=13, fontweight='bold')
    plt.imshow((dav2_scaled * 255.0).astype(np.uint8), cmap='Spectral_r')
    plt.axis('off')
    
    # Right View: Your strategically blended hybrid result
    plt.subplot(1, 2, 2)
    plt.title(f"Optimized Hybrid Ensemble (Alpha Blend: {ALPHA})", fontsize=13, fontweight='bold')
    plt.imshow(final_output_map, cmap='Spectral_r')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()