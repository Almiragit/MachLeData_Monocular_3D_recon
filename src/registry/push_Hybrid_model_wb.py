import os
import sys
import wandb

# ==========================================================
# 1. RESOLVE SYSTEM PATHS FROM SRC/REGISTRY
# ==========================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
weights_path = os.path.join(BASE_DIR, "checkpoints", "latest_hybrid_model.pth")

# Fallback search if checkpoints directory is missing
if not os.path.exists(weights_path):
    weights_path = os.path.join(BASE_DIR, "latest_hybrid_model.pth")

if not os.path.exists(weights_path):
    raise FileNotFoundError(
        f"❌ Could not find 'latest_hybrid_model.pth' automatically.\n"
        f"Please verify where your training loop dropped your weights file."
    )

# ==========================================================
# 2. RUN ID AUTO-LOOKUP ASSISTANT
# ==========================================================
print("🔍 Scanning your W&B workspace to find your active project runs...")
try:
    api = wandb.Api()
    # Pulls the active project runs under your profile
    runs = api.runs("Monocular-3D-Reconstruction")
    
    print("\n📋 FOUND RECENT RUNS IN YOUR PROJECT:")
    print("-" * 60)
    print(f"{'RUN DISPLAY NAME':<30} | {'8-CHARACTER RUN ID':<15}")
    print("-" * 60)
    for r in runs[:5]:  # Show the 5 most recent runs
        print(f"{r.name:<30} | {r.id:<15}")
    print("-" * 60)
except Exception:
    print("\n💡 Couldn't fetch your runs automatically (make sure you are logged into wandb via terminal).")
    print("You can find your 8-character ID on your web dashboard under the run overview panel.")

# Ask you directly via terminal input so you don't have to hardcode it!
print("\n📝 Paste or type your 8-character W&B Run ID from the list or dashboard above:")
YOUR_RUN_ID = input("👉 Enter Run ID: ").strip()

if not YOUR_RUN_ID:
    print("❌ Run ID cannot be blank. Exiting.")
    sys.exit()

# ==========================================================
# 3. CONNECT AND STREAM MODEL ARTIFACT
# ==========================================================
print(f"\n🔄 Connecting and merging into active run session: {YOUR_RUN_ID}...")
run = wandb.init(
    project="Monocular-3D-Reconstruction",
    id=YOUR_RUN_ID,
    resume="allow"
)

print(f"📦 Packaging model file into a secure W&B Artifact container...")
artifact = wandb.Artifact(
    name="depth_anything_hybrid_ensemble", 
    type="model",
    description="Optimized hybrid model architecture utilizing a metric-trained CNN decoder and a frozen DA-V2 transformer core."
)

# Add the file to the registry folder
artifact.add_file(weights_path)
run.log_artifact(artifact)

run.finish()
print(f"\n🎉 Successfully saved! Check the 'Artifacts' tab inside run {YOUR_RUN_ID} to view it.")