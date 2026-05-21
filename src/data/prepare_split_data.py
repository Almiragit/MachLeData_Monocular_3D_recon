import h5py
import numpy as np
import os
import torch
from tqdm import tqdm


def split_and_save_individual(mat_path, out_dir):
    print("--- Console: Opening .mat file ---")
    f = h5py.File(mat_path, 'r')
    images = f['images']
    depths = f['depths']

    num_samples = images.shape[0]
    indices = np.random.permutation(num_samples)

    # 80/10/10 Split
    splits = {
        'train': indices[:int(0.8 * num_samples)],
        'val': indices[int(0.8 * num_samples):int(0.9 * num_samples)],
        'test': indices[int(0.9 * num_samples):]
    }

    # Create directory structure: data/processed/train, etc.
    for name in splits.keys():
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)

    print("--- Console: Saving images individually to prevent RAM spikes ---")

    for name, idxs in splits.items():
        for i in tqdm(idxs, desc=f"Saving {name}"):
            # 1. Extract and fix orientation
            img = np.array(images[i]).transpose(2, 1, 0)  # (H, W, C)
            dep = np.array(depths[i]).transpose(1, 0)    # (H, W)

            # 2. Save as a single dictionary file per index
            # Example: data/processed/train/sample_105.pt
            sample_path = os.path.join(out_dir, name, f"sample_{i}.pt")
            torch.save({
                'image': img,
                'depth': dep
            }, sample_path)

    f.close()
    print(f"--- Success! Data saved to {out_dir} ---")


if __name__ == "__main__":
    # Keep paths aligned with DVC/configs
    split_and_save_individual(
        'data/nyu/raw/nyu_depth_v2_labeled.mat', 'data/nyu/processed')
