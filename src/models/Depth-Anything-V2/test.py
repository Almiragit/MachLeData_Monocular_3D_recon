import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import tkinter as tk
from tkinter import filedialog
import open3d as o3d

from depth_anything_v2.dpt import DepthAnythingV2


# -----------------------------
# POINT CLOUD + VIEWER
# -----------------------------
def create_and_view_point_cloud(depth, image):
    h, w = depth.shape

    fx = fy = 500
    cx, cy = w / 2, h / 2

    points = []
    colors = []

    for y in range(h):
        for x in range(w):
            z = depth[y, x] / 255.0

            if z <= 0:
                continue

            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            Z = z

            points.append([X, Y, Z])
            colors.append(image[y, x] / 255.0)

    points = np.array(points)
    colors = np.array(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# -----------------------------
# MAIN
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--img-path', type=str, default=None)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')

    parser.add_argument('--encoder', type=str, default='vitb',
                        choices=['vits', 'vitb', 'vitl', 'vitg'])

    parser.add_argument('--pred-only', dest='pred_only', action='store_true')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    print("Loading model...")

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(
        torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu')
    )
    depth_anything = depth_anything.to(DEVICE).eval()

    print("Model loaded!")

    # -----------------------------
    # FILE PICKER
    # -----------------------------
    if args.img_path is None:
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )

        if not file_path:
            print("No image selected. Exiting...")
            exit()

        filenames = [file_path]
    else:
        filenames = [args.img_path]

    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # -----------------------------
    # PROCESS IMAGE
    # -----------------------------
    for k, filename in enumerate(filenames):
        print(f'Processing: {filename}')

        raw_image = cv2.imread(filename)
        rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        # Depth prediction
        depth = depth_anything.infer_image(raw_image, args.input_size)

        depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_norm = depth_norm.astype(np.uint8)

        # Save depth map
        name = os.path.splitext(os.path.basename(filename))[0]

        depth_path = os.path.join(args.outdir, f"{name}_depth.png")
        cv2.imwrite(depth_path, depth_norm)

        # Color visualization
        if args.grayscale:
            depth_vis = np.repeat(depth_norm[..., None], 3, axis=-1)
        else:
            depth_vis = (cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        vis_path = os.path.join(args.outdir, f"{name}_vis.png")

        if args.pred_only:
            cv2.imwrite(vis_path, depth_vis)
        else:
            split = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined = cv2.hconcat([raw_image, split, depth_vis])
            cv2.imwrite(vis_path, combined)

        # -----------------------------
        # POINT CLOUD + SAVE + VIEW
        # -----------------------------
        print("Generating 3D point cloud...")

        pcd = create_and_view_point_cloud(depth_norm, rgb_image)

        pcd_path = os.path.join(args.outdir, f"{name}_pointcloud.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)

        print("Point cloud saved:", pcd_path)

        # -----------------------------
        # OPEN 3D VIEWER
        # -----------------------------
        print("Opening 3D viewer...")
        o3d.visualization.draw_geometries([pcd])

        print("Done!")