"""
Visualize original vs reconstructed point clouds to sanity-check the
autoencoder, since Chamfer loss alone can be misleading (mode collapse
can still produce a low loss number).

Edit the variables below, then run: python visualize_reconstructions.py
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from autoencoder import AutoEncoder
from dataset import PointCloudDataset

# ---------------------------------------------------------------------------
# Edit these
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = r"checkpoints\checkpoint.pt"
DATA_DIR = r"C:\Users\rehan\Documents\3D image\data\ModelNet40_pointclouds"
N_POINTS = 1024
LATENT_DIM = 512
N_SAMPLES = 10
SAVE_PATH = "pics/reconstruction_comparison25.png"
# ---------------------------------------------------------------------------


def plot_comparison(original, reconstructed, save_path=None, n_samples=4):
    """
    original, reconstructed: numpy arrays of shape (n_samples, N, 3)
    Plots original (top row) vs reconstructed (bottom row) side by side.
    """
    fig = plt.figure(figsize=(4 * n_samples, 8))

    for i in range(n_samples):
        ax1 = fig.add_subplot(2, n_samples, i + 1, projection="3d")
        pts = original[i]
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, 2], cmap="viridis")
        ax1.set_title(f"Original {i}")
        ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])

        ax2 = fig.add_subplot(2, n_samples, n_samples + i + 1, projection="3d")
        pts_r = reconstructed[i]
        ax2.scatter(pts_r[:, 0], pts_r[:, 1], pts_r[:, 2], s=2, c=pts_r[:, 2], cmap="viridis")
        ax2.set_title(f"Reconstructed {i}")
        ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved comparison plot to {save_path}")
    plt.show()
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = AutoEncoder(latent_dim=LATENT_DIM, n_points=N_POINTS).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    # Handle either a raw state_dict or a dict with a "model_state_dict" key
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()


    # Load a few samples from the dataset
    dataset = PointCloudDataset(data_dir=DATA_DIR, n_points=N_POINTS)

    """
    
    modelnet10_categories = {
        "bathtub", "bed", "chair", "desk", "dresser", 
        "monitor", "night_stand", "sofa", "table", "toilet"
    }
    categories = {"airplane", "bookshelf", "bottle", "cone", "curtain", "flower_pot", "guitar", "keyboard",
                    "piano", "plant", "tent", "tv_stand","vase","xbox"}

    dataset.files = [ 
        f for f in dataset.files
        if os.path.basename(os.path.dirname(os.path.dirname(f))) in categories
    ]"""

    print(f"Total files after filtering: {len(dataset.files)}")

    indices = np.random.choice(len(dataset), size=N_SAMPLES, replace=False)

    originals = torch.stack([dataset[i] for i in indices]).to(device)

    with torch.no_grad():
        reconstructions = model(originals)

    originals_np = originals.cpu().numpy()
    reconstructions_np = reconstructions.cpu().numpy()

    plot_comparison(originals_np, reconstructions_np,
                     save_path=SAVE_PATH, n_samples=N_SAMPLES)


if __name__ == "__main__":
    main()