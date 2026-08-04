import torch 
import glob
import os 
import numpy as np 
from torch.utils.data import Dataset, DataLoader, random_split

val_fraction = 0.1

class PointCloudDataset(Dataset): 
    "Loads pre-sample point clouds saved as individual .npy files"

    def __init__(self, data_dir, n_points=1024):
        self.files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True))
        if not self.files: 
            raise RuntimeError(f"No .npy point clouds found in {data_dir}")
        self.n_points = n_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Calls every time DataLoader wants an example 
        # Safety check to ensure that index has 1024 points
        pts = np.load(self.files[index]).astype(np.float32)
        if pts.shape[0] != self.n_points : 
            # resample if the file has a different point cloud
            # if file has fewer than 1024 points, then some indices are repeated
            # so that the output has exactly 1024 
            # if file has more than 1024 points, then 1024 indices are randomly selected 
            # so that the output has exactly 1024
            choice = np.random.choice(pts.shape[0], self.n_points, replace= (pts.shape[0] < self.n_points))
            pts = pts[choice]
        return torch.from_numpy(pts)

def _get_dataloader_():

    dataset = PointCloudDataset(data_dir=r"data\ModelNet40_pointclouds", n_points=1024) # n_points redundant but good to know 

    modelnet10_categories = {
        "bathtub", "bed", "chair", "desk", "dresser", 
        "monitor", "night_stand", "sofa", "table", "toilet"
    }

    dataset.files = [ 
        f for f in dataset.files
        if os.path.basename(os.path.dirname(os.path.dirname(f))) not in modelnet10_categories
    ]

    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size

    train_dataset, valid_dataset = random_split(
        dataset, [train_size, val_size], 
        generator = torch.Generator().manual_seed(42) # to ensure reproducibility
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True) # important for the encoder
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, drop_last=False)

    return train_loader, valid_loader