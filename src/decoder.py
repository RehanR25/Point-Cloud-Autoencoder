import torch
from torch import nn 

class Decoder(nn.Module): 

    def __init__(self, latent_dim=512, grid_size=32):
        super().__init__()
        self.grid_size = grid_size
        self.n_points = grid_size * grid_size  #32*32 =1024

        # Fixed 2D grid, not learned, created once and reused every forward pass
        grid = self._build_grid(grid_size)
        self.register_buffer("grid", grid)

        # Fold 1: (2D grid co-ordination + latent) -> First 3D guess
        self.fold1 = nn.Sequential(
            nn.Conv1d(2 + latent_dim, 512, 1), nn.ReLU(), 
            nn.Conv1d(512, 512, 1), nn.ReLU(), 
            nn.Conv1d(512, 3, 1),
        )

        # Fold 2: ( Fold1 output + latent) -> Refined 3D position
        self.fold2 = nn.Sequential(
            nn.Conv1d(3 + latent_dim, 512, 1), nn.ReLU(),
            nn.Conv1d(512, 512, 1), nn.ReLU(), 
            nn.Conv1d(512, 3, 1),
        )

    # Creates a flat square grid of (u, v) coordinates, e.g. -1 to 1
    def _build_grid(self, grid_size): 
        # Creates grid_size evenly spaved values from -1 to 1 along each axis
        x = torch.linspace(-1, 1, grid_size)
        y = torch.linspace(-1, 1, grid_size)

        # Bild every combination of (x,y) pairs -> a full 2D grid
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        # Flatten from (grid_size, grid_size) into (grid_size*grid_size, ) each
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        # Stack into shape (n_points, 2), each rpw is one (x, y) coordinate
        grid = torch.stack([grid_x, grid_y], dim=1)

        return grid


    def forward(self, x): 

        # x : (B, latent_dim)
        B = x.shape[0]

        # (B, n_points, 2)
        grid = self.grid.unsqueeze(0).expand(B, -1, -1)  
        # (B, n_points, latent_dim)
        x_expanded = x.unsqueeze(1).expand(-1, self.n_points, -1)

        # --- Fold 1 --- 
        # (B, n_points, 2)
        x = torch.cat([grid, x_expanded], dim=2)
        # (B, 2 + latent_dim, n_points) for Conv1d
        x = x.transpose(1,2)
        # (B, n_points, 3)
        fold1_out = self.fold1(x).transpose(1, 2)

        # --- Fold 2 --- 
        # (B, n_points, 3 + latent_dim)
        x = torch.cat([fold1_out, x_expanded], dim=2)

        x = x.transpose(1, 2)
        # (B, n_points, 3)
        fold2_out = self.fold2(x).transpose(1, 2)

        return fold2_out

    # Decoder being changed to FoldingNet-style 