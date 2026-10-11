import torch 
from torch import nn 

class Encoder(nn.Module): 
# the latent dimension can changed 
# for either more compression and less detail 
# vs less compression and more detail preserved
    def __init__(self, latent_dim=512): 
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(), 
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(), 
            nn.Conv1d(256, latent_dim, 1), nn.BatchNorm1d(latent_dim)
        )
    def forward(self, x): 
        x = x.transpose(1, 2) # turns (B, N, 3) into (B, 3, N) for Conv1d
        x = self.mlp(x)
        # takes the max value across n points dimension per feature channel
        # so the order of the point does not matter
        x = torch.max(x, dim=2)[0] 
        return x 