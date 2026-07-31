import torch 
from torch import nn
from encoder import Encoder
from decoder import Decoder

class AutoEncoder(nn.Module): 
    def __init__(self, latent_dim=512, n_points=1024):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, grid_size=32) # 32*32=1024

    def forward(self, x): 
        out = self.encoder(x)
        return self.decoder(out)