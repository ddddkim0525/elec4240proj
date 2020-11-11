import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True)
        )
        self. decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        code = self.encoder(x)
        reconstruct = self.decoder(code)
        return reconstruct
