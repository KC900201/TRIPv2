import torch
import torch.nn as nn


# Initialize discriminator class (basic)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential()

    def forward(self, x):
        return self.disc(x)
