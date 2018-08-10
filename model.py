import torch
from torch import nn

class Autoencoder(nn.Module):
  def __init__(self):
    super(Autoencoder, self).__init__()

    self.encoder = nn.Sequential(
      nn.Linear(28 * 28, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 12),
      nn.ReLU(),
      nn.Linear(12, 2)
    )

    self.decoder = nn.Sequential(
      nn.Linear(2, 12),
      nn.ReLU(),
      nn.Linear(12, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 28 * 28),
      nn.Tanh()
    )

  def forward(self, x):
    z = self.encoder(x)
    xhat = self.decoder(z)
    return xhat
