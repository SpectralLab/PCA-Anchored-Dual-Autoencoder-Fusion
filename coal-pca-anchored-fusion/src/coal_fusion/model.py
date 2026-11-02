"""Network definitions reconstructed/retained from the experiment archive."""
import numpy as np
import torch
from torch import nn


class AnchoredAE(nn.Module):
    """PCA-projection + nonlinear residual; PCA branch may be frozen.

    For a centered spectrum x-mu:
        z = PCA(x-mu) + residual_encoder(x-mu)
        x_hat = PCA_backprojection(z) + residual_decoder(z) + mu.

    Constructor matches the archived model state-dict naming and shapes.
    This is a reconstruction of archived inference architecture, NOT the
    lost original AE pretraining source.
    """

    def __init__(self, mean, components, width=64, freeze=True):
        super().__init__()
        mean = torch.as_tensor(np.asarray(mean).copy(), dtype=torch.float32)
        components = torch.as_tensor(np.asarray(components).copy(), dtype=torch.float32)
        if mean.ndim != 1 or components.ndim != 2 or components.shape[1] != mean.numel():
            raise ValueError("mean and PCA components have incompatible dimensions")
        nin, nz, width = int(mean.numel()), int(components.shape[0]), int(width)
        self.register_buffer("mean", mean.clone())
        self.encoder = nn.Linear(nin, nz, bias=False)
        self.decoder = nn.Linear(nz, nin, bias=False)
        self.enc_residual = nn.Sequential(nn.Linear(nin, width), nn.GELU(), nn.Linear(width, nz))
        self.dec_residual = nn.Sequential(nn.Linear(nz, width), nn.GELU(), nn.Linear(width, nin))
        with torch.no_grad():
            self.encoder.weight.copy_(components)
            self.decoder.weight.copy_(components.T)
        self.encoder.weight.requires_grad_(not freeze)
        self.decoder.weight.requires_grad_(not freeze)

    def forward(self, x):
        centered = x - self.mean
        latent = self.encoder(centered) + self.enc_residual(centered)
        reconstructed = self.decoder(latent) + self.dec_residual(latent) + self.mean
        return reconstructed, latent


class GlobalANN(nn.Module):
    """Original downstream regression architecture: 32-32-16, GELU, dropout."""

    def __init__(self, features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, 32), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(32, 32), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)


def encode(model, matrix):
    """Evaluate pretrained encoder in inference mode."""
    model.eval()
    with torch.no_grad():
        _, z = model(torch.as_tensor(np.asarray(matrix), dtype=torch.float32))
    return z.detach().cpu().numpy()
