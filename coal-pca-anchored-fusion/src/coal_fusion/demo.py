"""Completely synthetic example with deliberately non-physical random spectra."""
from pathlib import Path
import numpy as np


def create_synthetic_data(path, n_sources=5, n_per_source=22, nir_channels=28,
                          raman_channels=45, seed=2026):
    rng = np.random.default_rng(seed)
    labels = [f"S{i+1}" for i in range(n_sources)]
    n = n_sources * n_per_source
    source = np.repeat(labels, n_per_source)
    latents = rng.normal(size=(n, 5))
    source_effect = np.repeat(rng.normal(scale=0.15, size=(n_sources, 1)), n_per_source, axis=0)
    latent_mixture = latents + source_effect
    nir = latent_mixture @ rng.normal(size=(5, nir_channels)) + 0.2 * rng.normal(size=(n, nir_channels))
    raman = latent_mixture @ rng.normal(size=(5, raman_channels)) + 0.2 * rng.normal(size=(n, raman_channels))
    target = (18 + 2 * latents[:, 0] - latents[:, 1] + 0.2 * latents[:, 2] ** 2
              + 0.15 * rng.normal(size=n))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xn=nir.astype(np.float32), xr=raman.astype(np.float32),
                        y=target.astype(np.float64), sources=source,
                        ids=np.asarray([f"FAKE_{i+1:04d}" for i in range(n)]))
    return path
