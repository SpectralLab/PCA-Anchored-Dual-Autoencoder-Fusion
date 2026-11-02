"""Optional documented replicate-level spectral preprocessing functions.

These are utilities, not automatically applied by the modeling CLI. Input .npz
matrices must already be preprocessed and reduced to one vector per sample.
"""
import numpy as np
from scipy.signal import savgol_filter


def sg_derivative_snv(spectra, window_length, polyorder=2):
    """Apply SG first derivative, then SNV, to each replicate spectrum.

    Args:
        spectra: array (..., n_wavelengths); no sample identifiers.
        window_length: NIR=11 or Raman=15 in the documented experiment.
    """
    x = np.asarray(spectra, dtype=float)
    if x.ndim < 2 or x.shape[-1] < window_length:
        raise ValueError("Expected spectra with trailing spectral axis")
    if window_length % 2 != 1 or window_length <= polyorder:
        raise ValueError("window_length must be odd and greater than polyorder")
    derived = savgol_filter(x, window_length, polyorder, deriv=1, axis=-1)
    centered = derived - derived.mean(axis=-1, keepdims=True)
    std = derived.std(axis=-1, keepdims=True)
    return centered / np.maximum(std, 1e-12)


def preprocess_and_average_replicates(sample_replicates, window_length):
    """Preprocess each replicate before averaging, preserving sample boundaries.

    sample_replicates: iterable of arrays shaped (n_replicates, n_channels).
    Every sample may have a different n_replicates but same n_channels.
    """
    result = []
    for reps in sample_replicates:
        reps = np.asarray(reps)
        if reps.ndim != 2 or not len(reps):
            raise ValueError("Each sample must have at least one 1D replicate")
        result.append(sg_derivative_snv(reps, window_length).mean(axis=0))
    return np.asarray(result, dtype=np.float32)
