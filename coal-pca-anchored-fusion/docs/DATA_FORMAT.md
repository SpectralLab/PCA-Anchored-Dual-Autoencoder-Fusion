# Input data format (private, not provided)

Only local preprocessed numerical arrays are accepted. The CLI expects an `.npz` archive with:

```python
import numpy as np

# Illustration ONLY. The arrays below must come from a lawful PRIVATE dataset.
# DO NOT run with actual samples in this public repository or commit .npz files.
np.savez_compressed(
    "/outside/this/repository/private_coal_data.npz",
    xn=X_nir,           # shape (N, channels_nir); NIR 239 variables in the study
    xr=X_raman,         # shape (N, channels_raman); Raman 1024 variables in the study
    y=volatile_matter,  # shape (N,); air-dried volatile matter, wt.%
    sources=plants,     # shape (N,); e.g., strings of factory/source group
    ids=physical_ids,   # shape (N,); one identifier per physical coal sample
)
```

The following preprocessing belongs upstream of the `.npz` matrix preparation: measure repeats from one coal cup; apply Savitzky–Golay first derivative to each accepted spectrum (polynomial order 2; 11-point NIR / 15-point Raman in the manuscript), then standard normal variate (SNV), then average repeats **within the same physical sample**. The Python `spectral.py` helpers implement those operations if you have lawfully obtained individual replicate arrays, but no instrument file parser or automatic replicate-matching logic is included.

**Important boundary:** original data dimensions and sample counts are documentation, **not files supplied by this release**. The program validates that each sample ID is unique. All PCA, AE, feature scaling, PLS and ANN fitting takes place on the calibration-source subset only in each LOSO fold.

No data download endpoint is provided; contact the data owner regarding appropriate scientific data access.
