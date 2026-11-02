# PCA-Anchored Dual-Autoencoder Fusion of Raman and NIR Spectra

**Research code for coal volatile-matter regression with source-held-out evaluation.**

This repository organizes code associated with the manuscript *PCA-Anchored Dual-Autoencoder Fusion of Raman and Near-Infrared Spectra for Cross-Source Coal Volatile Matter Prediction*. It contains **no real coal spectra, chemical reference values, sample identifiers, model checkpoints, or prediction outputs**.

> **Important provenance distinction:** The original AE-pretraining source was not completely preserved in the archived project. The AE **architecture** was reconstructed and checked against archived inference. The independent-training implementation here was frozen for a later 10-seed exploratory repeat; it is **not a bitwise reproduction of the original three-seed AE pretrainer**. See [PROVENANCE.md](PROVENANCE.md) before using the code to interpret published metrics.

## Model overview

Each modality has a 12-dimensional representation:

```text
NIR spectral vector -> fixed NIR PCA(12) + nonlinear NIR residual AE -> z_NIR (12)
Raman spectral vector -> fixed Raman PCA(12) + nonlinear Raman residual AE -> z_Raman (12)
                                                            |
                                         concat(z_NIR, z_Raman) -> 24 features
                                                            |
                                  standardization -> compact ANN (32-32-16) -> V_ad
```

The PCA branch is **added** to the trainable nonlinear residual branch within each modality, **not concatenated** with that branch. The final NIR/Raman representations are concatenated. This is a fixed-PCA **additive correction**, not mathematically separate linear/nonlinear components.

## Install

Python 3.9+ required; a virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate         # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
python -m pytest -q
```

On hardware without a CUDA device, PyTorch can run on CPU. Multi-seed, five-source training may take substantial time. The code defaults to single-thread BLAS in the CLI for repeatability; results may vary across library versions/platforms.

## Run without private data (smoke test)

Generate an entirely synthetic dataset **on your own machine**:

```bash
coal-fusion demo-data --out data/synthetic_demo.npz
coal-fusion fresh --data data/synthetic_demo.npz --out runs/demo \
  --holds S1 --seeds 11 --components 6 --ae-epochs 5 --ann-epochs 15
```

`data/` and `runs/` are git-ignored. Synthetic-data outcomes have **no scientific meaning**.

## Prepare your own private data

The command-line tool expects a **locally supplied** `.npz` file with **one physical coal sample per row**. The input matrices must already have undergone the study's replicate-level preprocessing (SG first derivative and SNV, each replicate processed before the within-sample average).

| Array | Required shape | Description |
|---|---|---|
| `xn` | `(N, 239)` in original setup | Preprocessed NIR float matrix |
| `xr` | `(N, 1024)` in original setup | Preprocessed Raman float matrix |
| `y` | `(N,)` | Air-dried volatile matter, wt.% |
| `sources` | `(N,)` | Plant/source labels (e.g. source groups) |
| `ids` | `(N,)` | Unique **physical sample** IDs |

The research dataset comprised 605 samples across five sources, but this **is not distributed**. More details: [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## Reconstructed independent AE pretraining (exploratory)

Use the newly frozen AE training implementation; nothing is loaded from original checkpoints.

```bash
coal-fusion fresh \
  --data /private/path/to/preprocessed_coal.npz \
  --out runs/independent_10seed \
  --seeds 11,23,47,71,101,131,173,211,257,307
```

For every seed, all source-held-out folds fit PCA, new NIR and Raman AEs, and matched ANN regressors **without using the held-out source for fitting or checkpoint selection**. It writes a per-fold metrics CSV, a seed-paired PCA24 vs AE24 CSV, and a **private** per-sample predictions CSV in the ignored output folder.

Default: 12 PCA components per modality, 64 hidden units in the residual AE, 100 AE epochs maximum, 32-32-16 ANN and up to 300 epochs, source-stratified internal validation (random_state=817), dropout 0.1, AdamW optimizer. See `src/coal_fusion/training.py` and `PROVENANCE.md` for full details.

## Historical checkpoint replay (original three-seed comparisons)

If you have the **trusted original private AE checkpoint archive**, prepare the path structure on your own computer:

```text
/private/outer/
  KF/NIRAE_seed7.pt          KF/RamanAE_seed7.pt
  KF/NIRAE_seed123.pt        KF/RamanAE_seed123.pt
  ...
  YX/NIRAE_seed3407.pt      YX/RamanAE_seed3407.pt
```

Then run:

```bash
coal-fusion archived \
  --data /private/path/to/preprocessed_coal.npz \
  --checkpoints /private/outer \
  --out runs/historical_checkpoint_replay \
  --seeds 7,123,3407
```

This re-evaluates the **fixed historical AE encoders** and trains downstream ANN with four inputs: NIR PCA12, NIR AE12, dual PCA24, and dual AE24. It does **not** independently reconstruct how the historical checkpoint weights were pretrained. **Only load checkpoint files you trust** (`torch.load` uses Python object deserialization for this historical format).

The original manuscript's three-seed RMSE values must not be combined statistically with the independent 10-seed reconstruction as if they were identically executed AE-pretraining runs.

## PLS references

```bash
coal-fusion pls --data /private/path/to/preprocessed_coal.npz \
  --out runs/pls --max-lv 20
```

This runs NIR-only and early-fusion Raman+NIR PLS with the latent-variable count selected **only using calibration-source internal leave-one-source-out** cross-validation. Inputs are standardized on the training subset of each inner/outer fold. This is one explicitly specified PLS protocol, not an exhaustive search over preprocessing variants.

## Outputs / evaluation protocol

- **Five-fold leave-one-whole-source-out:** train on four source groups and test on the excluded group.
- **No tuning on the excluded source:** PCA, AE, scalers, PLS latent variables and ANN epoch selection are fit/selected using calibration sources.
- **Independent physical samples:** repeated spectra must be processed and averaged per sample before constructing the matrix.
- **Metrics:** source-specific and pooled RMSE, MAE, R², signed bias; for the AE comparison, matched seed-wise RMSE differences.
- **Do not mistake repeat-seed uncertainty for source-population uncertainty.** A small across-seed SD is not a confidence interval for performance on new sources.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the validation and interpretation boundaries.

## Repository structure

```text
src/coal_fusion/
  model.py       Archived architecture + ANN network
  training.py    Reconstructed AE pretraining, ANN fit, historical checkpoint replay
  evaluation.py  Leave-one-source-out evaluation + PLS
  spectral.py    Optional SG derivative + SNV replicate preprocessing utilities
  data.py        Strict private-data format checks
  demo.py        Synthetic-data generator only
  cli.py         Command-line entry points
tests/           Synthetic tests; no genuine measurements
scripts/         Local release privacy audit
docs/            Data format, validation, release checklist
```

## Data availability and licensing

No real spectra/labels/checkpoints are included. Users must supply data they have a right to use. Run [scripts/check_release.py](scripts/check_release.py) before publishing. Outputs are git-ignored because predictions contain reference targets and IDs. The project author should choose a code license before making the repository public; none is assumed here.

For Chinese instructions see [README_CN.md](README_CN.md).
