# Source provenance and scope

This release is a **curated research-code package**. It is not the missing historical code archive reproduced verbatim.

## 1. Historical 2026-10-31 global-ANN experiment

- Manuscript primary result: dual PCA24–ANN mean pooled RMSE 1.9787 percentage points and dual AE24–ANN 1.7880 percentage points, across three fixed seeds (7, 123, 3407).
- The original downstream ANN code and state-dict-compatible PCA-anchored AE **inference** architecture were preserved/recovered in the 2026-10-31 project handoff. The source was refactored into `model.py` and `training.py`.
- Original *pretraining* source for these fixed AE checkpoints was **not** completely preserved. The model's *weights* exist in the private archive, but they are not included here.
- `coal-fusion archived` loads those locally supplied, trusted checkpoints and refits/evaluates the archived-style global ANN. It is **checkpoint replay**, not from-scratch original AE reproduction.

## 2. Independent 10-seed repeat on 2026-11-02

- `coal-fusion fresh` follows the frozen reconstructed AE protocol from `complete_retrain.py` in the experimental handoff. There are 10 independent seeds: 11, 23, 47, 71, 101, 131, 173, 211, 257, and 307.
- Additive PCA branch is fixed; residual 2-layer MLP is 64 hidden units. Residual last encoder/decoder layers use Gaussian initialization SD 0.001. The pretraining reconstruction loss is normalized by the **outer calibration** PCA reconstruction MSE (fixed scale).
- The AE internal split is 80:20 stratified by source using random_state 817, with checkpoint selection every five epochs up to 100 epochs. After selecting an epoch, the AE is reinitialized and fit to the full outer calibration set for that number of epochs.
- The same 32-32-16 ANN architecture, source-stratified internal split and post-selection full calibration refit are used for PCA and AE regressors. Output targets are standardized using outer calibration targets only. Both models use the same random seed per outer fold.
- `fresh` is internally consistent; it is not identical to lost historic pretraining code, and its results **must not be pooled with three original AE runs as one 13-run experimental cohort**.

## 3. Critical limitations and author verification

- Random-seed restarts of the same five source groups quantify sensitivity to training initialization; they do **not** independently establish a confidence interval for unseen sources.
- Prior method exploration on the same 605-sample development cohort means LOSO performance should not be portrayed as a prospective, never-before-seen external validation.
- The supplied `.npz` must contain the **already preprocessed** one-row-per-physical-sample matrices. `spectral.py` provides utility preprocessing, but does not automatically transform raw repeated instrument exports into the matrix.
- Checkpoint `.pt` loading in historic compatibility mode uses `weights_only=False` to support the source format. Only load checkpoints from a trusted origin.
- Numeric results may vary with CPU/GPU and library versions. The original manuscript values are **not asserted to be reproduced in every fresh environment**.
- No license is assigned until the code owner chooses one.
