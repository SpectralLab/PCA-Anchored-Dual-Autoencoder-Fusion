# Evaluation and reproducibility notes

1. **LOSO split.** Entire source groups held out once each. Evaluation predictions are paired by physical sample and source group.
2. **Primary matched comparison.** PCA24 and AE24 use identical preprocessed NIR/Raman samples, 12 dimensions per modality and the same downstream ANN architecture, optimization settings and seed pairing.
3. **AE pretraining.** The publicly released `fresh` procedure is reconstructed and frozen, not an exact recovery of the original AE-pretraining implementation. Original three-seed historical AE weights are private and optional in `archived` mode.
4. **Training/validation flow.** All transforms and internal training-duration selection use only outer calibration sources. The source-stratified 80:20 split is used to select the ANN training length. AE training likewise chooses epochs on a calibration-only internal validation subset.
5. **Ensemble.** Seed-averaged predictions are a separate experiment and must not be conflated with mean RMSE of separately evaluated models.
6. **Statistics.** RMSE averaged across seeds is not the same quantity as pooled RMSE after averaging seed predictions. Between-seed SD addresses training variability, not population-level source uncertainty. Five coal sources limit inference about an entire universe of new sources.
7. **Privacy.** Locally generated `predictions_PRIVATE.csv` includes reference targets and physical sample identifiers. Do not publish. Model checkpoint `.pt` files and `.npz` spectra/labels are intentionally absent.
8. **PLS.** The included PLS baseline is a defined inner-source-CV protocol; changing preprocessing/feature scaling/LV search produces different results and should be recorded before consulting held-out errors.
9. **Preprocessing.** `fresh` and `archived` start from the per-sample processed matrices, not from raw instrument signal and acquisition metadata.
