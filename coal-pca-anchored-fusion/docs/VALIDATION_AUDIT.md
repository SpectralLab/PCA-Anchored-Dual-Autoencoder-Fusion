# Source-only release audit (2026-11-02)

## Source inclusion check

- No real spectral matrices, laboratory targets, physical sample records, original predictions or saved AE/ANN weight files included.
- No `.npz`, `.npy`, `.csv`, `.tsv`, `.pt`, `.pth`, `.ckpt`, `.zip`, `.mat`, `.xlsx` or equivalent data/model artifacts included.
- The untrusted original code/data handoff ZIP files are **not** included.
- The package generates entirely synthetic test matrices locally for smoke tests.

## Runtime checks (executed in the preparation environment)

- Python unit tests: 6 passed (synthetic only).
- `coal-fusion demo-data`: succeeded.
- `coal-fusion fresh` on synthetic S1/seed11: succeeded and generated paired PCA24/AE24 metrics.
- `coal-fusion pls` on synthetic data (max LV 3): succeeded for five held-out synthetic source groups.
- `coal-fusion summarize`: succeeded.
- `python scripts/check_release.py`: passed.

## Private historical-checkpoint integrity check

A one-fold test (KF, seed 7) was performed **outside this release tree**, with the data and checkpoint weights retained only in a private temporary directory. The maximum absolute difference against the archived per-sample predictions was:

| Comparison | Maximum absolute difference (pp) |
|---|---:|
| NIR-PCA12 ANN | 0.00002289 |
| NIR-AE12 ANN | 0.00001144 |
| Dual PCA24 ANN | 0.00001717 |
| Dual AE24 ANN | 0.00000954 |

This confirms close downstream replay for that one checked fold; it does not verify that the **lost original AE pretraining** was reconstructed exactly, nor does it validate every original fold. None of those private test inputs, outputs or checkpoint files are part of the GitHub package.

## Release provenance

See `PROVENANCE.md`. Further independent review of source permissions and choice of software license remains the responsibility of the repository owner.
