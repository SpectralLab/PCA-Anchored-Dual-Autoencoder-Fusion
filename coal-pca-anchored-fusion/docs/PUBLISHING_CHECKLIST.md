# GitHub release checklist

- [ ] Upload only the cleaned `coal-pca-anchored-fusion-github` directory or its clean ZIP.
- [ ] `python scripts/check_release.py` passes.
- [ ] `git status --short` shows no source spectra, target arrays, model `.pt` files, predictions, logs, `.npz` files or local `.env`.
- [ ] `python -m pytest -q` passes using **synthetic** data.
- [ ] Public text describes original checkpoint replay separately from reconstructed AE training.
- [ ] Contact information, repository ownership and chosen license have been verified by the authors.
- [ ] No claim that the public package alone reproduces the original three-seed AE pretraining precisely.
- [ ] If providing any scientific data later, secure explicit data-owner approval and re-audit privacy and provenance.
