from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.decomposition import PCA
from coal_fusion.demo import create_synthetic_data
from coal_fusion.data import load_npz, SpectralData
from coal_fusion.model import AnchoredAE, GlobalANN
from coal_fusion.training import FitConfig, run_fresh_fold
from coal_fusion.evaluation import summarize, run_pls_baselines
from coal_fusion.spectral import sg_derivative_snv, preprocess_and_average_replicates


def test_anchored_pca_branch_frozen():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 15)).astype(np.float32)
    pca = PCA(4).fit(x)
    m = AnchoredAE(pca.mean_, pca.components_, width=8, freeze=True)
    assert not m.encoder.weight.requires_grad
    assert not m.decoder.weight.requires_grad
    reconstructed, z = m(torch.from_numpy(x))
    assert reconstructed.shape == (20, 15)
    assert z.shape == (20, 4)
    assert torch.isfinite(reconstructed).all()


def test_fake_npz_loads_and_split(tmp_path):
    source = create_synthetic_data(tmp_path / "data" / "synthetic.npz", n_per_source=12)
    d = load_npz(source)
    tr, te = d.split("S1")
    assert len(tr) == 48 and len(te) == 12
    assert set(d.ids[tr]).isdisjoint(set(d.ids[te]))
    assert len(d.plants) == 5


def test_reject_duplicate_physical_sample_ids(tmp_path):
    f = create_synthetic_data(tmp_path / "fake.npz")
    with np.load(f) as original:
        d = {k: original[k].copy() for k in original.files}
    d["ids"][0] = d["ids"][1]
    np.savez_compressed(f, **d)
    with pytest.raises(ValueError, match="unique"):
        load_npz(f)


def test_one_fold_small_training_smoke(tmp_path):
    f = create_synthetic_data(tmp_path / "fake.npz", n_per_source=12,
                              nir_channels=18, raman_channels=25)
    d = load_npz(f)
    torch.set_num_threads(1)
    config = FitConfig(components=3, ae_width=8, ae_epochs=5, ann_epochs=10)
    results, diagnostic = run_fresh_fold(d, "S1", 11, config)
    frame = pd.DataFrame(results)
    assert len(frame) == 2 * 12
    assert set(frame.model) == {"AE24", "PCA24"}
    assert frame.predicted.notna().all()
    table, pair = summarize(frame)
    assert pair is not None and len(pair) == 1
    assert len(table[table.source == "ALL"]) == 2


def test_pls_calibration_only_smoke(tmp_path):
    f = create_synthetic_data(tmp_path / "fake.npz", n_per_source=12,
                              nir_channels=18, raman_channels=25)
    d = load_npz(f)
    preds, lv = run_pls_baselines(d, max_components=2)
    assert len(preds) == 2 * len(d.y)
    assert len(lv) == 2 * 5
    assert set(preds.model) == {"NIR_PLS", "Raman_NIR_PLS"}
    assert preds.predicted.notna().all()


def test_preprocess_replication():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 4, 25))
    transformed = sg_derivative_snv(x, window_length=11)
    assert transformed.shape == x.shape
    assert np.allclose(transformed.mean(axis=-1), 0, atol=1e-5)
    averaged = preprocess_and_average_replicates([x[0], x[1]], 11)
    assert averaged.shape == (2, 25)
