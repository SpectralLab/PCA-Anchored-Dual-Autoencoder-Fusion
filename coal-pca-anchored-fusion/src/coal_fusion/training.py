"""Preserved core training logic for the downstream ANN and reconstructed AE.

No held-out source labels enter PCA/AE fitting or hyperparameter selection.
This module does not implement the lost original AE pretraining source.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .model import AnchoredAE, GlobalANN, encode


@dataclass(frozen=True)
class FitConfig:
    components: int = 12
    ae_width: int = 64
    ae_epochs: int = 100
    ann_epochs: int = 300
    batch_size: int = 64
    validate_every: int = 5
    split_seed: int = 817
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    init_sd: float = 0.001  # Specific to reconstructed AE pretraining
    early_patience: int = 50  # ANN only, after 80 epochs


def pca_fit(train_matrix, config):
    max_components = min(train_matrix.shape)
    if config.components > max_components or config.components < 1:
        raise ValueError(f"components must be in [1,{max_components}]")
    return PCA(n_components=config.components, svd_solver="full").fit(train_matrix)


def stratified_indices(source, config):
    source = np.asarray(source).astype(str)
    if min(np.unique(source, return_counts=True)[1]) < 2:
        raise ValueError("Each calibration source needs at least two samples for 80:20 stratification")
    return train_test_split(
        np.arange(len(source)), test_size=0.20,
        random_state=config.split_seed, stratify=source,
    )


def fit_reconstructed_ae(x, source, seed, pca, config):
    """New frozen independent-repeat protocol; not a verbatim old pretrainer.

    Uses small residual output-layer initialization and a fixed scalar loss
    normalization, as in the 2026-11-02 reconstructed 10-seed experiment.
    """
    x = np.asarray(x, dtype=np.float32)
    xt = torch.as_tensor(x)
    ti, vi = stratified_indices(source, config)
    pca_reconstruction = pca.inverse_transform(pca.transform(x))
    normalizer = float(max(1e-8, np.mean((x - pca_reconstruction) ** 2)))

    def fresh_model():
        torch.manual_seed(int(seed))
        net = AnchoredAE(pca.mean_, pca.components_, config.ae_width, freeze=True)
        with torch.no_grad():
            for layer in (net.enc_residual[-1], net.dec_residual[-1]):
                nn.init.normal_(layer.weight, mean=0.0, std=config.init_sd)
                layer.bias.zero_()
        return net

    def fit(ids, epochs, validation=None):
        net = fresh_model()
        rng = np.random.RandomState(int(seed))
        params = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        best, best_epoch = float("inf"), config.validate_every
        for epoch in range(1, epochs + 1):
            net.train()
            batches = np.array_split(rng.permutation(ids),
                max(1, int(np.ceil(len(ids) / config.batch_size))))
            for indices in batches:
                optimizer.zero_grad(set_to_none=True)
                reconstructed, _ = net(xt[indices])
                loss = (reconstructed - xt[indices]).square().mean() / normalizer
                loss.backward()
                nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()
            if validation is not None and epoch % config.validate_every == 0:
                net.eval()
                with torch.no_grad():
                    recon, _ = net(xt[validation])
                    score = float((recon - xt[validation]).square().mean())
                if score < best - 1e-8:
                    best, best_epoch = score, epoch
        return net, best_epoch, best

    _, selected_epoch, valid_mse = fit(ti, config.ae_epochs, vi)
    model, _, _ = fit(np.arange(len(x)), selected_epoch)
    model.eval()
    return model, {"checkpoint_epoch": int(selected_epoch),
                   "validation_mse": float(valid_mse),
                   "pca_reconstruction_normalizer": normalizer}


def fit_ann(x, y, source, seed, config):
    """Archived 32-32-16 ANN training protocol, refactored without path globals."""
    y = np.asarray(y, dtype=np.float64)
    ymean, ystd = float(y.mean()), float(y.std())
    if ystd <= 1e-10:
        raise ValueError("Target y has zero variance in this calibration fold")
    targets = torch.tensor((y - ymean) / ystd, dtype=torch.float32)
    xt = torch.tensor(np.asarray(x, dtype=np.float32))
    ti, vi = stratified_indices(source, config)

    def fit(indices, max_epochs, validation=None):
        torch.manual_seed(int(seed))
        rng = np.random.RandomState(int(seed))
        net = GlobalANN(xt.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=config.learning_rate,
                                weight_decay=config.weight_decay)
        best, best_epoch, last_improvement = float("inf"), config.validate_every, 0
        for epoch in range(1, max_epochs + 1):
            net.train()
            batches = np.array_split(rng.permutation(indices),
                max(1, int(np.ceil(len(indices) / config.batch_size))))
            for batch in batches:
                opt.zero_grad(set_to_none=True)
                loss = (net(xt[batch]) - targets[batch]).square().mean()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            if validation is not None and epoch % config.validate_every == 0:
                net.eval()
                with torch.no_grad():
                    score = float((net(xt[validation]) - targets[validation]).square().mean())
                if score < best - 1e-6:
                    best, best_epoch, last_improvement = score, epoch, epoch
                if epoch >= 80 and epoch - last_improvement >= config.early_patience:
                    break
        return net, best_epoch

    _, chosen = fit(ti, config.ann_epochs, vi)
    model, _ = fit(np.arange(len(y)), chosen)
    return model, ymean, ystd, chosen


def regress(features_train, features_test, y, source, seed, config):
    scaler = StandardScaler().fit(features_train)
    train = scaler.transform(features_train).astype(np.float32)
    test = scaler.transform(features_test).astype(np.float32)
    model, ymean, ystd, epochs = fit_ann(train, y, source, seed, config)
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(test, dtype=torch.float32)).cpu().numpy() * ystd + ymean
    return np.asarray(pred, dtype=np.float64), {"ann_epochs": int(epochs)}


def run_fresh_fold(data, held, seed, config):
    """Matched PCA24 vs newly trained AE24 on one leave-source-out fold."""
    tr, te = data.split(held)
    pca_features, ae_features, diagnostics = {}, {}, {}
    for mod, key in (("NIR", "xn"), ("Raman", "xr")):
        spectra = getattr(data, key)
        pca = pca_fit(spectra[tr], config)
        model, meta = fit_reconstructed_ae(spectra[tr], data.sources[tr], seed, pca, config)
        pca_features[mod] = (pca.transform(spectra[tr]), pca.transform(spectra[te]))
        ae_features[mod] = (encode(model, spectra[tr]), encode(model, spectra[te]))
        diagnostics[f"{mod}_ae"] = meta

    records = []
    for name, representation in (("PCA24", pca_features), ("AE24", ae_features)):
        train = np.column_stack([representation["NIR"][0], representation["Raman"][0]])
        test = np.column_stack([representation["NIR"][1], representation["Raman"][1]])
        preds, meta = regress(train, test, data.y[tr], data.sources[tr], seed, config)
        diagnostics[f"{name}_ann"] = meta
        for sid, actual, prediction in zip(data.ids[te], data.y[te], preds):
            records.append({"held": str(held), "seed": int(seed), "model": name,
                            "sample_id": str(sid), "actual": float(actual),
                            "predicted": float(prediction)})
    return records, diagnostics


def load_archived_encoder(checkpoints, held, mod, seed):
    """Read *trusted local* historic checkpoints. Never download/run untrusted .pt."""
    from pathlib import Path
    path = Path(checkpoints) / str(held) / f"{mod}AE_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing local AE checkpoint: {path}")
    # Historical PyTorch checkpoints contain numpy objects; weights_only=True
    # is not compatible. Only load checkpoints produced by your own project.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = AnchoredAE(ck["mean"], ck["components"], ck["width"], ck["freeze"])
    model.load_state_dict(ck["state"], strict=True)
    model.eval()
    return model


def run_archived_fold(data, held, seed, config, checkpoints):
    """Refit downstream ANN against existing *private* original encoders.

    Historical setup with four inputs (12 NIR or 24 Raman+NIR features).
    This does NOT independently rerun lost original AE pretraining.
    """
    tr, te = data.split(held)
    pcas, aes = {}, {}
    for mod, key in (("NIR", "xn"), ("Raman", "xr")):
        spectra = getattr(data, key)
        pca = pca_fit(spectra[tr], config)
        pcas[mod] = (pca.transform(spectra[tr]), pca.transform(spectra[te]))
        model = load_archived_encoder(checkpoints, held, mod, seed)
        aes[mod] = (encode(model, spectra[tr]), encode(model, spectra[te]))

    variants = {
        "NIR_PCA12": pcas["NIR"],
        "NIR_AE12": aes["NIR"],
        "PCA24": (np.column_stack((pcas["NIR"][0], pcas["Raman"][0])),
                   np.column_stack((pcas["NIR"][1], pcas["Raman"][1]))),
        "AE24": (np.column_stack((aes["NIR"][0], aes["Raman"][0])),
                  np.column_stack((aes["NIR"][1], aes["Raman"][1]))),
    }
    records, diagnostics = [], {}
    for name, (train, test) in variants.items():
        pred, meta = regress(train, test, data.y[tr], data.sources[tr], seed, config)
        diagnostics[name] = meta
        for sid, actual, prediction in zip(data.ids[te], data.y[te], pred):
            records.append({"held": str(held), "seed": int(seed), "model": name,
                            "sample_id": str(sid), "actual": float(actual),
                            "predicted": float(prediction)})
    return records, diagnostics
