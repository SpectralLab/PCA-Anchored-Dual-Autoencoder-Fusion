"""Paired LOSO evaluation, summaries, and calibration-only PLS baselines."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from .training import run_fresh_fold, run_archived_fold


def metrics(group):
    e = np.asarray(group["predicted"], dtype=float) - np.asarray(group["actual"], dtype=float)
    return {"n": int(len(e)), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
            "r2": float(r2_score(group["actual"], group["predicted"])) if len(e) > 1 else float("nan")}


def summarize(predictions):
    df = pd.DataFrame(predictions).copy()
    required = {"held", "seed", "model", "sample_id", "actual", "predicted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Empty predictions")
    if df[list(required)].isna().any().any():
        raise ValueError("Missing predictions or identifiers")
    if df.duplicated(["seed", "model", "sample_id"]).any():
        raise ValueError("A sample was predicted more than once for the same seed/model")
    detail = []
    for (seed, model), subset in df.groupby(["seed", "model"], sort=True):
        for source in ["ALL"] + sorted(subset.held.astype(str).unique()):
            group = subset if source == "ALL" else subset[subset.held == source]
            detail.append({"seed": int(seed), "model": model, "source": source, **metrics(group)})
    out = pd.DataFrame(detail)
    pair = out[out.source == "ALL"].pivot(index="seed", columns="model", values="rmse")
    paired = None
    if "AE24" in pair and "PCA24" in pair:
        paired = pair[["PCA24", "AE24"]].dropna().reset_index()
        paired["pca_minus_ae_pp"] = paired.PCA24 - paired.AE24
        paired["improvement_pct"] = 100.0 * paired.pca_minus_ae_pp / paired.PCA24
    return out, paired


def run_experiment(data, mode, seeds, held_sources, config, output, checkpoints=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    detail = output / "fold_predictions"
    detail.mkdir(exist_ok=True)
    for seed in seeds:
        for held in held_sources:
            path = detail / f"{mode}_{held}_seed{seed}.csv"
            if path.exists():
                print("REUSE", path.name, flush=True)
                continue
            if mode == "fresh":
                records, info = run_fresh_fold(data, held, seed, config)
            elif mode == "archived":
                records, info = run_archived_fold(data, held, seed, config, checkpoints)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            df = pd.DataFrame(records)
            assert df.groupby("model").size().eq(sum(data.sources == held)).all()
            tmp = path.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(path)
            print("DONE", held, seed, {k: round(metrics(g)["rmse"], 4)
                                       for k, g in df.groupby("model")}, flush=True)
    files = sorted(detail.glob(f"{mode}_*_seed*.csv"))
    # Only collect requested seeds and held-out groups for this invocation.
    selected = [f for f in files if any(f.name == f"{mode}_{h}_seed{s}.csv"
                                        for h in held_sources for s in seeds)]
    df = pd.concat([pd.read_csv(p, dtype={"sample_id": str}) for p in selected], ignore_index=True)
    out, paired = summarize(df)
    df.to_csv(output / "predictions_PRIVATE.csv", index=False)
    out.to_csv(output / "metrics.csv", index=False)
    if paired is not None:
        paired.to_csv(output / "paired_pca_ae.csv", index=False)
    return df, out, paired


def _nested_pls_components(x, y, source, max_components):
    """Use only training-source folds to choose PLS components.

    Single calibration source held out inside the outer calibration set.
    Each inner fold fits StandardScaler only on its own training subset.
    """
    source = np.asarray(source).astype(str)
    unique = sorted(set(source))
    if len(unique) < 2:
        raise ValueError("PLS needs at least two calibration sources")
    least_train = min(sum(source != one) for one in unique)
    cap = min(int(max_components), x.shape[1], least_train - 1)
    if cap < 1:
        raise ValueError("Insufficient samples for nested PLS")
    records = []
    for nlv in range(1, cap + 1):
        errors = []
        for inner_held in unique:
            tr, val = source != inner_held, source == inner_held
            scaling = StandardScaler().fit(x[tr])
            model = PLSRegression(n_components=nlv, scale=False)
            model.fit(scaling.transform(x[tr]), y[tr])
            pred = model.predict(scaling.transform(x[val])).reshape(-1)
            errors.extend(((pred - y[val]) ** 2).tolist())
        records.append((nlv, float(np.mean(errors))))
    return min(records, key=lambda z: (z[1], z[0]))[0], records


def run_pls_baselines(data, max_components=20):
    """NIR-only and combined spectral PLS under strict outer LOSO.

    Paired-modality PLS uses variable-wise training StandardScaler. Neither
    variant uses AE representations or outer test labels for tuning.
    """
    rows, chosen = [], []
    for held in data.plants:
        tr, te = data.split(held)
        for modelname, x in (("NIR_PLS", data.xn),
                             ("Raman_NIR_PLS", np.column_stack((data.xn, data.xr)))):
            y_train = data.y[tr]
            nlv, score_grid = _nested_pls_components(x[tr], y_train,
                                                       data.sources[tr], max_components)
            scaler = StandardScaler().fit(x[tr])
            pls = PLSRegression(n_components=nlv, scale=False)
            pls.fit(scaler.transform(x[tr]), y_train)
            prediction = pls.predict(scaler.transform(x[te])).reshape(-1)
            chosen.append({"held": held, "model": modelname, "lv": int(nlv),
                           "calibration_source_cv_mse": dict(score_grid)[nlv]})
            for sid, actual, predicted in zip(data.ids[te], data.y[te], prediction):
                rows.append({"held": held, "seed": 0, "model": modelname,
                             "sample_id": str(sid), "actual": float(actual),
                             "predicted": float(predicted)})
    return pd.DataFrame(rows), pd.DataFrame(chosen)
