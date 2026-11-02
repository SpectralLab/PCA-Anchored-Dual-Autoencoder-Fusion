"""Command-line interface: no bundled private data and no hard-coded paths."""
import os
# Limit BLAS workers before numpy/torch imports: reproduces single-thread protocol.
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from . import __version__
from .data import load_npz
from .training import FitConfig
from .evaluation import run_experiment, run_pls_baselines, summarize
from .demo import create_synthetic_data


def comma_ints(value):
    try:
        return [int(x) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc


def build_parser():
    p = argparse.ArgumentParser(prog="coal-fusion",
                                description="PCA-anchored dual AE fusion, with private input data")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", type=Path, required=True,
                        help="Path to private preprocessed .npz (not shipped with this repository)")
    common.add_argument("--out", type=Path, required=True,
                        help="Local output directory; do not commit result CSVs")
    common.add_argument("--components", type=int, default=12)

    fresh = sub.add_parser("fresh", parents=[common],
                           help="Independently retrain reconstructed AE + matched PCA/ANN")
    fresh.add_argument("--seeds", type=comma_ints, default=[11,23,47,71,101,131,173,211,257,307])
    fresh.add_argument("--holds", default="all", help="Comma-separated held-out sources or 'all'")
    fresh.add_argument("--ae-epochs", type=int, default=100)
    fresh.add_argument("--ann-epochs", type=int, default=300)
    fresh.add_argument("--ae-width", type=int, default=64)

    historical = sub.add_parser("archived", parents=[common],
                                help="Re-fit ANN with trusted PRIVATE original AE checkpoints")
    historical.add_argument("--checkpoints", type=Path, required=True,
                            help="LOCAL outer/<source>/(NIR|Raman)AE_seed*.pt directory")
    historical.add_argument("--seeds", type=comma_ints, default=[7,123,3407])
    historical.add_argument("--holds", default="all")
    historical.add_argument("--ann-epochs", type=int, default=300)

    pls = sub.add_parser("pls", parents=[common],
                         help="NIR-PLS and Raman-NIR PLS with calibration-source-only inner CV")
    pls.add_argument("--max-lv", type=int, default=20)

    demo = sub.add_parser("demo-data", help="Create fake data locally; not real samples")
    demo.add_argument("--out", type=Path, default=Path("data/synthetic_demo.npz"))
    demo.add_argument("--samples-per-source", type=int, default=22)

    summary = sub.add_parser("summarize", help="Recalculate metrics from local predictions")
    summary.add_argument("--predictions", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "demo-data":
        filename = create_synthetic_data(args.out, n_per_source=args.samples_per_source)
        print("Synthetic data created locally:", filename)
        return
    if args.command == "summarize":
        df = pd.read_csv(args.predictions, dtype={"sample_id": str})
        summary, pair = summarize(df)
        args.out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out / "metrics.csv", index=False)
        if pair is not None:
            pair.to_csv(args.out / "paired_pca_ae.csv", index=False)
        print(summary[summary.source == "ALL"].to_string(index=False))
        return

    # No user data is ever uploaded by this script.
    data = load_npz(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    with threadpool_limits(limits=1):
        if args.command == "pls":
            preds, choices = run_pls_baselines(data, max_components=args.max_lv)
            summary, _ = summarize(preds)
            preds.to_csv(args.out / "pls_predictions_PRIVATE.csv", index=False)
            choices.to_csv(args.out / "pls_component_selection.csv", index=False)
            summary.to_csv(args.out / "pls_metrics.csv", index=False)
            print(summary[summary.source == "ALL"].to_string(index=False))
        else:
            holds = data.plants if args.holds == "all" else [s.strip() for s in args.holds.split(",")]
            if len(holds) != len(set(holds)) or not set(holds).issubset(set(data.plants)):
                raise ValueError("Requested held-out sources must exist and be unique")
            if len(args.seeds) != len(set(args.seeds)):
                raise ValueError("Seeds must be unique")
            config = FitConfig(components=args.components,
                               ae_width=getattr(args, "ae_width", 64),
                               ae_epochs=getattr(args, "ae_epochs", 100),
                               ann_epochs=args.ann_epochs)
            mode = "fresh" if args.command == "fresh" else "archived"
            _, summary, paired = run_experiment(data, mode, args.seeds, holds,
                                                 config, args.out,
                                                 checkpoints=getattr(args, "checkpoints", None))
            print(summary[summary.source == "ALL"].to_string(index=False))
            if paired is not None:
                print("\nPaired RMSE by seed:\n", paired.to_string(index=False))
            record = {
                "mode": mode, "version": __version__, "config": config.__dict__,
                "seeds": args.seeds, "held_sources": holds,
                "note": "AE pretraining is reconstructed, not the lost original source"
                        if mode == "fresh" else "Only historical checkpoint inference + downstream ANN re-fit",
            }
            (args.out / "run_config.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
