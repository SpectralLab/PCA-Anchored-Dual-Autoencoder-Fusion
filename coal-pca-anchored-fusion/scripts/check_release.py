#!/usr/bin/env python3
"""Conservative privacy audit for the PUBLIC source directory only."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}
BLOCKED_EXT = {".npz", ".npy", ".csv", ".tsv", ".pt", ".pth", ".ckpt",
               ".mat", ".h5", ".hdf5", ".parquet", ".pkl", ".joblib",
               ".xlsx", ".xls", ".zip", ".jsonl", ".log", ".pickle"}
BLOCKED_DIRS = {"data", "private", "private_data", "datasets", "raw",
                "runs", "results", "checkpoints", "models", "outputs", "artifacts"}


def scan(root):
    issues = []
    for file in root.rglob("*"):
        if not file.is_file() or any(p in IGNORED_PARTS for p in file.relative_to(root).parts):
            continue
        relative = file.relative_to(root)
        if any(p in BLOCKED_DIRS for p in relative.parts[:-1]):
            issues.append((str(relative), "blocked private/output directory"))
        elif file.suffix.lower() in BLOCKED_EXT:
            issues.append((str(relative), "potential data/model artifact"))
        elif file.stat().st_size > 1_000_000:
            issues.append((str(relative), "unexpected large file (>1 MB)"))
    return issues


if __name__ == "__main__":
    issues = scan(ROOT)
    for filename, why in issues:
        print("BLOCKED:", filename, why)
    if issues:
        print(f"FAIL: {len(issues)} files need manual review.", file=sys.stderr)
        sys.exit(1)
    print("PASS: no disallowed data/model artifacts detected in release tree.")
    print("Note: also inspect `git status` and Git history before publishing.")
