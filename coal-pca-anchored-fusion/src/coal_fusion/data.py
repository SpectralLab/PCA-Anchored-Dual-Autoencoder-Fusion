"""Private-data loader; the repository itself does not distribute data."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class SpectralData:
    xn: np.ndarray
    xr: np.ndarray
    y: np.ndarray
    sources: np.ndarray
    ids: np.ndarray

    def validate(self):
        n = len(self.y)
        if self.xn.ndim != 2 or self.xr.ndim != 2:
            raise ValueError("xn and xr must be 2D matrices")
        if not all(len(a) == n for a in (self.xn, self.xr, self.sources, self.ids)):
            raise ValueError("xn, xr, y, sources, ids must have the same sample count")
        if n == 0:
            raise ValueError("Dataset is empty")
        if self.y.ndim != 1 or self.sources.ndim != 1 or self.ids.ndim != 1:
            raise ValueError("y, sources, ids must be 1D")
        if any(not np.isfinite(a).all() for a in (self.xn, self.xr, self.y)):
            raise ValueError("Non-finite values found; clean the private dataset first")
        if len(set(map(str, self.ids))) != n:
            raise ValueError("Physical sample IDs must be unique (no replicate leakage)")
        if len(set(map(str, self.sources))) < 3:
            raise ValueError("At least three sources are needed for source-stratified LOSO")
        if np.std(self.y) < 1e-10:
            raise ValueError("Target y has zero variance")
        return self

    @property
    def plants(self):
        return sorted(set(map(str, self.sources)))

    def split(self, held):
        tr = np.flatnonzero(self.sources.astype(str) != str(held))
        te = np.flatnonzero(self.sources.astype(str) == str(held))
        if not len(tr) or not len(te):
            raise ValueError(f"Invalid held-out source: {held}")
        assert not set(map(str, self.ids[tr])).intersection(map(str, self.ids[te]))
        return tr, te


def load_npz(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Private input dataset not found: {path}")
    with np.load(path, allow_pickle=False) as f:
        required = {"xn", "xr", "y", "sources", "ids"}
        missing = required - set(f.files)
        if missing:
            raise ValueError(f"Missing .npz arrays: {sorted(missing)}")
        return SpectralData(
            xn=np.asarray(f["xn"], dtype=np.float32),
            xr=np.asarray(f["xr"], dtype=np.float32),
            y=np.asarray(f["y"], dtype=np.float64).reshape(-1),
            sources=np.asarray(f["sources"]).astype(str).reshape(-1),
            ids=np.asarray(f["ids"]).astype(str).reshape(-1),
        ).validate()
