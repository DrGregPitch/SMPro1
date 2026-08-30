"""Dataset loading for drug-discovery molecular property benchmarks.

These are the canonical MoleculeNet / Therapeutics Data Commons regression sets --
the standard benchmarks the field reports on, and the ones the 2026 data-leakage
literature re-examines. They are downloaded on demand (no data blobs committed) from
the public DeepChem mirror.

The point of this repository is *honest evaluation*: the loaders here are
deliberately thin, because the interesting work is in how you split and score (see
``splits.py``), not in the loading.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_MIRROR = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"

#: name -> (remote file, smiles column, target column, units, physical note)
DATASETS = {
    "lipophilicity": ("Lipophilicity.csv", "smiles", "exp", "logD",
                      "octanol/water partition -- a solvation free-energy difference"),
    "esol": ("delaney-processed.csv", "smiles",
             "measured log solubility in mols per litre", "logS",
             "aqueous solubility -- set by solute-solvent interaction energies"),
    "freesolv": ("SAMPL.csv", "smiles", "expt", "kcal/mol",
                 "hydration free energy -- a literal DeltaG, the most physical target here"),
}

__all__ = ["MoleculeDataset", "load_dataset", "DATASETS"]


@dataclass
class MoleculeDataset:
    """Validated molecules + a real-valued target.

    Attributes
    ----------
    smiles, y
        Aligned arrays of canonical SMILES and target values.
    mols
        Parsed RDKit molecules (aligned).
    name, target_name, units, note
        Provenance and the one-line physical interpretation of the target.
    report
        What was dropped during cleaning -- print it in a README.
    """

    smiles: np.ndarray
    y: np.ndarray
    mols: list
    name: str = ""
    target_name: str = "y"
    units: str = ""
    note: str = ""
    report: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"MoleculeDataset({self.name}, n={len(self)}, "
                f"target={self.target_name!r} [{self.units}])")


def _fetch(remote: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / remote
    if not dest.exists():
        urllib.request.urlretrieve(_MIRROR + remote, dest)  # noqa: S310 - trusted https
    return dest


def load_dataset(name: str, cache: str | Path = "data_cache") -> MoleculeDataset:
    """Download (once), parse, canonicalize, and de-duplicate a benchmark dataset."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    remote, scol, ycol, units, note = DATASETS[name]
    df = pd.read_csv(_fetch(remote, Path(cache)))
    n_raw = len(df)
    df = df.dropna(subset=[scol, ycol])

    canon, mols, ys, seen = [], [], [], {}
    n_bad = 0
    for smi, yv in zip(df[scol].astype(str), pd.to_numeric(df[ycol], errors="coerce")):
        if not np.isfinite(yv):
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            n_bad += 1
            continue
        cs = Chem.MolToSmiles(m)
        if cs in seen:                     # de-duplicate on canonical form
            continue
        seen[cs] = True
        canon.append(cs)
        mols.append(m)
        ys.append(float(yv))

    report = {"rows_in_file": n_raw, "dropped_unparseable": n_bad,
              "rows_final": len(canon), "duplicates_removed": len(df) - n_bad - len(canon)}
    return MoleculeDataset(
        smiles=np.array(canon, dtype=object), y=np.asarray(ys, dtype=float),
        mols=mols, name=name, target_name=ycol, units=units, note=note, report=report,
    )
