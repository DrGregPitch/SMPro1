"""Featurizers for drug-like molecules: ECFP fingerprints and med-chem descriptors.

Deliberately classical. The point of this repository is honest *evaluation*, and a
well-understood ECFP + descriptor representation is the right substrate for that: it
is a strong baseline, it is interpretable, and it does not let a fancy model hide a
leaky split behind an impressive-looking number. Swapping in learned representations
(a GNN, a ChemBERTa embedding) is a drop-in change to this one module.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

__all__ = ["morgan", "descriptors", "featurize", "DESCRIPTOR_NAMES"]

# Medicinal-chemistry descriptors: the properties a chemist reads off a structure,
# and the ones that govern the physical targets here (solvation, permeability).
_DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    ("RingCount", Descriptors.RingCount),
    ("MolMR", Descriptors.MolMR),
    ("LabuteASA", Descriptors.LabuteASA),
    ("BertzCT", Descriptors.BertzCT),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
]
DESCRIPTOR_NAMES = [n for n, _ in _DESCRIPTORS]

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def morgan(mols: Sequence[Chem.Mol], n_bits: int = 2048, counts: bool = True) -> np.ndarray:
    """ECFP4 (Morgan radius 2) fingerprints; counts usually beat bits for regression."""
    gen = _GEN if n_bits == 2048 else rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=n_bits)
    out = np.zeros((len(mols), n_bits), dtype=np.float64)
    for i, m in enumerate(mols):
        fp = gen.GetCountFingerprintAsNumPy(m) if counts else gen.GetFingerprintAsNumPy(m)
        out[i] = fp
    return out


def descriptors(mols: Sequence[Chem.Mol], impute_idx: Sequence[int] | None = None) -> np.ndarray:
    """Curated med-chem descriptor block, NaN-imputed by column median.

    ``impute_idx`` selects the rows the imputation median is computed from -- pass
    the TRAIN indices. An imputation statistic is a fitted parameter, so taking it
    over the whole dataset lets test molecules influence the values the model is
    trained on. That is textbook leakage even though it is tiny, and in a repository
    whose subject is leakage it is the first thing a reader will check.

    Defaults to every row, which reproduces the earlier behaviour for callers that
    have no split to hand. In practice none of the bundled datasets produce a single
    NaN, so the two paths agree numerically -- the point is that the code no longer
    depends on that being true.
    """
    out = np.zeros((len(mols), len(_DESCRIPTORS)), dtype=np.float64)
    for i, m in enumerate(mols):
        for j, (_, fn) in enumerate(_DESCRIPTORS):
            try:
                v = float(fn(m))
            except Exception:
                v = np.nan
            out[i, j] = v if np.isfinite(v) else np.nan
    ref = out if impute_idx is None else out[np.asarray(impute_idx, int)]
    with np.errstate(all="ignore"):
        col_med = np.nanmedian(ref, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    bad = np.where(~np.isfinite(out))
    out[bad] = np.take(col_med, bad[1])
    return out


def featurize(mols: Sequence[Chem.Mol], kind: str = "ecfp+desc", n_bits: int = 2048,
              impute_idx: Sequence[int] | None = None) -> tuple[np.ndarray, list[str]]:
    """Return ``(X, feature_names)`` for ``"ecfp"``, ``"desc"``, or ``"ecfp+desc"``.

    ``impute_idx`` is forwarded to :func:`descriptors`; pass the train indices so the
    descriptor imputation is fitted on training rows only. Fingerprints are computed
    per molecule and need no such care.
    """
    if kind == "ecfp":
        return morgan(mols, n_bits), [f"ecfp_{i}" for i in range(n_bits)]
    if kind == "desc":
        return descriptors(mols, impute_idx), list(DESCRIPTOR_NAMES)
    if kind == "ecfp+desc":
        X = np.hstack([morgan(mols, n_bits), descriptors(mols, impute_idx)])
        names = [f"ecfp_{i}" for i in range(n_bits)] + list(DESCRIPTOR_NAMES)
        return X, names
    raise ValueError(f"unknown featurizer kind {kind!r}")
