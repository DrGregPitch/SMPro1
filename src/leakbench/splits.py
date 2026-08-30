"""Splitters, and a diagnostic for how much a split actually leaks.

A random split of a molecular dataset is close to meaningless, and the 2026
data-leakage literature (HonestAffinity; PDBbind CleanSplit; "Systematic Data
Leakage in Protein-Ligand Benchmarks") is largely about this: chemical datasets are
dense with near-duplicate analogues, so a random test set almost always has a close
cousin in training. Models score well and then fail on the next scaffold a chemist
actually wants to make.

The honest reporting standard is **random alongside at least one structured split**;
the gap between them is the single most informative number in a molecular-property
study. :func:`split_difficulty` turns "we used a scaffold split" from an assertion
into a measurement -- the nearest-neighbour Tanimoto similarity from each test
molecule back to the training set.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

__all__ = ["Split", "random_split", "scaffold_split", "cluster_split",
           "split_difficulty", "SPLITTERS"]

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class Split(dict):
    """A train/test split: name -> integer index array."""

    @property
    def train(self) -> np.ndarray:
        return self["train"]

    @property
    def test(self) -> np.ndarray:
        return self["test"]

    def sizes(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.items()}


def random_split(n: int, frac_train: float = 0.8, seed: int = 0) -> Split:
    """Uniform random split -- the optimistic reference point, never the only number."""
    order = np.random.default_rng(seed).permutation(n)
    k = int(round(frac_train * n))
    return Split(train=order[:k], test=order[k:])


def _fps(mols):
    return [_GEN.GetFingerprint(m) for m in mols]


def scaffold_split(mols: Sequence, frac_train: float = 0.8) -> Split:
    """Bemis-Murcko scaffold split -- the field-standard structured split.

    Molecules are grouped by their Murcko scaffold; whole scaffolds go to train or
    test, never both, so the test set is genuinely new chemistry. Largest scaffolds
    are assigned to train first (the deterministic MoleculeNet convention).
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(mols):
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else ""
        groups[scaf].append(i)
    train: list[int] = []
    test: list[int] = []
    cutoff = frac_train * len(mols)
    for g in sorted(groups.values(), key=len, reverse=True):
        (train if len(train) < cutoff else test).extend(g)
    return Split(train=np.asarray(train, int), test=np.asarray(test, int))


def cluster_split(mols: Sequence, frac_train: float = 0.8, cutoff: float = 0.6) -> Split:
    """Butina-cluster the molecules by fingerprint distance, then split by cluster.

    Usually the hardest honest split: it separates near-neighbours a scaffold split
    can miss (two different scaffolds can still be very similar). ``cutoff`` is a
    Tanimoto *distance* threshold; report what you used.
    """
    fps = _fps(mols)
    n = len(fps)
    dists: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    train: list[int] = []
    test: list[int] = []
    ct = frac_train * n
    for c in sorted(clusters, key=len, reverse=True):
        (train if len(train) < ct else test).extend(c)
    return Split(train=np.asarray(train, int), test=np.asarray(test, int))


def split_difficulty(mols: Sequence, split: Split) -> dict[str, float]:
    """Quantify how far the test set really sits from training.

    Returns the distribution of nearest-neighbour Tanimoto similarity from each test
    molecule to its closest training molecule, plus the fraction of the test set
    with a very close training analogue (>0.9) -- the number that exposes a leaky
    split. Put this in your README; it is what makes an honest split a *measurement*.
    """
    fps = _fps(mols)
    train_fps = [fps[i] for i in split.train]
    nn = []
    for i in split.test:
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)
        nn.append(max(sims) if sims else 0.0)
    nn = np.asarray(nn)
    return {
        "nn_similarity_median": float(np.median(nn)),
        "nn_similarity_mean": float(nn.mean()),
        "frac_near_duplicate_gt_0.9": float((nn > 0.9).mean()),
        "frac_novel_lt_0.4": float((nn < 0.4).mean()),
    }


#: Registry so scripts can select a splitter by name.
SPLITTERS = {"random": random_split, "scaffold": scaffold_split, "cluster": cluster_split}
