"""Tests for leakbench, organised by the invariant they protect.

The load-bearing test is ``test_structured_splits_are_harder_than_random`` -- the
whole thesis of the repository is that a structured split is measurably less leaky
than a random one, so it gets pinned.
"""

from __future__ import annotations

import numpy as np
import pytest

from leakbench import (
    cluster_split,
    featurize,
    load_dataset,
    random_split,
    scaffold_split,
    split_difficulty,
)

# small, fast dataset for the suite
DS = "freesolv"


@pytest.fixture(scope="module")
def ds():
    return load_dataset(DS)


def test_dataset_loads_and_is_clean(ds):
    assert len(ds) > 500
    assert len(ds.smiles) == len(ds.y) == len(ds.mols)
    assert np.isfinite(ds.y).all()
    assert ds.report["rows_final"] == len(ds)


def test_featurizers_have_matching_names(ds):
    for kind in ("ecfp", "desc", "ecfp+desc"):
        X, names = featurize(ds.mols[:20], kind, n_bits=256)
        assert X.shape[1] == len(names)
        assert np.isfinite(X).all()


def test_splits_are_disjoint_and_complete(ds):
    n = len(ds)
    for sp in (random_split(n), scaffold_split(ds.mols), cluster_split(ds.mols)):
        allidx = np.concatenate([sp.train, sp.test])
        assert len(allidx) == n
        assert len(np.unique(allidx)) == n


def test_scaffold_split_never_shares_a_scaffold(ds):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    sp = scaffold_split(ds.mols)
    scaf = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in ds.mols])
    assert not (set(scaf[sp.train]) & set(scaf[sp.test]))


def test_structured_splits_are_harder_than_random(ds):
    """The thesis: a scaffold split leaks measurably less than a random split."""
    rnd = split_difficulty(ds.mols, random_split(len(ds), seed=0))
    scaf = split_difficulty(ds.mols, scaffold_split(ds.mols))
    assert scaf["nn_similarity_median"] < rnd["nn_similarity_median"]
    assert scaf["frac_near_duplicate_gt_0.9"] <= rnd["frac_near_duplicate_gt_0.9"]


def test_split_difficulty_keys(ds):
    d = split_difficulty(ds.mols, random_split(len(ds)))
    for k in ("nn_similarity_median", "frac_near_duplicate_gt_0.9", "frac_novel_lt_0.4"):
        assert k in d and 0.0 <= d[k] <= 1.0


def test_structured_split_guards_single_dominant_group():
    """Regression: the greedy fill must never return an empty test set.

    A congeneric series (one dominant scaffold) used to put every molecule in
    train -- scaffold_split now moves the last group to test, and a dataset
    where ALL molecules share one scaffold raises instead of silently
    returning an unusable split.
    """
    from rdkit import Chem

    # two scaffolds, 5 molecules each: with frac_train=0.8 the old code put
    # both groups in train (5 < 8 both times) and returned an empty test
    benzenes = [Chem.MolFromSmiles(f"c1ccccc1{'C' * i}") for i in range(1, 6)]
    pyridines = [Chem.MolFromSmiles(f"c1ccncc1{'C' * i}") for i in range(1, 6)]
    sp = scaffold_split(benzenes + pyridines, frac_train=0.8)
    assert len(sp.test) > 0 and len(sp.train) > 0
    assert len(sp.train) + len(sp.test) == 10

    # single scaffold for every molecule -> loud failure, not an empty test
    with pytest.raises(ValueError):
        scaffold_split(benzenes, frac_train=0.8)
