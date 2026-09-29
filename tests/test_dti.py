"""Tests for the drug-target binding module.

The load-bearing test is ``test_cold_both_shares_no_entity``: the whole point of
cold splits is that a held-out pair has neither its drug nor its target in
training, so that gets pinned.
"""

from __future__ import annotations

import numpy as np
import pytest

from smpro1.dti import (
    load_davis,
    pair_features,
    pair_split,
    pair_split_report,
    protein_features,
)


@pytest.fixture(scope="module")
def ds():
    return load_davis()


def test_davis_loads_and_reconciles(ds):
    assert len(ds) > 20000
    assert ds.report["n_drugs"] > 0 and ds.report["n_targets"] > 0
    assert ds.report["pairs_final"] == len(ds)
    assert np.isfinite(ds.y).all()
    # pKd is a sane range for a Kd panel (nM..sub-nM), censored floor at 5
    assert 4.9 < ds.y.min() < 5.1 and ds.y.max() < 12


def test_protein_features_are_normalized_composition(ds):
    pf, names = protein_features(ds.seqs)
    v = next(iter(pf.values()))
    assert len(v) == len(names) == 20 + 400 + 1
    assert 0 <= v[:20].sum() <= 1.001          # AAC is a normalized fraction
    assert np.isfinite(v).all()


def test_pair_feature_blocks_line_up(ds):
    feats = pair_features(ds, n_bits=256)
    n = len(ds)
    assert feats["ligand"].shape[0] == feats["protein"].shape[0] == n
    assert feats["full"].shape[1] == feats["ligand"].shape[1] + feats["protein"].shape[1]


def test_cold_both_shares_no_entity(ds):
    """A cold-both test pair must have neither drug nor target seen in training."""
    sp = pair_split(ds, "cold_both", seed=0)
    assert len(sp.test) > 0 and len(sp.train) > 0
    rep = pair_split_report(ds, sp)
    assert rep["frac_test_drug_seen"] == 0.0
    assert rep["frac_test_target_seen"] == 0.0


def test_random_pair_leaks_both_sides(ds):
    """The contrast: a random pair split has most test entities seen in training."""
    rep = pair_split_report(ds, pair_split(ds, "random_pair", seed=0))
    assert rep["frac_test_drug_seen"] > 0.9      # 68 drugs, dense -> all seen
    assert rep["frac_test_target_seen"] > 0.5


def test_cold_target_holds_targets_out(ds):
    sp = pair_split(ds, "cold_target", seed=0)
    train_t = set(ds.targets[sp.train])
    test_t = set(ds.targets[sp.test])
    assert not (train_t & test_t)
