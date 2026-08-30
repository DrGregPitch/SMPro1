#!/usr/bin/env python3
"""Model ladder + calibrated uncertainty on a single benchmark, honest split.

    python scripts/run_model_ladder.py --dataset freesolv --outdir results

Shows the repo does more than critique splits: it builds a proper predictor. Climbs
the ladder (mean -> gradient boosting -> deep ensemble), all under a scaffold split,
then recalibrates the ensemble's uncertainty on validation and draws the two figures
a property model needs -- calibration and selective prediction (error vs. the
fraction of most-confident predictions you act on). Ladder and calibration are
reused from polytools.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from polytools import (
    EnsembleRegressor,
    GBMRegressor,
    MeanRegressor,
    SigmaRecalibrator,
    regression_metrics,
    selective_prediction_curve,
    uncertainty_metrics,
)
from polytools.metrics import calibration_curve

from leakbench import featurize, load_dataset, scaffold_split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="freesolv")
    p.add_argument("--outdir", default="results", type=Path)
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset)
    X, _ = featurize(ds.mols, "ecfp+desc")
    sp = scaffold_split(ds.mols)
    # carve a validation slice out of train for recalibration (never touch test)
    rng = np.random.default_rng(0)
    tr = rng.permutation(sp.train)
    n_val = max(int(0.15 * len(tr)), 10)
    val, train = tr[:n_val], tr[n_val:]
    te = sp.test

    print(f"{ds}  |  scaffold split: {len(train)} train / {len(val)} val / {len(te)} test")
    print("\nModel ladder (test R2, honest scaffold split):")
    models = {
        "mean": MeanRegressor(),
        "gbm": GBMRegressor(n_estimators=400, backend="lightgbm"),
        "gbm_ensemble": EnsembleRegressor(
            factory=lambda i: GBMRegressor(n_estimators=400, random_state=i,
                                           quantile_uncertainty=False, backend="lightgbm"),
            n_members=5, bootstrap=True, include_member_sigma=False),
    }
    for name, m in models.items():
        m.fit(X[train], ds.y[train])
        r = regression_metrics(ds.y[te], m.predict(X[te]))
        print(f"  {name:13s} R2={r['r2']:5.2f}  RMSE={r['rmse']:5.2f} {ds.units}")

    # calibrated uncertainty from the ensemble
    ens = models["gbm_ensemble"]
    mu_v, sd_v = ens.predict(X[val], return_std=True)
    recal = SigmaRecalibrator().fit(ds.y[val], mu_v, sd_v)
    mu_t, sd_raw = ens.predict(X[te], return_std=True)
    sd_cal = recal.transform(sd_raw)
    um_raw = uncertainty_metrics(ds.y[te], mu_t, sd_raw)
    um_cal = uncertainty_metrics(ds.y[te], mu_t, sd_cal)
    print(f"\nEnsemble uncertainty: miscalibration area {um_raw['miscalibration_area']:.3f} "
          f"-> {um_cal['miscalibration_area']:.3f} after recalibration (0 is perfect)")

    if args.no_figure:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for sd, lab in [(sd_raw, "raw"), (sd_cal, "recalibrated")]:
        exp, obs = calibration_curve(ds.y[te], mu_t, sd)
        a1.plot(exp, obs, "o-", ms=3, label=lab)
    a1.plot([0, 1], [0, 1], "k--", lw=1, alpha=.6, label="ideal")
    a1.set_xlabel("expected coverage")
    a1.set_ylabel("observed coverage")
    a1.set_title("Calibration")
    a1.legend(fontsize=8)
    a1.set_aspect("equal", "box")

    curve = selective_prediction_curve(ds.y[te], mu_t, sd_cal)
    a2.plot(curve["coverage"] * 100, curve["rmse"], "o-", ms=3, color="#2b6cb0")
    a2.set_xlabel("coverage (%)")
    a2.set_ylabel(f"RMSE ({ds.units})")
    a2.set_title("Selective prediction\n(act on the most confident fraction)")
    a2.grid(alpha=.25)
    fig.suptitle(f"{args.dataset}: calibrated uncertainty on an honest scaffold split", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.outdir / f"uncertainty_{args.dataset}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {args.outdir / f'uncertainty_{args.dataset}.png'}")


if __name__ == "__main__":
    main()
