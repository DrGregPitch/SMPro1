#!/usr/bin/env python3
"""Binding affinity, honestly: where does a DTI model's performance come from?

    python scripts/run_binding_benchmark.py --outdir results

Trains three models on DAVIS (kinase inhibitor Kd panel) under four pair splits:

* models: **protein-only**, **ligand-only** (memorization baselines by
  construction -- neither can see the interaction), and **full** (ligand+protein);
* splits: random_pair -> cold_drug -> cold_target -> cold_both.

Two numbers fall out. The *collapse* (full-model R2 from random_pair to
cold_both) is how much a random pair split flatters. The *interaction gap*
(full minus best single-sided baseline, per split) is the part of the
performance that is genuinely learned interaction rather than memorized
per-entity affinity -- the quantity the 2026 leakage literature (HonestAffinity,
CleanSplit) argues benchmarks should report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from polytools import GBMRegressor, regression_metrics

from leakbench.dti import load_davis, pair_features, pair_split, pair_split_report

MODELS = ["protein_only", "ligand_only", "full"]
SPLITS = ["random_pair", "cold_drug", "cold_target", "cold_both"]
BLOCK = {"protein_only": "protein", "ligand_only": "ligand", "full": "full"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results", type=Path)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ds = load_davis()
    print(f"{ds}  | censored at pKd=5: {ds.report['censored_frac_pkd5']:.0%} "
          f"(assay panel floor -- most pairs are non-binders)")
    feats = pair_features(ds)

    rows = []
    for sname in SPLITS:
        sp = pair_split(ds, sname, seed=args.seed)
        rep = pair_split_report(ds, sp)
        print(f"\n{sname}: {int(rep['n_train'])} train / {int(rep['n_test'])} test | "
              f"test drug seen in train: {rep['frac_test_drug_seen']:.0%}, "
              f"target seen: {rep['frac_test_target_seen']:.0%}")
        for mname in MODELS:
            X = feats[BLOCK[mname]]
            model = GBMRegressor(n_estimators=400, quantile_uncertainty=False,
                                 backend="lightgbm").fit(X[sp.train], ds.y[sp.train])
            m = regression_metrics(ds.y[sp.test], model.predict(X[sp.test]))
            rows.append({"split": sname, "model": mname, "r2": m["r2"],
                         "rmse": m["rmse"], **rep})
            print(f"  {mname:13s} R2={m['r2']:6.2f}  RMSE={m['rmse']:.2f} pKd")

    res = pd.DataFrame(rows)
    res.to_csv(args.outdir / "binding_results.csv", index=False)
    piv = res.pivot(index="model", columns="split", values="r2").loc[MODELS, SPLITS]
    print("\nR2 by model and split:")
    print(piv.round(2).to_string())
    piv.round(3).to_csv(args.outdir / "binding_table.csv")

    print("\nInteraction gap (full minus best single-sided baseline):")
    for sname in SPLITS:
        gap = piv.loc["full", sname] - piv.loc[["protein_only", "ligand_only"], sname].max()
        print(f"  {sname:12s} {gap:+.2f}")

    if args.no_figure:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(SPLITS))
    w = 0.26
    colors = {"protein_only": "#b8b8b8", "ligand_only": "#e69f00", "full": "#2b6cb0"}
    for k, mname in enumerate(MODELS):
        vals = [piv.loc[mname, s] for s in SPLITS]
        ax.bar(x + (k - 1) * w, vals, w, label=mname.replace("_", " "),
               color=colors[mname], edgecolor="white")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["random\npair", "cold\ndrug", "cold\ntarget", "cold\nboth"])
    ax.set_ylabel("test R² (pKd)")
    ax.set_title("Binding affinity on DAVIS: a ligand-only model rides the random split\n"
                 "cold splits reveal how little interaction was actually learned")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.outdir / "binding_leakage.png", dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {args.outdir / 'binding_leakage.png'}")


if __name__ == "__main__":
    main()
