#!/usr/bin/env python3
"""The headline result: how much do random splits inflate drug-property models?

    python scripts/run_leakage_benchmark.py --outdir results

For each benchmark dataset, trains the same model on a random split and on
structured (scaffold, cluster) splits, and measures how far each test set really
sits from training. The gap between random and structured is the leakage the 2026
literature is about -- reported here as a systematic effect across datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from polytools import GBMRegressor, regression_metrics

from leakbench import (
    cluster_split,
    featurize,
    load_dataset,
    random_split,
    scaffold_split,
    split_difficulty,
)

DATASETS = ["lipophilicity", "esol", "freesolv"]


def evaluate(ds, X, split):
    model = GBMRegressor(n_estimators=400, backend="lightgbm").fit(
        X[split.train], ds.y[split.train])
    m = regression_metrics(ds.y[split.test], model.predict(X[split.test]))
    d = split_difficulty(ds.mols, split)
    return m["r2"], m["rmse"], d["nn_similarity_median"], d["frac_near_duplicate_gt_0.9"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results", type=Path)
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in DATASETS:
        ds = load_dataset(name)
        X, _ = featurize(ds.mols, "ecfp+desc")
        print(f"\n{ds}  ({ds.note})")
        splits = {
            "random": random_split(len(ds), seed=0),
            "scaffold": scaffold_split(ds.mols),
            "cluster": cluster_split(ds.mols),
        }
        for sname, sp in splits.items():
            r2, rmse, nn, dup = evaluate(ds, X, sp)
            rows.append({"dataset": name, "split": sname, "r2": r2, "rmse": rmse,
                         "nn_similarity": nn, "frac_near_dup": dup})
            print(f"  {sname:9s} R2={r2:5.2f}  RMSE={rmse:5.2f}  "
                  f"NN-sim={nn:.2f}  near-dup>{0.9}={dup:.0%}")

    res = pd.DataFrame(rows)
    res.to_csv(args.outdir / "leakage_results.csv", index=False)

    piv = res.pivot(index="dataset", columns="split", values="r2")[["random", "scaffold", "cluster"]]
    # the ratio is only meaningful when the random-split model actually works;
    # near-zero or negative random R2 would print absurd or sign-flipped
    # percentages, so report NaN ("n/a") there instead
    with np.errstate(all="ignore"):
        infl = (piv["random"] - piv["scaffold"]) / piv["random"] * 100
    piv["random_inflation_%"] = infl.where(piv["random"] >= 0.2)
    print("\nR2 by dataset and split (random inflation = how much the random split lies):")
    print(piv.round(2).to_string())
    piv.round(3).to_csv(args.outdir / "leakage_table.csv")

    if args.no_figure:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    order = ["random", "scaffold", "cluster"]
    x = np.arange(len(DATASETS))
    w = 0.26
    colors = {"random": "#c0392b", "scaffold": "#2b6cb0", "cluster": "#2b8a3e"}
    for k, s in enumerate(order):
        vals = [res[(res.dataset == d) & (res.split == s)].r2.iloc[0] for d in DATASETS]
        ax.bar(x + (k - 1) * w, vals, w, label=s, color=colors[s], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["Lipophilicity\n(logD)", "ESOL\n(solubility)", "FreeSolv\n(hydration ΔG)"])
    ax.set_ylabel("test R²")
    ax.set_title("Random splits inflate drug-property models\n"
                 "The honest number is the structured-split bar, not the red one")
    ax.legend(title="split", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(args.outdir / "leakage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure -> {args.outdir / 'leakage.png'}")


if __name__ == "__main__":
    main()
