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

from smpro1 import (
    cluster_split,
    featurize,
    load_dataset,
    random_split,
    scaffold_split,
    split_difficulty,
)

DATASETS = ["lipophilicity", "esol", "freesolv"]
N_SEEDS = 5          # random splits are re-drawn; scaffold and cluster are deterministic


def evaluate(ds, split):
    """Fit and score one split. Descriptor imputation is fitted on TRAIN rows only."""
    X, _ = featurize(ds.mols, "ecfp+desc", impute_idx=split.train)
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
        print(f"\n{ds}  ({ds.note})")
        # a random split is one draw from a distribution, so quoting a single seed
        # quotes sampling noise as a result; scaffold and cluster are deterministic
        # and need no repeats
        runs = [("random", random_split(len(ds), seed=s)) for s in range(N_SEEDS)]
        runs += [("scaffold", scaffold_split(ds.mols)), ("cluster", cluster_split(ds.mols))]
        for sname, sp in runs:
            r2, rmse, nn, dup = evaluate(ds, sp)
            rows.append({"dataset": name, "split": sname, "r2": r2, "rmse": rmse,
                         "nn_similarity": nn, "frac_near_dup": dup})
        for sname in ("random", "scaffold", "cluster"):
            g = [r for r in rows if r["dataset"] == name and r["split"] == sname]
            r2 = np.array([r["r2"] for r in g])
            sd = f" +/-{r2.std():.2f}" if len(g) > 1 else ""
            print(f"  {sname:9s} R2={r2.mean():5.2f}{sd}  "
                  f"RMSE={np.mean([r['rmse'] for r in g]):5.2f}  "
                  f"NN-sim={np.mean([r['nn_similarity'] for r in g]):.2f}  "
                  f"near-dup>0.9={np.mean([r['frac_near_dup'] for r in g]):.0%}"
                  f"{f'  (n={len(g)} seeds)' if len(g) > 1 else ''}")

    res = pd.DataFrame(rows)
    res.to_csv(args.outdir / "leakage_results.csv", index=False)

    piv = res.pivot_table(index="dataset", columns="split", values="r2",
                          aggfunc="mean")[["random", "scaffold", "cluster"]]
    rsd = res[res.split == "random"].groupby("dataset").r2.std()
    # the ratio is only meaningful when the random-split model actually works;
    # near-zero or negative random R2 would print absurd or sign-flipped
    # percentages, so report NaN ("n/a") there instead
    with np.errstate(all="ignore"):
        infl = (piv["random"] - piv["scaffold"]) / piv["random"] * 100
        # spread of the inflation across random draws, so the headline carries its own
        # uncertainty instead of resting on whichever seed happened to be run first
        infl_sd = (rsd / piv["random"] * 100) * (piv["scaffold"] / piv["random"])
    piv["random_r2_sd"] = rsd
    piv["random_inflation_%"] = infl.where(piv["random"] >= 0.2)
    piv["random_inflation_sd"] = infl_sd.where(piv["random"] >= 0.2)
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
        sel = [res[(res.dataset == d) & (res.split == s)].r2 for d in DATASETS]
        vals = [v.mean() for v in sel]          # mean over seeds; one value for deterministic splits
        errs = [v.std() if len(v) > 1 else 0.0 for v in sel]
        ax.bar(x + (k - 1) * w, vals, w, yerr=errs, capsize=3, label=s,
               color=colors[s], edgecolor="white")
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
