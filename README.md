# SMPro1

*Small Molecule–Protein, project 1.*

**Honest evaluation for drug-discovery ML — small-molecule property prediction and small-molecule–protein binding affinity, with the leakage measured instead of assumed.**

![CI](https://github.com/DrGregPitch/SMPro1/actions/workflows/ci.yml/badge.svg)
&nbsp;·&nbsp; MIT &nbsp;·&nbsp; Python 3.10–3.12

A molecular ML model that looks excellent on a random train/test split can be nearly useless on the next chemical series a project actually wants to make. Chemical datasets are dense with near-duplicate analogues, so a random test set almost always has a close cousin in training — the model interpolates and the score is inflated. A line of 2026 work addresses this directly (*HonestAffinity*, PDBbind CleanSplit, *Systematic Data Leakage in Protein-Ligand Benchmarks*). `smpro1` measures it and reports the number that survives.

![Random splits inflate R² on all three benchmarks relative to the scaffold and cluster splits.](assets/leakage.png)

## The result

Same model and features on three standard benchmarks — **only the split changes:**

| dataset | random R² | scaffold R² | cluster R² | random inflated by |
|:---|---:|---:|---:|---:|
| Lipophilicity (logD) | 0.71 ± 0.03 | 0.60 | 0.52 | +16 ± 3% |
| ESOL (aqueous solubility) | 0.91 ± 0.01 | 0.80 | 0.77 | +11 ± 1% |
| **FreeSolv (hydration ΔG)** | **0.92 ± 0.02** | **0.63** | 0.85 | **+32 ± 1%** |

Random splits are re-drawn over 5 seeds and reported as mean ± sd; scaffold and cluster splits are deterministic, so they are single values. Inflation is quoted against the scaffold split.

FreeSolv moves the most: R² = 0.92 on a random split and 0.63 on new scaffolds, with RMSE rising from 1.09 to 3.13 kcal/mol.

`split_difficulty()` reports how far each test set sits from training. Ranges span the three datasets:

| split | median nearest-neighbour similarity to train | test with a >0.9 near-duplicate in train |
|:---|---:|---:|
| random | 0.54 – 0.70 | **7 – 10%** |
| scaffold | 0.22 – 0.52 | 0 – 3% |
| cluster | 0.32 – 0.33 | 0% |

These are ranges rather than averages because a scaffold split is not uniformly hard. On Lipophilicity it still leaves 3% of the test set with a near-twin in training at a median similarity of 0.52; on FreeSolv it reaches 0.22 with no near-duplicates at all.

Several percent of each random test set has a near-twin in training, which is where the inflation comes from.

## Which split is hardest depends on the dataset

One reading of the table is that the most physical target leaks most: hydration free
energy is a literal ΔG, so a new scaffold should break an interpolating model hardest.
Measuring the same inflation against the cluster split instead reverses the ordering:

| dataset | vs scaffold | vs cluster |
|:---|---:|---:|
| **FreeSolv** (hydration ΔG) | **+32%** | **+7%** |
| ESOL (solubility) | +11% | +15% |
| Lipophilicity (logD) | +16% | +27% |

FreeSolv goes from the most-inflated dataset to the least. A target that was
intrinsically hard to extrapolate across chemistry would be penalised by both
structured splits; FreeSolv is penalised by only one.

The reason is that half of FreeSolv has no scaffold at all. 320 of its 642 molecules are
acyclic, so Bemis–Murcko assigns them all the empty scaffold, they form one group,
and that group lands in train. The FreeSolv "scaffold split" is therefore largely an
acyclic → cyclic extrapolation, which is why it is the most severe split measured
here (nearest-neighbour similarity 0.22, no near-duplicates) while its cluster split
is unremarkable (0.33, in line with the others).

So "we used a scaffold split" does not by itself name a difficulty. The same phrase
covers a 0.52-similarity test set on Lipophilicity and a 0.22-similarity one on
FreeSolv, and the inflation reported follows the split chosen. More than one
structured split, with `split_difficulty()` alongside it, is what makes the number
interpretable.

## Run it

```bash
git clone https://github.com/DrGregPitch/SMPro1 && cd SMPro1
uv venv && uv pip install -e ".[dev]"        # pulls the model ladder from polytools
uv run python scripts/run_leakage_benchmark.py --outdir results   # ~1 min: table + figure
uv run pytest tests -v
```

Datasets are the canonical MoleculeNet / Therapeutics Data Commons regression sets, downloaded on demand (no data committed).

## What's inside

- **Honest splitters, implemented** (`splits.py`) — random, Bemis–Murcko scaffold, and Butina-cluster, plus `split_difficulty()`, which measures how out-of-distribution a split is rather than asserting it. The hardest split is not always the same one: scaffold for FreeSolv, cluster for Lipophilicity.
- **Model ladder + calibration, reused from [`polytools`](https://github.com/DrGregPitch/polytools)** — the same honest-evaluation infrastructure built for polymers works unchanged on drug-like molecules, because a feature matrix doesn't care what kind of molecule it came from. Gradient boosting on ECFP + medicinal-chemistry descriptors, with deep-ensemble uncertainty and a `SigmaRecalibrator`.
- **Med-chem featurization** (`featurize.py`) — ECFP4 fingerprints + the descriptor block a chemist reads off a structure (MolWt, LogP, TPSA, HBD/HBA, …). Deliberately classical, so model capacity cannot mask a leaky split.

---

## Binding affinity: where the performance comes from

Property prediction leaks through *similar molecules*. Drug–target binding affinity can leak through either side of the pair. On a random split of (drug, target) pairs, a model can score well by memorizing each drug's promiscuity and each target's affinity level — without learning anything about the interaction between them. The 2026 work on this (HonestAffinity, PDBbind CleanSplit) is largely about catching it.

`smpro1.dti` catches it two ways, on the **DAVIS** kinase panel (68 inhibitors × ~379 kinases, dissociation constants; pKd is a free energy, ΔG = −RT ln Kd, ≈1.36 kcal/mol per unit):

1. **Single-sided baselines** — a **protein-only** and a **ligand-only** model. Neither can see the interaction, so any score they earn is memorization by construction.
2. **Cold splits** — hold out drugs, targets, or both, so "will it work on new chemistry / a new target?" is actually the question asked.

![On a random pair split a ligand-only model — which never sees the target — scores R²=0.29; cold splits collapse everything toward zero.](assets/binding_leakage.png)

| model | random pair | cold drug | cold target | cold both |
|:---|---:|---:|---:|---:|
| protein-only (memorization) | 0.07 | 0.08 | −0.00 | 0.00 |
| ligand-only (memorization) | 0.29 | −0.06 | 0.30 | −0.04 |
| **full (ligand + protein)** | **0.53** | **0.09** | **0.36** | **0.02** |

*Test R² on pKd.* Three things the random-split number alone does not show:

- **The full model's 0.53 is mostly not interaction.** A ligand-only model — which literally never sees the protein it is scoring against — reaches 0.29 on the same split. The *interaction gap* (full minus the best single-sided baseline) is only **+0.25**.
- **Cold-both is essentially unsolved.** For a genuinely new drug against a new target, every model sits at R² ≈ 0. That is also the deployment scenario, and nothing here addresses it.
- **The asymmetry follows the dataset's shape.** On `cold_target`, ligand-only (0.30) nearly matches full (0.36): with only 68 drugs — all seen in training — the model rides drug identity. On `cold_drug`, ligand-only goes *negative*: new chemistry breaks the memorization it was leaning on. Which side leaks depends on which side is small.

Run it: `python scripts/run_binding_benchmark.py`.

## Part of a portfolio

`smpro1` extends the honest-evaluation work from polymers into drug discovery. Companion repos:

- [**polytools**](https://github.com/DrGregPitch/polytools) — the harness this reuses; polymer property prediction.
- [**copolybench**](https://github.com/DrGregPitch/copolybench) — copolymer representation learning.
- [**formulate**](https://github.com/DrGregPitch/formulate) — active learning for formulation.

## Limitations

Binding uses sequence-composition protein features, not 3D structure or docking — that's the next tier. DAVIS is a single kinase panel censored at pKd 5 (mostly non-binders), and the property benchmarks are three regression sets: the leakage pattern is general, the exact numbers are dataset-specific. Scaffold and cluster splits are strong structured splits but not the last word — time-based splits stress a model differently again.

## License

MIT.
