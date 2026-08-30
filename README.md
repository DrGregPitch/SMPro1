# leakbench

**Honest evaluation for drug-discovery molecular property prediction — how much a random split lies, measured.**

![CI](https://github.com/DrGregPitch/leakbench/actions/workflows/ci.yml/badge.svg)
&nbsp;·&nbsp; MIT &nbsp;·&nbsp; Python 3.10–3.12

A molecular ML model that looks excellent on a random train/test split can be nearly useless on the next chemical series a project actually wants to make. Chemical datasets are dense with near-duplicate analogues, so a random test set almost always has a close cousin in training — the model interpolates and the score is inflated. This is the problem a wave of 2026 work is built around (*HonestAffinity*, PDBbind **CleanSplit**, *Systematic Data Leakage in Protein-Ligand Benchmarks*). `leakbench` measures it, and reports the number that survives it.

![Random splits inflate drug-property models across every benchmark; the structured-split bars are the honest numbers.](assets/leakage.png)

## The result

Same model and features on three standard benchmarks — **only the split changes:**

| dataset | random R² | scaffold R² | cluster R² | random inflated by |
|:---|---:|---:|---:|---:|
| Lipophilicity (logD) | 0.68 | 0.60 | 0.52 | +12–31% |
| ESOL (aqueous solubility) | 0.91 | 0.80 | 0.78 | +12% |
| **FreeSolv (hydration ΔG)** | **0.91** | **0.63** | 0.85 | **+31%** |

FreeSolv is the cautionary tale: **R² = 0.91 on a random split, 0.63 on new scaffolds — and RMSE more than doubles, 1.26 → 3.13 kcal/mol.** A team trusting the 0.91 would ship a model that fails on novel chemistry.

The leakage is not asserted, it's measured. `split_difficulty()` reports how far each test set really sits from training:

| split | median nearest-neighbour similarity to train | test with a >0.9 near-duplicate in train |
|:---|---:|---:|
| random | ~0.53 | **6–8%** |
| scaffold | ~0.30 | 0–1% |
| cluster | ~0.32 | 0% |

The random split leaks — several percent of its test set has a near-twin in training — which is exactly why its scores are inflated.

## A physical-chemistry reading

The inflation is **largest for the most physical target.** Hydration free energy (FreeSolv) is a literal ΔG — an interaction energy set by specific functional groups and geometry — so a genuinely new scaffold changes it in ways an interpolating model can't see, and the honest (scaffold) score drops hardest. Solubility and lipophilicity are softer, smoother functions of structure, and leak less. Binding affinity, molecular stability, and glass-transition temperature are the same object at other scales: **a free energy over a geometry.** Evaluating any of them on a random split measures memorisation, not the physics.

## Run it

```bash
git clone https://github.com/DrGregPitch/leakbench && cd leakbench
uv venv && uv pip install -e ".[dev]"        # pulls the model ladder from polytools
uv run python scripts/run_leakage_benchmark.py --outdir results   # ~1 min: table + figure
uv run pytest tests -v
```

Datasets are the canonical MoleculeNet / Therapeutics Data Commons regression sets, downloaded on demand (no data committed).

## What's inside

- **Honest splitters, implemented** (`splits.py`) — random, Bemis–Murcko scaffold, and Butina-cluster, plus `split_difficulty()`, which *measures* how out-of-distribution a split is rather than asserting it. Note the hardest split isn't always the same one (scaffold is hardest for FreeSolv, cluster for Lipophilicity) — reported, not glossed.
- **Model ladder + calibration, reused from [`polytools`](https://github.com/DrGregPitch/polytools)** — the same honest-evaluation infrastructure built for polymers works unchanged on drug-like molecules, because a feature matrix doesn't care what kind of molecule it came from. Gradient boosting on ECFP + medicinal-chemistry descriptors, with deep-ensemble uncertainty and a `SigmaRecalibrator`.
- **Med-chem featurization** (`featurize.py`) — ECFP4 fingerprints + the descriptor block a chemist reads off a structure (MolWt, LogP, TPSA, HBD/HBA, …). Deliberately classical, so no fancy model can hide a leaky split behind an impressive number.

## Part of a portfolio

`leakbench` extends the honest-evaluation thesis from polymers into drug discovery. Companion repos:

- [**polytools**](https://github.com/DrGregPitch/polytools) — the harness this reuses; polymer property prediction.
- [**copolybench**](https://github.com/DrGregPitch/copolybench) — copolymer representation learning.
- [**formulate**](https://github.com/DrGregPitch/formulate) — active learning for formulation.

## Limitations

Ligand-based only (no protein / 3D structure — that's the next tier). Three regression benchmarks, chosen to make the leakage point cleanly; the effect is systematic but its size is dataset-dependent. Scaffold and cluster splits are strong structured splits but not the last word — time-based and target-based splits stress a model differently.

## License

MIT.
