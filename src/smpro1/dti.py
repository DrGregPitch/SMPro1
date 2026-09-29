"""Drug-target binding affinity: data, pair featurization, and honest pair splits.

Binding affinity is the physical chemist's quantity: pKd is a free energy in
disguise (dG_bind = -RT ln Kd; one pKd unit ~ 1.36 kcal/mol at 298 K). It is also
where evaluation leakage does the most damage, because a drug-target *pair* can
leak through either side: a random pair split lets the model memorize each drug's
promiscuity profile and each target's affinity level without learning anything
about the interaction between them.

The honest evaluation therefore needs two things this module provides:

* **Cold splits** -- hold out drugs (``cold_drug``), targets (``cold_target``),
  or both (``cold_both``), so the deployment question "will it work on new
  chemistry / a new target?" is actually asked.
* **Single-sided baselines** -- models that see ONLY the ligand or ONLY the
  protein. Any performance they achieve is memorization by construction; the gap
  between the full model and the best single-sided baseline is the part that is
  genuinely learned interaction.
"""

from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from .featurize import morgan
from .splits import Split

RDLogger.DisableLog("rdApp.*")

#: TDC (MIT) redistributes the classic DTI sets via Harvard Dataverse.
_DAVIS_URL = "https://dataverse.harvard.edu/api/access/datafile/5219748"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

__all__ = [
    "DTIDataset", "load_davis", "protein_features", "pair_features",
    "pair_split", "pair_split_report",
]


@dataclass
class DTIDataset:
    """Validated drug-target pairs with a pKd label."""

    drugs: np.ndarray          # canonical SMILES per pair
    targets: np.ndarray        # target id per pair
    seqs: dict                 # target id -> sequence
    mols: dict                 # canonical SMILES -> RDKit mol
    y: np.ndarray              # pKd per pair
    report: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"DTIDataset(pairs={len(self)}, drugs={len(self.mols)}, "
                f"targets={len(self.seqs)})")


def _fetch_url(url: str, dest: Path) -> Path:
    """Atomic download: temp file then rename, so a dropped connection can
    never leave a truncated file that poisons the cache."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        # Harvard Dataverse 403s the default urllib User-Agent; present a
        # browser-like one so the fetch works from CI runners too.
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:  # noqa: S310 - trusted https
            shutil.copyfileobj(r, f)
        tmp.rename(dest)
    return dest


def load_davis(cache: str | Path = "data_cache") -> DTIDataset:
    """The DAVIS kinase panel: 68 inhibitors x ~379 kinases, dissociation constants.

    Labels are converted to pKd = 9 - log10(Kd_nM). A known property worth
    stating rather than hiding: the assay panel is censored at 10 uM, so a large
    majority of pairs sit exactly at pKd = 5 ("no measurable binding") -- the
    dataset is mostly non-binders, and models are largely being asked to find
    the binders. The load report carries the censored fraction.
    """
    path = _fetch_url(_DAVIS_URL, Path(cache) / "davis.tab")
    df = pd.read_csv(path, sep="\t")
    df.columns = ["drug_id", "smiles", "target_id", "seq", "kd_nm"]
    n_raw = len(df)
    df = df.dropna(subset=["smiles", "seq", "kd_nm"])

    mols: dict[str, object] = {}
    canon: dict[str, str] = {}
    n_bad = 0
    for smi in df["smiles"].unique():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            n_bad += 1
            continue
        cs = Chem.MolToSmiles(m)
        canon[smi] = cs
        mols.setdefault(cs, m)
    df = df[df["smiles"].isin(canon)]
    drugs = df["smiles"].map(canon).to_numpy(dtype=object)

    y = 9.0 - np.log10(df["kd_nm"].to_numpy(dtype=float))
    seqs = dict(df.groupby("target_id")["seq"].first().items())

    report = {
        "pairs_in_file": n_raw,
        "unparseable_drugs": n_bad,
        "pairs_final": len(df),
        "n_drugs": len(mols),
        "n_targets": len(seqs),
        "censored_frac_pkd5": float((y == 5.0).mean()),
    }
    return DTIDataset(drugs=drugs, targets=df["target_id"].to_numpy(dtype=object),
                      seqs=seqs, mols=mols, y=y, report=report)


def protein_features(seqs: dict) -> tuple[dict, list[str]]:
    """Classical sequence-composition features per target.

    Amino-acid composition (20) + dipeptide composition (400) + log-length: the
    standard alignment-free baseline. Deliberately classical -- the point of this
    module is honest evaluation, and a simple representation cannot hide a leaky
    split behind an impressive architecture.
    """
    aa_index = {a: i for i, a in enumerate(AMINO_ACIDS)}
    dipep = ["".join(p) for p in product(AMINO_ACIDS, repeat=2)]
    dp_index = {d: i for i, d in enumerate(dipep)}

    out: dict[str, np.ndarray] = {}
    for tid, s in seqs.items():
        v = np.zeros(20 + 400 + 1)
        n = max(len(s), 1)
        for ch in s:
            if ch in aa_index:
                v[aa_index[ch]] += 1
        for i in range(len(s) - 1):
            d = s[i:i + 2]
            if d in dp_index:
                v[20 + dp_index[d]] += 1
        v[:20] /= n
        v[20:420] /= max(n - 1, 1)
        v[420] = np.log10(n)
        out[tid] = v
    names = [f"aac_{a}" for a in AMINO_ACIDS] + [f"dp_{d}" for d in dipep] + ["log_len"]
    return out, names


def pair_features(ds: DTIDataset, n_bits: int = 1024) -> dict[str, np.ndarray]:
    """Per-pair feature blocks: ``ligand``, ``protein``, and ``full`` (concat).

    The single-sided blocks exist on purpose: they are the memorization
    baselines. Ligand fingerprints and protein compositions are computed once
    per unique entity and broadcast to pairs.
    """
    uniq_smiles = list(ds.mols)
    L_uniq = morgan([ds.mols[s] for s in uniq_smiles], n_bits=n_bits)
    l_index = {s: i for i, s in enumerate(uniq_smiles)}
    L = L_uniq[[l_index[s] for s in ds.drugs]]

    pf, _ = protein_features(ds.seqs)
    P = np.vstack([pf[t] for t in ds.targets])

    return {"ligand": L, "protein": P, "full": np.hstack([L, P])}


def pair_split(ds: DTIDataset, kind: str, frac_test: float = 0.2,
               seed: int = 0) -> Split:
    """Pair-level splits that ask increasingly honest questions.

    * ``random_pair`` -- the optimistic reference; both sides of most test pairs
      were seen in training.
    * ``cold_drug`` -- held-out drugs: "will it work on new chemistry?"
    * ``cold_target`` -- held-out targets: "will it work on a new protein?"
    * ``cold_both`` -- both unseen; the test set is the *intersection* (pairs of
      held-out drug x held-out target), train excludes both hold-out sets
      entirely, so nothing on either side of a test pair was ever seen.
    """
    rng = np.random.default_rng(seed)
    n = len(ds)

    if kind == "random_pair":
        order = rng.permutation(n)
        k = int(round((1 - frac_test) * n))
        return Split(train=order[:k], test=order[k:])

    drugs = list(ds.mols)
    targets = list(ds.seqs)
    rng.shuffle(drugs)
    rng.shuffle(targets)
    hold_d = set(drugs[: max(1, int(round(frac_test * len(drugs))))])
    hold_t = set(targets[: max(1, int(round(frac_test * len(targets))))])
    in_d = np.array([d in hold_d for d in ds.drugs])
    in_t = np.array([t in hold_t for t in ds.targets])

    if kind == "cold_drug":
        test = in_d
        train = ~in_d
    elif kind == "cold_target":
        test = in_t
        train = ~in_t
    elif kind == "cold_both":
        test = in_d & in_t
        train = ~in_d & ~in_t          # exclude every pair touching a held-out entity
    else:
        raise ValueError(f"unknown split kind {kind!r}")
    return Split(train=np.where(train)[0], test=np.where(test)[0])


def pair_split_report(ds: DTIDataset, split: Split) -> dict[str, float]:
    """How leaky is this pair split, measured -- the DTI analogue of
    ``split_difficulty``: what fraction of test pairs have their drug (or
    target) present somewhere in training?"""
    train_drugs = set(ds.drugs[split.train])
    train_targets = set(ds.targets[split.train])
    td = np.array([d in train_drugs for d in ds.drugs[split.test]])
    tt = np.array([t in train_targets for t in ds.targets[split.test]])
    return {
        "n_train": float(len(split.train)),
        "n_test": float(len(split.test)),
        "frac_test_drug_seen": float(td.mean()),
        "frac_test_target_seen": float(tt.mean()),
    }
