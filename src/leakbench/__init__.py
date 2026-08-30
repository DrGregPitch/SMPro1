"""leakbench -- honest evaluation for drug-discovery molecular property prediction.

A random train/test split flatters a molecular model because chemical datasets are
dense with near-duplicate analogues. This package measures that leakage and reports
the number that survives it: performance under a scaffold or cluster split, with the
split's difficulty quantified rather than asserted.

Reuses the model ladder and calibration machinery from ``polytools`` -- the same
honest-evaluation infrastructure, now pointed at drug-like small molecules instead
of polymers, because the physics (energies and geometry setting a property) is the
same object at a different scale.
"""

from .data import DATASETS, MoleculeDataset, load_dataset
from .featurize import descriptors, featurize, morgan
from .splits import (
    SPLITTERS,
    Split,
    cluster_split,
    random_split,
    scaffold_split,
    split_difficulty,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DATASETS", "MoleculeDataset", "load_dataset",
    "featurize", "morgan", "descriptors",
    "Split", "random_split", "scaffold_split", "cluster_split",
    "split_difficulty", "SPLITTERS",
]
