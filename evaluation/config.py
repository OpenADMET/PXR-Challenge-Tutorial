"""Configuration file for the OpenADMET PXR blind challenge."""

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


def rae(y_true, y_pred):
    """Relative absolute error (RAE) metric for regression tasks."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))


# Activity dataset
ENDPOINTS = ["pEC50"]
ENDPOINTS_TO_LOG_TRANSFORM: list[str] = []
ACTIVITY_METRICS = [
    ("MAE", mean_absolute_error),
    ("RAE", rae),
    ("R2", r2_score),
    ("Spearman R", spearmanr),
    ("Kendall's Tau", kendalltau),
]
BOOTSTRAP_SAMPLES = 1000

# Structure dataset
STRUCTURE_METRICS = ["LDDT-PLI", "BiSyRMSD", "LDDT-LP"]
# Penalty applied to BiSyRMSD when OST cannot match the ligand (lower is better,
# so a large value is used; 20 Å is well outside any reasonable binding-site RMSD)
BISYRMSD_NAN_PENALTY: float = 20.0
