"""Population Stability Index (PSI) based feature drift detection.

Note: PSI thresholds (commonly 0.1 for "moderate" and 0.25 for "major" shift)
are *heuristics* from credit-risk practice, not universal rules. Choose a
threshold appropriate for your data and risk tolerance.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

EPSILON = 1e-6
DEFAULT_BINS = 10
DEFAULT_THRESHOLD = 0.2

PSI_DISCLAIMER = (
    "PSI thresholds are heuristics, not universal rules. Common rules of thumb "
    "(<0.1 stable, 0.1-0.25 moderate, >0.25 major) come from credit-risk "
    "practice and should be tuned for your own data."
)


class DriftError(ValueError):
    """Raised when drift detection inputs are invalid."""


def _clean(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def calculate_psi(reference, current, bins: int = DEFAULT_BINS) -> float:
    """Compute PSI between two 1-D samples.

    Bin edges are built from the reference sample (quantiles) and reused for
    the current sample, with open-ended outer bins so current values outside
    the reference range are still counted. NaN/inf values are dropped.
    Constant reference features return 0.0 if current is identical-valued,
    otherwise the proportions are compared using a single reference bin.
    Probabilities are floored at ``EPSILON`` to avoid log(0) and division by 0.
    """
    if not isinstance(bins, (int, np.integer)) or bins < 2:
        raise DriftError("bins must be an integer >= 2.")
    ref = _clean(reference)
    cur = _clean(current)
    if ref.size == 0 or cur.size == 0:
        raise DriftError("Reference and current samples must contain finite values.")

    if np.ptp(ref) == 0:
        # Constant reference: one bin below, one equal, one above the constant.
        c = ref[0]
        edges = np.array([-np.inf, c - EPSILON, c + EPSILON, np.inf])
    else:
        quantiles = np.quantile(ref, np.linspace(0, 1, int(bins) + 1))
        edges = np.unique(quantiles)
        if edges.size < 3:
            edges = np.array([-np.inf, edges[0], edges[-1], np.inf])
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_p = np.clip(ref_counts / ref_counts.sum(), EPSILON, None)
    cur_p = np.clip(cur_counts / cur_counts.sum(), EPSILON, None)
    psi = float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
    return max(psi, 0.0)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    bins: int = DEFAULT_BINS,
) -> pd.DataFrame:
    """Compute per-feature PSI and OK/DRIFT status.

    Returns a DataFrame with columns: feature, psi, status.
    """
    if not isinstance(reference, pd.DataFrame) or not isinstance(current, pd.DataFrame):
        raise DriftError("reference and current must be pandas DataFrames.")
    if reference.empty or current.empty:
        raise DriftError("Reference and current datasets must not be empty.")
    if set(reference.columns) != set(current.columns):
        missing = set(reference.columns) ^ set(current.columns)
        raise DriftError(f"Column mismatch between datasets: {sorted(missing)}")
    if not isinstance(threshold, (int, float, np.floating)) or not np.isfinite(threshold):
        raise DriftError("threshold must be a finite number.")
    if threshold <= 0:
        raise DriftError("threshold must be greater than 0.")

    rows = []
    for col in reference.columns:
        psi = calculate_psi(reference[col].to_numpy(), current[col].to_numpy(), bins)
        rows.append(
            {
                "feature": str(col),
                "psi": round(psi, 6),
                "status": "DRIFT" if psi >= threshold else "OK",
            }
        )
    return pd.DataFrame(rows, columns=["feature", "psi", "status"])


def summarize_drift(drift_df: pd.DataFrame, threshold: Optional[float] = None) -> dict:
    """Return a small summary dict for a drift table."""
    n = len(drift_df)
    n_drift = int((drift_df["status"] == "DRIFT").sum()) if n else 0
    return {
        "n_features": int(n),
        "n_drifted": n_drift,
        "max_psi": float(drift_df["psi"].max()) if n else 0.0,
        "mean_psi": float(drift_df["psi"].mean()) if n else 0.0,
        "threshold": threshold,
        "disclaimer": PSI_DISCLAIMER,
    }
