"""Model training and evaluation on Reference vs Current datasets."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]
MIN_TRAIN_ROWS = 10


class EvaluationError(ValueError):
    """Raised when models cannot be trained or evaluated."""


def build_models(seed: int = 42) -> Dict[str, object]:
    """Create the two models. Seeds are set where supported."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=1
        ),
    }


def _validate_xy(X, y, name: str) -> None:
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise EvaluationError(f"{name}: X must be a DataFrame and y a Series.")
    if len(X) == 0 or len(y) == 0:
        raise EvaluationError(f"{name} dataset is empty.")
    if len(X) != len(y):
        raise EvaluationError(f"{name}: X and y have different lengths.")
    if X.isna().any().any():
        raise EvaluationError(f"{name}: features contain NaN values.")


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Accuracy, precision, recall, F1 (binary, zero_division=0), each in [0, 1]."""
    kwargs = dict(zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, **kwargs)),
        "recall": float(recall_score(y_true, y_pred, **kwargs)),
        "f1": float(f1_score(y_true, y_pred, **kwargs)),
    }
    for key, value in metrics.items():
        if not (0.0 <= value <= 1.0) or not np.isfinite(value):
            raise EvaluationError(f"Metric '{key}' out of range: {value}")
    return metrics


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    X_current: pd.DataFrame,
    y_current: pd.Series,
    seed: int = 42,
) -> pd.DataFrame:
    """Train on the training set only; evaluate on Reference and Current.

    Returns a long-format DataFrame: model, dataset, accuracy, precision,
    recall, f1.
    """
    _validate_xy(X_train, y_train, "Training")
    _validate_xy(X_reference, y_reference, "Reference")
    _validate_xy(X_current, y_current, "Current")
    if len(X_train) < MIN_TRAIN_ROWS:
        raise EvaluationError(
            f"Training set needs at least {MIN_TRAIN_ROWS} rows (got {len(X_train)})."
        )
    if y_train.nunique() < 2:
        raise EvaluationError("Training labels contain only one class; need two.")
    if list(X_train.columns) != list(X_reference.columns) or list(
        X_train.columns
    ) != list(X_current.columns):
        raise EvaluationError("Feature columns differ between datasets.")

    rows = []
    for model_name, model in build_models(seed).items():
        model.fit(X_train, y_train)
        for dataset_name, X, y in (
            ("Reference", X_reference, y_reference),
            ("Current", X_current, y_current),
        ):
            m = compute_metrics(y, model.predict(X))
            rows.append({"model": model_name, "dataset": dataset_name, **m})
    return pd.DataFrame(rows, columns=["model", "dataset"] + METRIC_NAMES)


def compare_reference_current(results: pd.DataFrame) -> pd.DataFrame:
    """Wide comparison: Reference, Current and delta (Current - Reference).

    A negative delta means Current scored lower. Deltas may be zero or positive;
    drift does not always degrade metrics.
    """
    rows = []
    for model_name, group in results.groupby("model", sort=False):
        ref = group[group["dataset"] == "Reference"].iloc[0]
        cur = group[group["dataset"] == "Current"].iloc[0]
        for metric in METRIC_NAMES:
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "reference": float(ref[metric]),
                    "current": float(cur[metric]),
                    "delta": float(cur[metric] - ref[metric]),
                }
            )
    return pd.DataFrame(rows, columns=["model", "metric", "reference", "current", "delta"])
