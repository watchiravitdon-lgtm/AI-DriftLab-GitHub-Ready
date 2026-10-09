"""Synthetic data generation and distribution-shift simulation.

All datasets are synthetic (``sklearn.datasets.make_classification``) and exist
only for demonstrating drift detection. They do not represent real-world data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

MIN_SAMPLES = 300
MAX_SAMPLES = 20000
MIN_FEATURES = 2
MAX_FEATURES = 20
MAX_SHIFT_STRENGTH = 5.0
MIN_FRACTION = 0.1
MAX_FRACTION = 0.8


class DataGenerationError(ValueError):
    """Raised when data generation parameters are invalid."""


@dataclass
class ExperimentData:
    """Container for the three datasets used in an experiment."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_reference: pd.DataFrame
    y_reference: pd.Series
    X_current: pd.DataFrame
    y_current: pd.Series
    shifted_features: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)


def validate_params(
    n_samples: int,
    n_features: int,
    train_fraction: float,
    reference_fraction: float,
    shift_strength: float,
    n_shifted_features: int,
    seed: int,
) -> None:
    """Validate generation parameters and raise ``DataGenerationError`` if invalid."""
    if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, bool):
        raise DataGenerationError("n_samples must be an integer.")
    if not MIN_SAMPLES <= n_samples <= MAX_SAMPLES:
        raise DataGenerationError(
            f"n_samples must be between {MIN_SAMPLES} and {MAX_SAMPLES}."
        )
    if not isinstance(n_features, (int, np.integer)) or isinstance(n_features, bool):
        raise DataGenerationError("n_features must be an integer.")
    if not MIN_FEATURES <= n_features <= MAX_FEATURES:
        raise DataGenerationError(
            f"n_features must be between {MIN_FEATURES} and {MAX_FEATURES}."
        )
    for name, value in (
        ("train_fraction", train_fraction),
        ("reference_fraction", reference_fraction),
    ):
        if not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
            raise DataGenerationError(f"{name} must be a finite number.")
        if not MIN_FRACTION <= value <= MAX_FRACTION:
            raise DataGenerationError(
                f"{name} must be between {MIN_FRACTION} and {MAX_FRACTION}."
            )
    if train_fraction + reference_fraction >= 0.95:
        raise DataGenerationError(
            "train_fraction + reference_fraction must be below 0.95 so the "
            "current dataset is not empty."
        )
    if not isinstance(shift_strength, (int, float, np.floating)) or not np.isfinite(
        shift_strength
    ):
        raise DataGenerationError("shift_strength must be a finite number.")
    if not 0.0 <= shift_strength <= MAX_SHIFT_STRENGTH:
        raise DataGenerationError(
            f"shift_strength must be between 0 and {MAX_SHIFT_STRENGTH}."
        )
    if not isinstance(n_shifted_features, (int, np.integer)) or isinstance(
        n_shifted_features, bool
    ):
        raise DataGenerationError("n_shifted_features must be an integer.")
    if not 0 <= n_shifted_features <= n_features:
        raise DataGenerationError(
            "n_shifted_features must be between 0 and the number of features."
        )
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise DataGenerationError("seed must be a non-negative integer.")


def apply_shift(
    X: pd.DataFrame,
    columns: List[str],
    strength: float,
    reference_std: pd.Series,
) -> pd.DataFrame:
    """Return a *copy* of ``X`` with a mean shift applied to ``columns``.

    The shift is ``strength`` times the reference standard deviation of each
    column, so the same strength has a comparable effect on every feature.
    The input frame is never modified.
    """
    shifted = X.copy(deep=True)
    for col in columns:
        if col not in shifted.columns:
            raise DataGenerationError(f"Column '{col}' not found for shifting.")
        shifted[col] = shifted[col] + strength * float(reference_std[col])
    return shifted


def generate_experiment_data(
    n_samples: int = 3000,
    n_features: int = 8,
    train_fraction: float = 0.5,
    reference_fraction: float = 0.25,
    shift_strength: float = 1.0,
    n_shifted_features: int = 3,
    seed: int = 42,
) -> ExperimentData:
    """Generate Training / Reference / Current datasets.

    The data is split into three disjoint parts. The Current dataset is the
    only one modified by the shift; Training and Reference are left untouched.
    """
    validate_params(
        n_samples,
        n_features,
        train_fraction,
        reference_fraction,
        shift_strength,
        n_shifted_features,
        seed,
    )
    n_samples = int(n_samples)
    n_features = int(n_features)
    seed = int(seed)

    n_informative = max(2, n_features // 2)
    n_redundant = min(max(0, n_features // 4), n_features - n_informative)

    X_arr, y_arr = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        class_sep=1.0,
        flip_y=0.02,
        random_state=seed,
    )
    feature_names = [f"feature_{i + 1}" for i in range(n_features)]
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="target")

    # Two-step split. The held-out share is always strictly between 0 and 1
    # (train_fraction is capped at 0.8), so the splitter never receives a
    # test size that would leave the training side empty.
    holdout_fraction = 1.0 - train_fraction
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=holdout_fraction, random_state=seed, stratify=y
    )
    ref_share_of_rest = reference_fraction / holdout_fraction
    # ref_share_of_rest is in (0, 1) because train + reference < 0.95.
    X_reference, X_current, y_reference, y_current = train_test_split(
        X_rest,
        y_rest,
        train_size=ref_share_of_rest,
        random_state=seed,
        stratify=y_rest,
    )

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_reference = X_reference.reset_index(drop=True)
    y_reference = y_reference.reset_index(drop=True)
    X_current = X_current.reset_index(drop=True)
    y_current = y_current.reset_index(drop=True)

    if min(len(X_train), len(X_reference), len(X_current)) == 0:
        raise DataGenerationError("The split produced an empty dataset.")

    rng = np.random.default_rng(seed)
    shifted_idx = (
        sorted(rng.choice(n_features, size=n_shifted_features, replace=False).tolist())
        if n_shifted_features > 0
        else []
    )
    shifted_features = [feature_names[i] for i in shifted_idx]

    reference_std = X_reference.std(ddof=0)
    X_current = apply_shift(X_current, shifted_features, shift_strength, reference_std)

    return ExperimentData(
        X_train=X_train,
        y_train=y_train,
        X_reference=X_reference,
        y_reference=y_reference,
        X_current=X_current,
        y_current=y_current,
        shifted_features=shifted_features,
        feature_names=feature_names,
    )
