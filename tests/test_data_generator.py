import numpy as np
import pandas as pd
import pytest

from src.data_generator import (
    MAX_FEATURES,
    MAX_SAMPLES,
    MIN_FEATURES,
    MIN_SAMPLES,
    DataGenerationError,
    apply_shift,
    generate_experiment_data,
)


def test_generation_succeeds():
    d = generate_experiment_data()
    assert len(d.X_train) > 0 and len(d.X_reference) > 0 and len(d.X_current) > 0
    total = len(d.X_train) + len(d.X_reference) + len(d.X_current)
    assert total == 3000
    assert list(d.X_train.columns) == list(d.X_current.columns)
    assert set(d.y_train.unique()) == {0, 1}


def test_split_minimum_boundary():
    d = generate_experiment_data(
        n_samples=MIN_SAMPLES, n_features=MIN_FEATURES, train_fraction=0.1,
        reference_fraction=0.1, n_shifted_features=0,
    )
    assert min(len(d.X_train), len(d.X_reference), len(d.X_current)) > 0


def test_split_maximum_boundary():
    d = generate_experiment_data(
        n_samples=MAX_SAMPLES, n_features=MAX_FEATURES, train_fraction=0.8,
        reference_fraction=0.1, shift_strength=5.0, n_shifted_features=MAX_FEATURES,
    )
    assert len(d.X_current) > 0
    assert len(d.X_train) + len(d.X_reference) + len(d.X_current) == MAX_SAMPLES


def test_seed_reproducibility():
    a = generate_experiment_data(seed=7)
    b = generate_experiment_data(seed=7)
    c = generate_experiment_data(seed=8)
    pd.testing.assert_frame_equal(a.X_current, b.X_current)
    assert a.shifted_features == b.shifted_features
    assert not a.X_train.equals(c.X_train)


def test_shift_only_affects_current():
    base = generate_experiment_data(shift_strength=0.0, n_shifted_features=0, seed=3)
    shifted = generate_experiment_data(shift_strength=2.0, n_shifted_features=2, seed=3)
    pd.testing.assert_frame_equal(base.X_train, shifted.X_train)
    pd.testing.assert_frame_equal(base.X_reference, shifted.X_reference)
    changed = [
        c for c in base.X_current.columns
        if not np.allclose(base.X_current[c], shifted.X_current[c])
    ]
    assert sorted(changed) == sorted(shifted.shifted_features)
    assert len(changed) == 2


def test_apply_shift_does_not_mutate_input():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 1.0, 1.0]})
    before = X.copy()
    out = apply_shift(X, ["a"], 1.0, X.std(ddof=0))
    pd.testing.assert_frame_equal(X, before)
    assert not np.allclose(out["a"], X["a"])


def test_apply_shift_unknown_column():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(DataGenerationError):
        apply_shift(X, ["zzz"], 1.0, X.std(ddof=0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 10},
        {"n_samples": MAX_SAMPLES + 1},
        {"n_samples": 1000.5},
        {"n_features": 1},
        {"n_features": MAX_FEATURES + 1},
        {"train_fraction": 0.0},
        {"train_fraction": 0.9},
        {"train_fraction": 0.8, "reference_fraction": 0.8},
        {"train_fraction": 0.8, "reference_fraction": 0.2},
        {"reference_fraction": float("nan")},
        {"shift_strength": -1.0},
        {"shift_strength": 99.0},
        {"shift_strength": float("inf")},
        {"n_shifted_features": -1},
        {"n_shifted_features": 99},
        {"seed": -5},
        {"seed": True},
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(DataGenerationError):
        generate_experiment_data(**kwargs)
