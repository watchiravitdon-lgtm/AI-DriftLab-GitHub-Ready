import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_experiment_data
from src.model_evaluator import (
    METRIC_NAMES,
    EvaluationError,
    build_models,
    compare_reference_current,
    compute_metrics,
    train_and_evaluate,
)


@pytest.fixture(scope="module")
def data():
    return generate_experiment_data(n_samples=1000, seed=1)


def _run(d, **over):
    args = dict(
        X_train=d.X_train, y_train=d.y_train, X_reference=d.X_reference,
        y_reference=d.y_reference, X_current=d.X_current, y_current=d.y_current,
    )
    args.update(over)
    return train_and_evaluate(**args)


def test_train_and_evaluate_shape(data):
    res = _run(data)
    assert len(res) == 4  # 2 models x 2 datasets
    assert set(res["model"]) == {"Logistic Regression", "Random Forest"}
    assert set(res["dataset"]) == {"Reference", "Current"}
    assert list(res.columns) == ["model", "dataset"] + METRIC_NAMES


def test_metrics_in_range(data):
    res = _run(data)
    assert ((res[METRIC_NAMES] >= 0) & (res[METRIC_NAMES] <= 1)).all().all()


def test_models_learn_something(data):
    res = _run(data)
    ref = res[res["dataset"] == "Reference"]
    assert (ref["accuracy"] > 0.6).all()


def test_deterministic_with_seed(data):
    pd.testing.assert_frame_equal(_run(data, seed=5), _run(data, seed=5))


def test_models_use_seed_and_scaler():
    models = build_models(11)
    assert models["Random Forest"].random_state == 11
    assert models["Logistic Regression"].named_steps["clf"].random_state == 11
    assert "scaler" in models["Logistic Regression"].named_steps


def test_compute_metrics_zero_division():
    m = compute_metrics([0, 0, 0, 0], [0, 0, 0, 0])  # no positives predicted/true
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    assert m["accuracy"] == 1.0


def test_single_class_training_raises(data):
    with pytest.raises(EvaluationError, match="one class"):
        _run(data, y_train=pd.Series(np.zeros(len(data.y_train), dtype=int)))


def test_empty_and_small_training_raises(data):
    with pytest.raises(EvaluationError):
        _run(data, X_current=data.X_current.iloc[0:0], y_current=data.y_current.iloc[0:0])
    with pytest.raises(EvaluationError, match="at least"):
        _run(data, X_train=data.X_train.iloc[:5], y_train=data.y_train.iloc[:5])


def test_length_and_column_mismatch_raise(data):
    with pytest.raises(EvaluationError, match="different lengths"):
        _run(data, y_reference=data.y_reference.iloc[:-1])
    with pytest.raises(EvaluationError, match="columns"):
        _run(data, X_current=data.X_current.rename(columns={"feature_1": "zzz"}))


def test_nan_features_raise(data):
    bad = data.X_current.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(EvaluationError, match="NaN"):
        _run(data, X_current=bad)


def test_compare_reference_current(data):
    comp = compare_reference_current(_run(data))
    assert len(comp) == 2 * len(METRIC_NAMES)
    row = comp.iloc[0]
    assert row["delta"] == pytest.approx(row["current"] - row["reference"])
