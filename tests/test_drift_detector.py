import numpy as np
import pandas as pd
import pytest

from src.drift_detector import DriftError, calculate_psi, detect_drift, summarize_drift


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_psi_identical_is_zero(rng):
    x = rng.normal(size=2000)
    assert calculate_psi(x, x.copy()) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_shift(rng):
    ref = rng.normal(size=3000)
    cur = rng.normal(loc=2.0, size=3000)
    assert calculate_psi(ref, cur) > 0.25


def test_psi_nonnegative_and_finite(rng):
    ref = rng.normal(size=500)
    cur = rng.normal(loc=100, size=500)  # entirely outside reference range
    psi = calculate_psi(ref, cur)
    assert np.isfinite(psi) and psi >= 0


def test_psi_constant_feature_same():
    assert calculate_psi(np.ones(100), np.ones(100)) == pytest.approx(0.0, abs=1e-9)


def test_psi_constant_feature_shifted():
    psi = calculate_psi(np.ones(100), np.full(100, 5.0))
    assert np.isfinite(psi) and psi > 0.25


def test_psi_ignores_nan_and_inf(rng):
    ref = rng.normal(size=500)
    cur = np.concatenate([ref, [np.nan, np.inf, -np.inf]])
    assert calculate_psi(ref, cur) == pytest.approx(0.0, abs=1e-9)


def test_psi_empty_or_all_nan_raises():
    with pytest.raises(DriftError):
        calculate_psi([], [1.0, 2.0])
    with pytest.raises(DriftError):
        calculate_psi([np.nan, np.nan], [1.0])


def test_psi_invalid_bins(rng):
    with pytest.raises(DriftError):
        calculate_psi(rng.normal(size=50), rng.normal(size=50), bins=1)


def test_detect_drift_status_and_threshold(rng):
    ref = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000)})
    cur = ref.copy()
    cur["b"] = cur["b"] + 3.0
    out = detect_drift(ref, cur, threshold=0.2)
    assert list(out.columns) == ["feature", "psi", "status"]
    assert out.set_index("feature").loc["a", "status"] == "OK"
    assert out.set_index("feature").loc["b", "status"] == "DRIFT"
    strict = detect_drift(ref, cur, threshold=1e9)
    assert (strict["status"] == "OK").all()


def test_detect_drift_column_mismatch():
    ref = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    cur = pd.DataFrame({"b": [1.0, 2.0, 3.0]})
    with pytest.raises(DriftError, match="mismatch"):
        detect_drift(ref, cur)


def test_detect_drift_empty_and_bad_types():
    full = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(DriftError):
        detect_drift(full.iloc[0:0], full)
    with pytest.raises(DriftError):
        detect_drift(full, full.iloc[0:0])
    with pytest.raises(DriftError):
        detect_drift([1, 2], full)


@pytest.mark.parametrize("thr", [0, -0.1, float("nan"), float("inf")])
def test_detect_drift_invalid_threshold(thr):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(DriftError):
        detect_drift(df, df, threshold=thr)


def test_detect_drift_with_nan_column(rng):
    ref = pd.DataFrame({"a": rng.normal(size=300)})
    cur = ref.copy()
    cur.loc[:20, "a"] = np.nan
    out = detect_drift(ref, cur)
    assert np.isfinite(out["psi"]).all()


def test_summarize_drift():
    df = pd.DataFrame({"feature": ["a", "b"], "psi": [0.05, 0.5], "status": ["OK", "DRIFT"]})
    s = summarize_drift(df, 0.2)
    assert s["n_drifted"] == 1 and s["max_psi"] == 0.5 and "heuristic" in s["disclaimer"]
