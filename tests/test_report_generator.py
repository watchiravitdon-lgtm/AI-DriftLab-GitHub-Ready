import io
import json

import numpy as np
import pandas as pd
import pytest

from src.report_generator import (
    SYNTHETIC_WARNING,
    ReportError,
    build_report,
    drift_to_csv,
    report_to_json,
)


@pytest.fixture
def parts():
    drift = pd.DataFrame(
        {"feature": ["a", "b"], "psi": [np.float64(0.05), np.float64(0.4)], "status": ["OK", "DRIFT"]}
    )
    metrics = pd.DataFrame(
        {
            "model": ["M"] * 2,
            "dataset": ["Reference", "Current"],
            "accuracy": [0.9, 0.8], "precision": [0.9, 0.8], "recall": [0.9, 0.8], "f1": [0.9, 0.8],
        }
    )
    settings = {"seed": 42, "psi_threshold": 0.2, "n_samples": np.int64(1000)}
    sizes = {"training": 500, "reference": 250, "current": 250}
    return settings, sizes, drift, metrics


def test_build_report_contents(parts):
    settings, sizes, drift, metrics = parts
    r = build_report(settings, sizes, drift, metrics, ["b"])
    assert r["random_seed"] == 42
    assert r["dataset_sizes"] == sizes
    assert r["warning"] == SYNTHETIC_WARNING and "synthetic" in r["warning"].lower()
    assert r["generated_at_utc"].endswith("+00:00")
    assert r["summary"]["features_drifted"] == 1
    assert r["feature_drift"][1]["status"] == "DRIFT"
    assert len(r["model_metrics"]) == 2


def test_json_roundtrip(parts):
    r = build_report(*parts, ["b"])
    text = report_to_json(r)
    assert json.loads(text) == r


def test_numpy_types_and_nan_serialize(parts):
    settings, sizes, drift, metrics = parts
    metrics = metrics.copy()
    metrics.loc[0, "f1"] = np.nan
    r = build_report(settings, sizes, drift, metrics, [])
    json.dumps(r, allow_nan=False)  # strict JSON: NaN must have become null
    assert r["model_metrics"][0]["f1"] is None


def test_csv_export_opens_as_table(parts):
    drift = parts[2]
    text = drift_to_csv(drift)
    back = pd.read_csv(io.StringIO(text))
    assert list(back.columns) == ["feature", "psi", "status"]
    assert len(back) == 2


def test_invalid_inputs(parts):
    settings, sizes, drift, metrics = parts
    with pytest.raises(ReportError):
        build_report(settings, sizes, None, metrics, [])
    with pytest.raises(ReportError):
        build_report(settings, sizes, drift, None, [])
    with pytest.raises(ReportError):
        drift_to_csv(drift.iloc[0:0])
    with pytest.raises(ReportError):
        drift_to_csv(None)
