"""Experiment report generation (JSON + CSV)."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd

from .drift_detector import PSI_DISCLAIMER, summarize_drift

SYNTHETIC_WARNING = (
    "All data in this report is synthetic and was generated for demonstration. "
    "Results do not describe any real-world dataset or production model."
)


class ReportError(ValueError):
    """Raised when a report cannot be built."""


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    return obj


def build_report(
    settings: Dict[str, Any],
    dataset_sizes: Dict[str, int],
    drift_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    shifted_features: list,
) -> Dict[str, Any]:
    """Assemble the experiment report as a JSON-serializable dict."""
    if drift_df is None or metrics_df is None:
        raise ReportError("Drift and metric results are required.")
    threshold = settings.get("psi_threshold")
    summary = summarize_drift(drift_df, threshold)
    report = {
        "title": "AI-DriftLab Experiment Report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning": SYNTHETIC_WARNING,
        "settings": settings,
        "random_seed": settings.get("seed"),
        "dataset_sizes": dataset_sizes,
        "shifted_features": list(shifted_features),
        "summary": {
            "features_total": summary["n_features"],
            "features_drifted": summary["n_drifted"],
            "max_psi": summary["max_psi"],
            "mean_psi": summary["mean_psi"],
            "psi_note": PSI_DISCLAIMER,
        },
        "feature_drift": drift_df.to_dict(orient="records"),
        "model_metrics": metrics_df.to_dict(orient="records"),
    }
    report = _to_native(report)
    json.dumps(report)  # fail early if anything is not serializable
    return report


def report_to_json(report: Dict[str, Any]) -> str:
    """Serialize the report to a pretty JSON string."""
    return json.dumps(report, indent=2, ensure_ascii=False)


def drift_to_csv(drift_df: pd.DataFrame) -> str:
    """Serialize the feature-drift table to CSV text (UTF-8, header included)."""
    if drift_df is None or drift_df.empty:
        raise ReportError("No drift results to export.")
    buffer = io.StringIO()
    drift_df.to_csv(buffer, index=False)
    return buffer.getvalue()
