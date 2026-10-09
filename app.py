"""AI-DriftLab: Streamlit dashboard for data drift detection and model impact."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_generator import (
    MAX_FEATURES,
    MAX_SAMPLES,
    MAX_SHIFT_STRENGTH,
    MIN_FEATURES,
    MIN_SAMPLES,
    DataGenerationError,
    generate_experiment_data,
)
from src.drift_detector import (
    DEFAULT_THRESHOLD,
    PSI_DISCLAIMER,
    DriftError,
    detect_drift,
)
from src.model_evaluator import (
    METRIC_NAMES,
    EvaluationError,
    compare_reference_current,
    train_and_evaluate,
)
from src.report_generator import (
    SYNTHETIC_WARNING,
    ReportError,
    build_report,
    drift_to_csv,
    report_to_json,
)

BLUE = "#4C8DFF"
PURPLE = "#A66BFF"
GREEN = "#2ECC9A"
RED = "#FF6B6B"
BG = "#0E1117"
PANEL = "#161B26"
TEXT = "#E6EAF2"

st.set_page_config(page_title="AI-DriftLab", page_icon="🧪", layout="wide")

CSS = f"""
<style>
.stApp {{ background: linear-gradient(180deg, {BG} 0%, #111628 100%); color: {TEXT}; }}
[data-testid="stSidebar"] {{ background: {PANEL}; }}
.hero {{ padding: 1.2rem 1.5rem; border-radius: 14px; margin-bottom: 1rem;
  background: linear-gradient(120deg, rgba(76,141,255,.25), rgba(166,107,255,.25), rgba(46,204,154,.18));
  border: 1px solid rgba(166,107,255,.35); }}
.hero h1 {{ margin: 0; font-size: 2rem; color: {TEXT}; }}
.hero p {{ margin: .3rem 0 0 0; color: #AAB4C8; }}
.card {{ background: {PANEL}; border: 1px solid rgba(76,141,255,.3); border-radius: 12px;
  padding: 1rem 1.1rem; }}
.card .label {{ color: #8E9AB3; font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; }}
.card .value {{ font-size: 1.7rem; font-weight: 700; color: {TEXT}; }}
.card .sub {{ color: #8E9AB3; font-size: .8rem; }}
.ok {{ color: {GREEN}; font-weight: 700; }}
.drift {{ color: {RED}; font-weight: 700; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color("#2B3347")
    ax.grid(alpha=0.15, color="#8E9AB3")


def new_fig(figsize=(7, 3.8)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    return fig, ax


def style_legend(ax) -> None:
    leg = ax.legend(facecolor=PANEL, edgecolor="#2B3347")
    for text in leg.get_texts():
        text.set_color(TEXT)


# ---------------------------------------------------------------- Sidebar
def sidebar_settings() -> dict:
    st.sidebar.header("Experiment settings")
    seed = st.sidebar.number_input("Random seed", 0, 999999, 42, step=1)
    n_samples = st.sidebar.slider(
        "Dataset size", MIN_SAMPLES, MAX_SAMPLES, 3000, step=100
    )
    n_features = st.sidebar.slider("Number of features", MIN_FEATURES, MAX_FEATURES, 8)
    train_fraction = st.sidebar.slider("Training fraction", 0.10, 0.80, 0.50, 0.05)
    reference_fraction = st.sidebar.slider("Reference fraction", 0.10, 0.80, 0.25, 0.05)
    st.sidebar.caption(
        f"Current fraction = {1 - train_fraction - reference_fraction:.2f}"
    )
    shift_strength = st.sidebar.slider(
        "Shift strength (in reference std)", 0.0, float(MAX_SHIFT_STRENGTH), 1.0, 0.1
    )
    n_shifted = st.sidebar.slider(
        "Features to shift", 0, int(n_features), min(3, int(n_features))
    )
    psi_threshold = st.sidebar.slider(
        "PSI threshold", 0.05, 1.0, float(DEFAULT_THRESHOLD), 0.05
    )
    st.sidebar.caption(PSI_DISCLAIMER)
    return {
        "seed": int(seed),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "train_fraction": float(train_fraction),
        "reference_fraction": float(reference_fraction),
        "shift_strength": float(shift_strength),
        "n_shifted_features": int(n_shifted),
        "psi_threshold": float(psi_threshold),
    }


def run_experiment(settings: dict) -> dict:
    data = generate_experiment_data(
        n_samples=settings["n_samples"],
        n_features=settings["n_features"],
        train_fraction=settings["train_fraction"],
        reference_fraction=settings["reference_fraction"],
        shift_strength=settings["shift_strength"],
        n_shifted_features=settings["n_shifted_features"],
        seed=settings["seed"],
    )
    drift_df = detect_drift(
        data.X_reference, data.X_current, threshold=settings["psi_threshold"]
    )
    metrics_df = train_and_evaluate(
        data.X_train,
        data.y_train,
        data.X_reference,
        data.y_reference,
        data.X_current,
        data.y_current,
        seed=settings["seed"],
    )
    sizes = {
        "training": int(len(data.X_train)),
        "reference": int(len(data.X_reference)),
        "current": int(len(data.X_current)),
    }
    report = build_report(settings, sizes, drift_df, metrics_df, data.shifted_features)
    return {
        "settings": dict(settings),
        "data": data,
        "drift_df": drift_df,
        "metrics_df": metrics_df,
        "comparison_df": compare_reference_current(metrics_df),
        "sizes": sizes,
        "report": report,
        "json": report_to_json(report),
        "csv": drift_to_csv(drift_df),
    }


# ---------------------------------------------------------------- Tabs
def tab_dashboard(res: dict) -> None:
    drift_df = res["drift_df"]
    n_drift = int((drift_df["status"] == "DRIFT").sum())
    best = res["metrics_df"][res["metrics_df"]["dataset"] == "Current"]
    cols = st.columns(4)
    with cols[0]:
        metric_card("Features drifted", f"{n_drift} / {len(drift_df)}", "PSI ≥ threshold")
    with cols[1]:
        metric_card("Max PSI", f"{drift_df['psi'].max():.3f}", "highest single feature")
    with cols[2]:
        metric_card(
            "Best F1 (Current)", f"{best['f1'].max():.3f}", "across both models"
        )
    with cols[3]:
        s = res["sizes"]
        metric_card("Samples", f"{sum(s.values()):,}", f"train {s['training']:,} · ref {s['reference']:,} · cur {s['current']:,}")
    st.write("")
    st.info(
        "Drift does not always reduce model metrics. Compare Reference and Current "
        "results on the Model Performance tab before drawing conclusions."
    )
    st.caption(SYNTHETIC_WARNING)


def tab_performance(res: dict) -> None:
    metrics_df = res["metrics_df"]
    comp = res["comparison_df"]
    st.subheader("Metrics by model and dataset")
    st.dataframe(metrics_df.round(4), width="stretch", hide_index=True)
    st.subheader("Reference vs Current")
    st.dataframe(comp.round(4), width="stretch", hide_index=True)

    models = list(metrics_df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(11, 3.8), sharey=True)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes)
    x = np.arange(len(METRIC_NAMES))
    width = 0.38
    for ax, model in zip(axes, models):
        sub = metrics_df[metrics_df["model"] == model].set_index("dataset")
        ax.bar(x - width / 2, [sub.loc["Reference", m] for m in METRIC_NAMES], width, label="Reference", color=BLUE)
        ax.bar(x + width / 2, [sub.loc["Current", m] for m in METRIC_NAMES], width, label="Current", color=PURPLE)
        ax.set_xticks(x)
        ax.set_xticklabels(METRIC_NAMES)
        ax.set_ylim(0, 1.05)
        style_axes(ax, model, "Metric", "Score (0–1)")
        style_legend(ax)
    st.pyplot(fig)
    plt.close(fig)


def tab_drift(res: dict) -> None:
    drift_df = res["drift_df"]
    threshold = res["settings"]["psi_threshold"]
    st.subheader("Feature drift (PSI)")
    st.caption(PSI_DISCLAIMER)
    st.dataframe(
        drift_df.style.map(
            lambda v: f"color: {RED}; font-weight: bold" if v == "DRIFT" else (f"color: {GREEN}; font-weight: bold" if v == "OK" else ""),
            subset=["status"],
        ),
        width="stretch",
        hide_index=True,
    )
    fig, ax = new_fig((9, 4))
    colors = [RED if s == "DRIFT" else GREEN for s in drift_df["status"]]
    ax.bar(drift_df["feature"], drift_df["psi"], color=colors)
    ax.axhline(threshold, color=PURPLE, linestyle="--", label=f"Threshold = {threshold:.2f}")
    ax.set_xticks(range(len(drift_df)))
    ax.set_xticklabels(drift_df["feature"], rotation=45, ha="right")
    style_axes(ax, "PSI per feature", "Feature", "PSI")
    style_legend(ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def tab_explorer(res: dict) -> None:
    data = res["data"]
    feature = st.selectbox("Feature", data.feature_names, key="explorer_feature")
    bins = st.slider("Histogram bins", 5, 80, 30, key="explorer_bins")
    ref = data.X_reference[feature].to_numpy()
    cur = data.X_current[feature].to_numpy()
    lo = float(min(ref.min(), cur.min()))
    hi = float(max(ref.max(), cur.max()))
    if lo == hi:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, int(bins) + 1)
    fig, ax = new_fig((9, 4))
    ax.hist(ref, bins=edges, alpha=0.65, color=BLUE, label="Reference")
    ax.hist(cur, bins=edges, alpha=0.65, color=PURPLE, label="Current")
    style_axes(ax, f"Distribution of {feature}", feature, "Count")
    style_legend(ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    row = res["drift_df"][res["drift_df"]["feature"] == feature].iloc[0]
    st.write(f"PSI: **{row['psi']:.4f}** · Status: **{row['status']}**")
    if feature in data.shifted_features:
        st.caption("This feature was intentionally shifted in the Current dataset.")


def tab_report(res: dict) -> None:
    report = res["report"]
    st.subheader("Experiment report")
    st.warning(report["warning"])
    st.write(f"Generated (UTC): `{report['generated_at_utc']}`")
    st.write(f"Random seed: `{report['random_seed']}`")
    st.write("Dataset sizes:", report["dataset_sizes"])
    st.write("Shifted features:", report["shifted_features"] or "none")
    summary = report["summary"]
    st.write(
        f"{summary['features_drifted']} of {summary['features_total']} features flagged "
        f"as DRIFT (max PSI {summary['max_psi']:.3f})."
    )
    with st.expander("Settings"):
        st.json(report["settings"])
    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇ Download JSON report",
        data=res["json"],
        file_name="ai_driftlab_report.json",
        mime="application/json",
    )
    c2.download_button(
        "⬇ Download feature drift CSV",
        data=res["csv"],
        file_name="ai_driftlab_feature_drift.csv",
        mime="text/csv",
    )
    with st.expander("Raw JSON"):
        st.code(res["json"], language="json")


# ---------------------------------------------------------------- Main
def main() -> None:
    st.markdown(
        '<div class="hero"><h1>🧪 AI-DriftLab</h1>'
        "<p>Interactive laboratory for data drift detection and its impact on model performance.</p></div>",
        unsafe_allow_html=True,
    )
    settings = sidebar_settings()

    if st.sidebar.button("▶ Run Experiment", type="primary", width="stretch"):
        try:
            with st.spinner("Running experiment…"):
                st.session_state["result"] = run_experiment(settings)
                st.session_state.pop("error", None)
        except (DataGenerationError, DriftError, EvaluationError, ReportError) as exc:
            st.session_state["error"] = str(exc)
        except Exception:  # noqa: BLE001 - show a friendly message, no traceback
            st.session_state["error"] = (
                "Something unexpected went wrong while running the experiment. "
                "Please adjust the settings and try again."
            )

    if st.session_state.get("error"):
        st.error(st.session_state["error"])

    res = st.session_state.get("result")
    if res is None:
        st.info("No experiment yet. Choose settings in the sidebar and click **Run Experiment**.")
        return
    if res["settings"] != settings:
        st.caption("Sidebar settings changed since the last run — click Run Experiment to update results.")

    t1, t2, t3, t4, t5 = st.tabs(
        ["Dashboard", "Model Performance", "Data Drift", "Feature Explorer", "Experiment Report"]
    )
    with t1:
        tab_dashboard(res)
    with t2:
        tab_performance(res)
    with t3:
        tab_drift(res)
    with t4:
        tab_explorer(res)
    with t5:
        tab_report(res)


main()
