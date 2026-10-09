# AI-DriftLab

An interactive machine-learning laboratory for **detecting data drift** and
**measuring its impact on model performance**. Built with Python, Streamlit,
NumPy, Pandas, scikit-learn, Matplotlib and Pytest.

> **All data is synthetic.** Datasets are generated with
> `sklearn.datasets.make_classification`. Results illustrate the method and do
> not describe any real-world dataset or production model.

## Features

- Synthetic binary-classification data split into **Training / Reference / Current** sets.
- Controlled distribution shift applied **only to the Current set**
  (strength and number of shifted features are user-controlled).
- Per-feature **Population Stability Index (PSI)** with bins built from the
  Reference set and reused for Current; `OK` / `DRIFT` status from a
  user-defined threshold.
- **Logistic Regression** (with `StandardScaler` in a `Pipeline`) and
  **Random Forest**, trained on the Training set only and evaluated on both
  Reference and Current (Accuracy, Precision, Recall, F1).
- Dark-theme Streamlit dashboard: metric cards, Model Performance, Data Drift,
  Feature Explorer (histogram overlay) and Experiment Report tabs.
- Downloadable **JSON report** and **feature-drift CSV**.
- Pytest suite covering generation, PSI edge cases, models and reports.

## Architecture

```
UI (app.py)  ->  data_generator  ->  drift_detector
                               \->  model_evaluator
                                \-> report_generator
```

The UI only orchestrates; all computation lives in `src/` and is unit tested.

## Project structure

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard, session state, charts, downloads |
| `src/__init__.py` | Package marker and version |
| `src/data_generator.py` | Parameter validation, data generation, 3-way split, shift simulation |
| `src/drift_detector.py` | PSI calculation and per-feature OK/DRIFT table |
| `src/model_evaluator.py` | Model construction, training, metrics, Reference-vs-Current comparison |
| `src/report_generator.py` | JSON report and CSV export (UTC timestamp, synthetic-data warning) |
| `tests/conftest.py` | Makes the project root importable for pytest |
| `tests/test_*.py` | Unit tests for each module |
| `requirements.txt` | Dependencies (version ranges, not tight pins) |
| `.gitignore` | Excludes caches, virtual environments and secrets |
| `LICENSE` | MIT license |

## Install on Windows

Recommended: **Python 3.11 – 3.13** (the project was developed and tested on Python 3.13).

```powershell
# 1. Open PowerShell in the AI-DriftLab folder
python -m venv .venv

# 2. Activate the virtual environment
.venv\Scripts\Activate.ps1
# If scripts are blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# (Command Prompt users: .venv\Scripts\activate.bat)

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app

```powershell
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501), adjust the
sidebar settings and click **Run Experiment**.

## Run the tests

```powershell
python -m pytest -q
```

## Upload to GitHub

1. Create a new empty repository on github.com (no README/license, since they already exist).
2. In the project folder:

```powershell
git init
git add .
git commit -m "Initial commit: AI-DriftLab"
git branch -M main
git remote add origin https://github.com/<your-username>/AI-DriftLab.git
git push -u origin main
```

Make sure `app.py` is at the **root** of the repository, not inside an extra folder.

## Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with your GitHub account.
2. Click **Create app** and choose the repository and the `main` branch.
3. Set **Main file path** to `app.py`.
4. Click **Deploy**. Dependencies are installed from `requirements.txt`.

## Troubleshooting

- **`streamlit` is not recognized** – activate the virtual environment first, or run `python -m streamlit run app.py`.
- **PowerShell blocks `Activate.ps1`** – run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt.
- **`ModuleNotFoundError: src`** – run commands from the project root (the folder containing `app.py`).
- **Error message about fractions** – Training + Reference fractions must stay below 0.95 so the Current set is not empty.
- **Deploy fails on Streamlit Cloud** – check that `requirements.txt` and `app.py` are at the repository root.

## Limitations

- Synthetic data only; shift is a simple mean shift of selected features.
- PSI thresholds are **heuristics, not universal rules**. Small samples can show
  PSI "drift" even without any shift (noise), especially at the minimum dataset size.
- Drift does **not** always reduce model metrics; always compare Reference and Current results.
- PSI is univariate and does not detect changes in feature relationships or label drift.
- Models use default-style hyperparameters and are not tuned.

## License

MIT – see [LICENSE](LICENSE).
