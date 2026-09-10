# 05 — Developer guide

## Requirements

- Python 3.11+ (3.12 also works in this environment)
- Dependencies in `requirements.txt`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python3 main.py
```

Generates models, figures, and reports under `output/`.

## Notebook

```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

The notebook tells the same story as the pipeline (EDA → models → lift → cases → unified → survival).

## 2-minute demo

```bash
python3 scripts/demo_2min.py
```

Script: [`demo_2min.md`](demo_2min.md)

## Tests and CI

```bash
python3 -m pytest -q tests/
python3 scripts/ci_smoke.py
```

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR:

1. Unit tests
2. Pipeline smoke (accelerated critical path)
3. Demo script

Badge is shown in the root README.

## Repo structure

```text
telco-churn-scoring/
├── docs/                      # Documentation (you are here)
├── data/telco_churn.csv
├── src/
│   ├── data/loader.py
│   ├── features/engineering.py
│   ├── models/
│   │   ├── train.py           # calibration, costs, scoring
│   │   ├── tuning.py          # RandomizedSearchCV
│   │   └── lift.py            # lift / gains
│   ├── cases/
│   │   ├── commercial_potential.py
│   │   ├── anomaly_detection.py
│   │   ├── survival_analysis.py
│   │   └── unified_scoring.py
│   └── visualization/plots.py
├── notebooks/churn_analysis.ipynb
├── scripts/
│   ├── demo_2min.py
│   └── ci_smoke.py
├── tests/
├── output/{models,figures,reports}/
├── main.py
└── requirements.txt
```

## Output artifacts

| Path | Contents |
|------|----------|
| `output/reports/model_comparison.csv` | Case 1 metrics |
| `output/reports/lift_table_*.csv` | Lift by decile |
| `output/reports/churn_scoring.csv` | Churn ranking |
| `output/reports/case2_*.csv` | Upsell potential |
| `output/reports/case3_*.csv` | Anomalies |
| `output/reports/survival_*.csv` | Survival / Cox / time risk |
| `output/reports/unified_commercial_scoring.csv` | Commercial queue |
| `output/figures/` | Plots |
| `output/models/` | Base + calibrated `.pkl` files |

## Extending the project

| Idea | Where to change |
|------|-----------------|
| Unified-score weights | `src/cases/unified_scoring.py` |
| Cost matrix | `tune_threshold_cost` in `src/models/train.py` |
| Survival horizons | `build_survival_scoring(..., horizons=...)` |
| Add tests | `tests/test_core.py` |

## Related docs

- Index: [docs README](README.md)
- Methodology: [02_methodology.md](02_methodology.md)
- Results: [03_results.md](03_results.md)
