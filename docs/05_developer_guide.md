# 05 — Developer guide

## Requisitos

- Python 3.11+ (3.12 también ok en este entorno)
- Dependencias en `requirements.txt`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar el pipeline completo

```bash
python3 main.py
```

Genera modelos, figuras y reportes en `output/`.

## Notebook

```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

El notebook narra la misma historia que el pipeline (EDA → modelos → lift → cases → unificado → survival).

## Demo rápida (2 min)

```bash
python3 scripts/demo_2min.py
```

Guion: [`demo_2min.md`](demo_2min.md)

## Tests y CI

```bash
python3 -m pytest -q tests/
python3 scripts/ci_smoke.py
```

GitHub Actions (`.github/workflows/ci.yml`) corre en cada push/PR:

1. Unit tests
2. Pipeline smoke (camino crítico acelerado)
3. Demo script

Badge en el README raíz.

## Estructura del repo

```text
telco-churn-scoring/
├── docs/                      # Documentación (estás aquí)
├── data/telco_churn.csv
├── src/
│   ├── data/loader.py
│   ├── features/engineering.py
│   ├── models/
│   │   ├── train.py           # calibración, costes, scoring
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

## Artefactos de salida

| Ruta | Contenido |
|------|-----------|
| `output/reports/model_comparison.csv` | Métricas Case 1 |
| `output/reports/lift_table_*.csv` | Lift por decil |
| `output/reports/churn_scoring.csv` | Ranking de churn |
| `output/reports/case2_*.csv` | Potencial upsell |
| `output/reports/case3_*.csv` | Anomalías |
| `output/reports/survival_*.csv` | Survival / Cox / riesgo temporal |
| `output/reports/unified_commercial_scoring.csv` | Cola comercial |
| `output/figures/` | Gráficos |
| `output/models/` | `.pkl` base + calibrados |

## Extender el proyecto

| Idea | Dónde tocar |
|------|-------------|
| Cambiar pesos del unificado | `src/cases/unified_scoring.py` |
| Matriz de costes | `tune_threshold_cost` en `src/models/train.py` |
| Horizontes survival | `build_survival_scoring(..., horizons=...)` |
| Añadir tests | `tests/test_core.py` |

## Documentación relacionada

- Índice: [README de docs](README.md)
- Metodología: [02_methodology.md](02_methodology.md)
- Resultados: [03_results.md](03_results.md)
