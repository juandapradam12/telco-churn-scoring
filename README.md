# Telco Churn Scoring — Deep Dive

[![CI](https://github.com/juandapradam12/telco-churn-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/juandapradam12/telco-churn-scoring/actions/workflows/ci.yml)

**Author:** Juan Prada

> Most sales teams call the wrong customers.  
> This project turns telecom data into a **sales queue** (retain / upsell / investigate): the **top 20%** captures ~**51%** of churners (**2.5×** vs random).

---

## Documentation

**Start here:** [`docs/README.md`](docs/README.md)

| Doc | Contents |
|-----|----------|
| [Overview](docs/01_overview.md) | Problem, pitch, system map |
| [Methodology](docs/02_methodology.md) | Features, validation, models, survival |
| [Results](docs/03_results.md) | Metrics, lift, costs, artifacts |
| [Business playbook](docs/04_business_playbook.md) | Segments + Client A vs B |
| [Developer guide](docs/05_developer_guide.md) | Setup, CI, repo structure |

Quick reads: [one-pager](docs/results_one_pager.md) · [2-min demo](docs/demo_2min.md) · [A vs B](docs/example_client_a_vs_b.md)

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

```bash
python3 -m pytest -q tests/
python3 scripts/demo_2min.py
jupyter notebook notebooks/churn_analysis.ipynb
```

---

## What the system includes

| Layer | Output |
|-------|--------|
| Calibrated classification | Churn risk + tiers |
| Lift / cost thresholds | Prioritization under visit capacity |
| Survival (KM + Cox) | Urgency at 6/12/24 months |
| Upsell potential + anomalies | Growth and billing alerts |
| Unified score | Operational playbooks |

Details: [`docs/01_overview.md`](docs/01_overview.md)
