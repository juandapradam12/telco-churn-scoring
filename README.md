# Telco Churn Scoring

[![CI](https://github.com/juandapradam12/telco-churn-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/juandapradam12/telco-churn-scoring/actions/workflows/ci.yml)

**Stop calling the wrong customers.**

Most sales teams waste retention budget on random outreach.  
This project turns telecom churn data into a **ranked sales playbook**—who to retain, who to upsell, and who needs a billing check—so limited visit capacity actually hits risk.

> **Result:** contacting the **top 20%** of the ranked list captures ~**51%** of churners  
> (**2.5×** better than random).

**Author:** Juan Prada · Dataset: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## Why this matters

| Random calling | This ranking |
|----------------|--------------|
| Same effort for everyone | Effort follows risk + value + timing |
| Accuracy looks fine (~73%) while missing churners | Optimized for **recall, lift, and cost** |
| “Will they churn?” only | Also answers **when** and **what to do** |

With **26.5% churn**, a naive “always No Churn” model looks successful and still saves nobody.  
This repo is built for the real constraint: **you cannot visit everyone**.

---

## Proof in one glance

<p align="center">
  <img src="output/figures/lift_gains.png" alt="Lift and gains curves" width="900" />
</p>

| If you contact… | Churners captured | vs random |
|-----------------|------------------:|----------:|
| Top 10% | ~28% | **2.85×** |
| Top 20% | ~51% | **2.55×** |
| Top 30% | ~69% | **2.30×** |

Best classifier (holdout): **RandomForest** — F1 **0.66** · ROC-AUC **0.86** · PR-AUC **0.68** · ECE **0.016**

Survival adds timing: Cox concordance ~**0.83**; month-to-month contracts carry ~**9×** higher hazard.

---

## From score to action (Client A vs B)

| | **Client A** `3750-CKVKH` | **Client B** `9560-BBZXK` |
|--|--|--|
| Profile | New fiber, month-to-month | 36 months, two-year contract |
| 12-month churn risk | **~52%** | **~2%** |
| Upsell potential | ~€26 / month | ~€52 / month |
| Playbook | **Retain_HighValue** | **Grow_Upsell** |
| Reality in data | Did churn | Did not churn |

**A = put out the fire.**  
**B = don’t waste retention budget—sell more.**

That is the product: not a leaderboard of F1, a **queue your sales team can execute**.

<p align="center">
  <img src="output/figures/unified_commercial_scoring.png" alt="Unified commercial score distribution" width="900" />
</p>

---

## What the system does

```text
Telco customers
      │
      ├─ Case 1  Classification   → calibrated churn risk
      ├─ Case 2  Regression       → upsell € potential
      ├─ Case 3  Anomalies        → billing rarity alerts
      └─ Case 4  Survival (KM/Cox)→ risk within 6/12/24 months
                    │
                    ▼
         Unified commercial score + playbooks
         Retain_HighValue · Grow_Upsell · Investigate_Billing · …
```

Built with honest ML hygiene:
- train / val / test (no tuning on test)
- probability calibration + cost-based thresholds
- lift/gains for capacity-aware prioritization
- CI on every push (`pytest` + pipeline smoke)

---

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

2-minute terminal pitch:

```bash
python3 scripts/demo_2min.py
```

Narrative notebook:

```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

---

## Go deeper

| Want… | Open |
|-------|------|
| Full docs hub | [`docs/README.md`](docs/README.md) |
| Methods & validation | [`docs/02_methodology.md`](docs/02_methodology.md) |
| Metrics & artifacts | [`docs/03_results.md`](docs/03_results.md) |
| Sales playbooks | [`docs/04_business_playbook.md`](docs/04_business_playbook.md) |
| Setup / CI / structure | [`docs/05_developer_guide.md`](docs/05_developer_guide.md) |

---

## Suggested GitHub About blurb

```text
Most sales teams call the wrong customers. This project turns telco data into a ranked playbook—retain, upsell, or investigate—where the top 20% captures ~51% of churners (2.5× random).
```
