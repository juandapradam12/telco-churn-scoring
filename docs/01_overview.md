# 01 — Overview

## The problem

A field sales team cannot visit every customer. Calling at random:

- wastes visits on low-risk customers,
- arrives too late for those about to leave,
- fails to separate retention vs upsell vs billing issues.

The [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) dataset has **7,043 customers** and **~26.5% churn**.  
A naive model (“always No Churn”) gets ~73% accuracy **without detecting anyone**. Accuracy is the wrong metric.

## The pitch (problem → result)

**Problem:** you cannot call everyone.  
**Result:** an actionable ranking where contacting the **top 20%** captures ~**51%** of churners (**2.5×** vs random).

## What this project builds

This is not “just a classifier”. It is a **commercial prioritization system** with several layers:

| Layer | Question it answers | Output |
|-------|---------------------|--------|
| **Case 1 — Classification** | Will they churn? | Calibrated `churn_score` + tiers |
| **Case 2 — Regression** | How much more could they spend? | Monthly upsell potential + priority |
| **Case 3 — Anomalies** | Is billing behavior unusual? | `anomaly_score` |
| **Case 4 — Survival** | When might they churn? | P(churn within 6/12/24 months) |
| **Unified** | What should I do now? | Segment + recommended action |

## Design principles

1. **Honest validation:** train / val / test. Tuning on train, calibration/thresholds on val, metrics on test.
2. **Useful probabilities:** calibration (sigmoid) + Brier/ECE.
3. **Business decisions:** cost-based thresholds + lift/gains (not F1 alone).
4. **Timing:** survival adds urgency, not only static risk.
5. **Action:** the unified score turns ML into playbooks (`Retain_HighValue`, `Grow_Upsell`, …).

## Next

→ [02 — Methodology](02_methodology.md)
