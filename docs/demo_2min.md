# 2-minute demo — Telco Churn Scoring

Speaking script + commands. Goal: tell the story without walking the full pipeline.

---

## 0:00–0:20 · Hook

> Most sales teams call the wrong customers.  
> This ranking is not “one more model”: it is a sales queue.  
> **Result:** contacting the top 20% captures ~51% of churners (**2.5×** vs random).

Open: root `README.md` (pitch) or the repo About section.

---

## 0:20–0:50 · Problem framing

- Telecom dataset, **26.5% churn** → accuracy misleads.
- Business question: who do we visit first?
- *Whether* they leave is not enough: we also need *when* and *what to do*.

Show: `output/figures/lift_gains.png`

Key line: “Lift turns ML into visit capacity.”

---

## 0:50–1:20 · How decisions are made (Client A vs B)

Open: `docs/example_client_a_vs_b.md`

| | A `3750-CKVKH` | B `9560-BBZXK` |
|--|--|--|
| 12m risk | ~52% | ~2% |
| Playbook | Retain_HighValue | Grow_Upsell |

> **A = put out the fire.** **B = sell more.**

Optional: `output/figures/survival_kaplan_meier.png` (month-to-month vs longer contracts).

---

## 1:20–1:50 · What sits underneath (no deep dive)

One sentence per layer:
1. Calibrated classification + cost-based threshold
2. Survival (timing urgency)
3. Upsell + anomalies
4. Unified score → playbooks

Show: `output/reports/unified_commercial_scoring.csv` (top rows)  
or `output/figures/unified_commercial_scoring.png`

---

## 1:50–2:00 · Close

> This is not an F1 leaderboard: it is a system that says **who to call and why**.  
> Notebook: `notebooks/churn_analysis.ipynb` · Pipeline: `python3 main.py`

---

## Optional live command

```bash
python3 scripts/demo_2min.py
```

Prints lift top 20%, A vs B, and the pitch in the terminal.
