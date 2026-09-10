# Documentation — Telco Churn Scoring

Full project guide: from the business story to technical detail.

---

## Start here

| Document | Audience | Covers |
|----------|----------|--------|
| [01 — Overview](01_overview.md) | Everyone | Problem, pitch, scope, system map |
| [02 — Methodology](02_methodology.md) | Technical | Data, features, validation, models, survival |
| [03 — Results](03_results.md) | Technical / business | Metrics, lift, costs, survival, artifacts |
| [04 — Business playbook](04_business_playbook.md) | Business | Segments, Client A vs B, how to act |
| [05 — Developer guide](05_developer_guide.md) | Dev / ML | Setup, CI, structure, commands |

### Quick reads

- [One-pager](results_one_pager.md) — 1-page summary
- [2-minute demo](demo_2min.md) — speaking script + `python3 scripts/demo_2min.py`
- [Client A vs B](example_client_a_vs_b.md) — concrete commercial decision

### Project brief

- [`Enunciado_Proyecto_ML.pdf`](Enunciado_Proyecto_ML.pdf)

---

## In one sentence

> Most sales teams call the wrong customers.  
> This project turns telecom data into a **sales queue**: retain, upsell, or investigate — where the **top 20%** captures ~**51%** of churners (**2.5×** vs random).

---

## System diagram

```text
Telco data (7k customers, 26.5% churn)
        │
        ▼
 Feature engineering + train/val/test split
        │
        ├──────────────┬──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
   Case 1           Case 2         Case 3         Case 4
 Classification   Upsell $      Anomalies       Survival
 (churn score)    potential     IsolationForest  KM + Cox
        │              │              │              │
        └──────────────┴──────────────┴──────────────┘
                               │
                               ▼
                    Unified commercial score
                    (operational playbooks)
```

---

## Jump by question

| Question | Document |
|----------|----------|
| What problem does this solve? | [01 — Overview](01_overview.md) |
| Why not accuracy? | [01 — Overview](01_overview.md) · [02 — Methodology](02_methodology.md) |
| How do we validate without leakage? | [02 — Methodology](02_methodology.md) |
| What lift do we get? | [03 — Results](03_results.md) |
| What does survival mean here? | [02 — Methodology](02_methodology.md) · [03 — Results](03_results.md) |
| What do I do with a given customer? | [04 — Business playbook](04_business_playbook.md) |
| How do I run it / CI? | [05 — Developer guide](05_developer_guide.md) |
