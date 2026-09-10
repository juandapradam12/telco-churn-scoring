# 04 — Business playbook

How to use the system in commercial operations.

---

## From scores to actions

The model does not “decide alone”. It produces signals that map to playbooks:

| Segment | Typical signals | Action |
|---------|-----------------|--------|
| `Retain_HighValue` | High churn + high potential | Urgent retention + personalized offer |
| `Retain_InvestigateBill` | High churn + anomaly | Retention + billing review |
| `Retain_Urgent` | High churn | Urgent retention visit |
| `Nurture_Upsell` | Medium churn + high potential | Proactive contact + expansion offer |
| `Grow_Upsell` | Low churn + high potential | Upsell / cross-sell campaign |
| `Investigate_Billing` | Anomaly without high churn | Billing / special-conditions review |
| `Maintain` | Low risk / low potential | Periodic communication |

Operational file: `output/reports/unified_commercial_scoring.csv`

---

## Concrete example: Client A vs Client B

| | **A — `3750-CKVKH`** | **B — `9560-BBZXK`** |
|--|--|--|
| Profile | 2 months, fiber, month-to-month | 36 months, two-year contract |
| Churn score | ~0.61 (High) | ~0.08 (Low) |
| P(churn in 12m) | **~52%** | **~2%** |
| Upsell potential | ~€26/month | ~€52/month |
| Segment | **Retain_HighValue** | **Grow_Upsell** |
| Actual outcome | Churn = Yes | Churn = No |

### Commercial reading

- **A** = put out the fire (and save value): retain now.
- **B** = do not spend retention budget: push growth.

More detail: [example_client_a_vs_b.md](example_client_a_vs_b.md)

---

## How to prioritize with limited capacity

1. Sort by `commercial_priority_score` (or by `churn_prob_within_12m` if only urgency matters).
2. Take the top slice matching team visit capacity.
3. Apply the playbook from `commercial_segment`.

Practical lift rule:

> If you can only contact the **top 20%**, the ranking captures ~**51%** of churners (**2.5×** vs random).

---

## What to show in a 5-minute business meeting

1. `lift_gains.png` — is the ranking worth it?
2. Top of `unified_commercial_scoring.csv` — who enters today’s queue?
3. Client A vs B — is the action logic clear?
4. Kaplan–Meier by contract — why month-to-month is urgent?

Speaking script: [demo_2min.md](demo_2min.md)

---

## Next

→ [05 — Developer guide](05_developer_guide.md)
