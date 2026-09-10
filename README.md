# Telco Churn Scoring — Deep Dive

[![CI](https://github.com/juandapradam12/telco-churn-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/juandapradam12/telco-churn-scoring/actions/workflows/ci.yml)

**Autor:** Juan Prada

> Most sales teams call the wrong customers.  
> Este proyecto convierte datos telecom en una **cola comercial** (retener / upsell / investigar): el **top 20%** captura ~**51%** de churners (**2.5×** vs aleatorio).

---

## Documentación

**Empieza por el índice:** [`docs/README.md`](docs/README.md)

| Doc | Contenido |
|-----|-----------|
| [Overview](docs/01_overview.md) | Problema, alcance, mapa del sistema |
| [Methodology](docs/02_methodology.md) | Features, validación, modelos, survival |
| [Results](docs/03_results.md) | Métricas, lift, costes, artefactos |
| [Business playbook](docs/04_business_playbook.md) | Segmentos + Cliente A vs B |
| [Developer guide](docs/05_developer_guide.md) | Setup, CI, estructura |

Lecturas rápidas: [one-pager](docs/results_one_pager.md) · [demo 2 min](docs/demo_2min.md) · [A vs B](docs/example_client_a_vs_b.md)

**GitHub About description:**
```text
Most sales teams call the wrong customers. This project turns telco data into a ranked playbook—retain, upsell, or investigate—where the top 20% captures ~51% of churners (2.5× random).
```

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

## Qué incluye el sistema

| Capa | Output |
|------|--------|
| Clasificación calibrada | Riesgo de churn + tiers |
| Lift / costes | Priorización con capacidad limitada |
| Survival (KM + Cox) | Urgencia a 6/12/24 meses |
| Potencial + anomalías | Upsell y alertas de facturación |
| Score unificado | Playbooks operativos |

Detalle: [`docs/01_overview.md`](docs/01_overview.md)
