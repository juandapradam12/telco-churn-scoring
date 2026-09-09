# Demo de 2 minutos — Telco Churn Scoring

Script oral + comandos. Objetivo: contar el caso sin entrar en todo el pipeline.

---

## 0:00–0:20 · Gancho

> Most sales teams call the wrong customers.
> Aqui el ranking no es “un modelo mas”: es una cola comercial.
> **Resultado:** contactar el top 20% captura ~51% de los churners (**2.5×** vs aleatorio).

Abrir: `README.md` (seccion *The story*) o la About del repo.

---

## 0:20–0:50 · Planteamiento

- Dataset telecom, **26.5% churn** → accuracy engaña.
- Pregunta de negocio: ¿a quién visitamos primero?
- No basta *si* se va: hace falta *cuándo* y *qué hacer*.

Mostrar: `output/figures/lift_gains.png`

Frase clave: “El lift convierte ML en capacidad de visitas.”

---

## 0:50–1:20 · Como se decide (Cliente A vs B)

Abrir: `docs/example_client_a_vs_b.md`

| | A `3750-CKVKH` | B `9560-BBZXK` |
|--|--|--|
| Riesgo 12m | ~52% | ~2% |
| Playbook | Retain_HighValue | Grow_Upsell |

> **A = apaga incendio.** **B = vende mas.**

Opcional: `output/figures/survival_kaplan_meier.png` (mes a mes vs contrato largo).

---

## 1:20–1:50 · Que hay debajo (sin deep dive)

Una frase por capa:
1. Clasificacion calibrada + umbral por costes
2. Survival (urgencia temporal)
3. Upsell + anomalias
4. Score unificado → playbooks

Mostrar: `output/reports/unified_commercial_scoring.csv` (top filas)  
o `output/figures/unified_commercial_scoring.png`

---

## 1:50–2:00 · Cierre

> No es un leaderboard de F1: es un sistema que dice **a quién llamar y para qué**.
> Notebook: `notebooks/churn_analysis.ipynb` · Pipeline: `python3 main.py`

---

## Comando rapido (opcional en vivo)

```bash
python3 scripts/demo_2min.py
```

Imprime lift top 20%, A vs B y el gancho en consola.
