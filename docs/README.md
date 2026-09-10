# Documentación — Telco Churn Scoring

Guía completa del proyecto: de la historia de negocio al detalle técnico.

---

## Empieza aquí

| Documento | Para quién | Qué cubre |
|-----------|------------|-----------|
| [01 — Overview](01_overview.md) | Todos | Problema, gancho, alcance, mapa del sistema |
| [02 — Methodology](02_methodology.md) | Técnico | Datos, features, validación, modelos, survival |
| [03 — Results](03_results.md) | Técnico / negocio | Métricas, lift, costes, survival, artefactos |
| [04 — Business playbook](04_business_playbook.md) | Negocio | Segmentos, Cliente A vs B, cómo actuar |
| [05 — Developer guide](05_developer_guide.md) | Dev / ML | Setup, CI, estructura, comandos |

### Lecturas rápidas

- [One-pager](results_one_pager.md) — resumen de 1 página
- [Demo 2 minutos](demo_2min.md) — guion oral + `python3 scripts/demo_2min.py`
- [Ejemplo A vs B](example_client_a_vs_b.md) — decisión comercial concreta

### Brief del proyecto

- [`Enunciado_Proyecto_ML.pdf`](Enunciado_Proyecto_ML.pdf)

---

## En una frase

> Most sales teams call the wrong customers.  
> Este proyecto convierte datos telecom en una **cola comercial**: retener, upsell o investigar — donde el **top 20%** captura ~**51%** de churners (**2.5×** vs aleatorio).

---

## Diagrama del sistema

```text
Datos Telco (7k clientes, 26.5% churn)
        │
        ▼
 Feature engineering + split train/val/test
        │
        ├──────────────┬──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
   Case 1           Case 2         Case 3         Case 4
 Clasificación    Potencial     Anomalías       Survival
 (churn score)     upsell €     IsolationForest  KM + Cox
        │              │              │              │
        └──────────────┴──────────────┴──────────────┘
                               │
                               ▼
                    Score comercial unificado
                    (playbooks operativos)
```

---

## Navegación rápida por pregunta

| Pregunta | Documento |
|----------|-----------|
| ¿Qué problema resuelve? | [01 — Overview](01_overview.md) |
| ¿Por qué no usamos accuracy? | [01 — Overview](01_overview.md) · [02 — Methodology](02_methodology.md) |
| ¿Cómo se valida sin leakage? | [02 — Methodology](02_methodology.md) |
| ¿Qué lift tenemos? | [03 — Results](03_results.md) |
| ¿Qué es survival aquí? | [02 — Methodology](02_methodology.md) · [03 — Results](03_results.md) |
| ¿Qué hago con un cliente concreto? | [04 — Business playbook](04_business_playbook.md) |
| ¿Cómo lo corro / CI? | [05 — Developer guide](05_developer_guide.md) |
