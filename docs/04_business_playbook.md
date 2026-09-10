# 04 — Business playbook

Cómo se usa el sistema en operación comercial.

---

## De scores a acciones

El modelo no “decide solo”. Produce señales que se traducen en playbooks:

| Segmento | Señales típicas | Acción |
|----------|-----------------|--------|
| `Retain_HighValue` | Alto churn + alto potencial | Retención urgente + oferta personalizada |
| `Retain_InvestigateBill` | Alto churn + anomalía | Retención + revisión de facturación |
| `Retain_Urgent` | Alto churn | Visita urgente de retención |
| `Nurture_Upsell` | Churn medio + alto potencial | Contacto proactivo + ampliación |
| `Grow_Upsell` | Bajo churn + alto potencial | Campaña de upsell / cross-sell |
| `Investigate_Billing` | Anomalía sin churn alto | Revisión de facturación / condiciones |
| `Maintain` | Bajo riesgo / bajo potencial | Comunicación periódica |

Archivo operativo: `output/reports/unified_commercial_scoring.csv`

---

## Ejemplo concreto: Cliente A vs Cliente B

| | **A — `3750-CKVKH`** | **B — `9560-BBZXK`** |
|--|--|--|
| Perfil | 2 meses, fiber, mes a mes | 36 meses, contrato 2 años |
| Churn score | ~0.61 (High) | ~0.08 (Low) |
| P(churn en 12m) | **~52%** | **~2%** |
| Potencial upsell | ~€26/mes | ~€52/mes |
| Segmento | **Retain_HighValue** | **Grow_Upsell** |
| Realidad en datos | Churn = Yes | Churn = No |

### Lectura comercial

- **A** = apaga el incendio (y salva valor): retención ya.
- **B** = no gastes retención: empuja crecimiento.

Más detalle: [example_client_a_vs_b.md](example_client_a_vs_b.md)

---

## Cómo priorizar con capacidad limitada

1. Ordenar por `commercial_priority_score` (o por `churn_prob_within_12m` si solo importa urgencia).
2. Tomar el top según capacidad de visitas del equipo.
3. Aplicar el playbook del `commercial_segment`.

Regla práctica del lift:

> Si solo puedes contactar el **20%**, el ranking captura ~**51%** de churners (**2.5×** vs aleatorio).

---

## Qué mirar en una reunión de negocio (5 min)

1. `lift_gains.png` — ¿el ranking vale la pena?
2. Top de `unified_commercial_scoring.csv` — ¿quién entra hoy?
3. Cliente A vs B — ¿se entiende la lógica de acción?
4. Kaplan–Meier por contrato — ¿por qué mes a mes es urgente?

Guion oral: [demo_2min.md](demo_2min.md)

---

## Siguiente lectura

→ [05 — Developer guide](05_developer_guide.md)
