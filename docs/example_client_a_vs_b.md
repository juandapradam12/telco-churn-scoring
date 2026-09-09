# Ejemplo comercial: Cliente A vs Cliente B

Como se lee el scoring en una decision de fuerza de ventas.

---

## Descripcion del proyecto (GitHub / About)

**English (recomendada):**
```text
Turn telco customer data into an actionable sales playbook: who to retain, who to upsell, and who needs a billing check—ranked by calibrated risk and timing.
```

**Español (alternativa):**
```text
Convierte datos de clientes telecom en una cola comercial accionable: a quién retener, a quién hacer upsell y a quién revisar facturación, priorizado por riesgo calibrado y urgencia temporal.
```

---

## Los dos perfiles

| | **Cliente A — `3750-CKVKH`** | **Cliente B — `9560-BBZXK`** |
|--|--|--|
| Antiguedad | 2 meses | 36 meses |
| Contrato | Month-to-month | Two year |
| Internet | Fiber optic | (estable / bajo riesgo) |
| Factura mensual | ~€68 | — |
| **Churn score (clasificacion)** | **0.61** (High) | **0.08** (Low) |
| **P(churn en 12 meses)** survival | **51.6%** | **1.7%** |
| Potencial upsell | ~€26/mes | ~€52/mes |
| Segmento unificado | **Retain_HighValue** | **Grow_Upsell** |
| Que paso en realidad | Churn = Yes | Churn = No |

---

## Que implica para el equipo comercial

### Cliente A — retener YA
No basta con “tiene riesgo alto”. Survival dice que **en los proximos 12 meses tiene ~52% de probabilidad de irse**, y ademas es cliente nuevo en fibra mes a mes (el perfil de mayor hazard).

**Accion:** visita urgente + oferta de retencion personalizada (idealmente pasar a contrato anual) **antes** de que se complete el ciclo de fuga.
Es un caso `Retain_HighValue`: riesgo alto **y** potencial relevante (~€26).

### Cliente B — no gastar retencion; empujar crecimiento
Churn score bajo y survival a 12m ~**2%**: no es prioridad de retencion.
Pero tiene **alto potencial de upsell (~€52)**.

**Accion:** campana de ampliacion de servicios / upgrade. Segmento `Grow_Upsell`.
Gastar una visita de “rescate” aqui seria ineficiente.

---

## Lectura en una frase

> **A** es “apaga el incendio y salva valor”.  
> **B** es “no hay incendio: vende mas”.

Eso es lo que aporta combinar:
1. probabilidad de churn calibrada,
2. urgencia temporal (survival),
3. potencial comercial,
en un solo playbook.
