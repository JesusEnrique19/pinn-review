# Gate G1 — Decisión sobre la consulta y auditoría de etiquetas

## Situación

| Consulta | Registros | Recall seed | Recall hold-out | Precisión |
|---|---|---|---|---|
| v1 | 20 646 | 1.000 | 0.926 | 0.655 |
| **v3 (candidata)** | 19 158 | 1.000 | **0.988** | ≈ 0.69 |
| v3 solo nombres de método (Bloque A) | 11 497 | 0.778 | 0.901 | 0.835 |

Las cifras usan etiquetas del LLM pendientes de tu auditoría. El corpus v3 ya está descargado y
deduplicado: 18 111 papers.

**Ninguna consulta booleana alcanza a la vez recall ≥ 0.95 y precisión ≥ 0.80.** El ruido que queda
es semántico: "physics-informed" usado sin ninguna ecuación diferencial, regresiones sobre resultados
de simulación, arquitecturas inspiradas en EDPs para imágenes o grafos. Las palabras clave no
distinguen esos casos. Como el corpus bibliométrico (Tier 1) no se criba a mano, ese ~31 % de ruido
entraría en los mapas del campo.

## Decisión 1: cómo controlar la precisión del Tier 1

| Opción | Qué es | Coste | Riesgo ante revisores |
|---|---|---|---|
| **A (recomendada)** | Consulta v3 + **clasificador de relevancia** supervisado (embeddings de título y abstract + regresión logística) entrenado con etiquetas verificadas por humanos, con objetivo de precisión ≥ 0.85 y recall ≥ 0.95 medidos en un conjunto de prueba independiente. Análisis de sensibilidad con el subconjunto del Bloque A. | ~400 etiquetas más (LLM + auditoría humana) e instalar el modelo de embeddings (CPU, ~30 min de cómputo) | Bajo: es la práctica estándar en bibliometría a gran escala |
| B | Consulta v3 tal cual; se reporta la precisión ≈ 0.69 y se repiten los análisis con el Bloque A como sensibilidad | Ninguno adicional | Medio: un 31 % de ruido en el Tier 1 es criticable |
| C | Solo el Bloque A | Ninguno | Alto: pierde el 10 % del recall y los solvers neuronales sin la marca PINN (DGM, Deep Ritz…) |

## Decisión 2: auditoría (sigue pendiente)

`data/processed/g1/audit_v1.csv`, con 124 registros: rellena `human_label` (`y`/`n`). Es
imprescindible con cualquiera de las opciones, porque mide si las etiquetas del LLM son fiables. Con la
opción A, las etiquetas auditadas son además los datos de entrenamiento del clasificador.

Criterio IC1: el trabajo usa redes neuronales para **resolver, aprender o identificar ecuaciones
diferenciales** o sus operadores solución. "Physics-informed" sin ecuación diferencial: `n`.
