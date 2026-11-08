# Gate G0 — Checklist de aprobación del protocolo

Marca cada punto. Cuando todo esté marcado, el protocolo (`protocol/protocol.md`) se congela como v1.0
y se registra en OSF. Los cambios posteriores van a `protocol/deviations.md`.

## A. Decisiones científicas (tuyas, por política de IJACSA)

Aprobadas sin cambios por el autor el 2026-10-30. Protocolo congelado como v1.0.

- [x] **Preguntas de investigación RQ1–RQ5** (protocolo §2): aprobadas o con cambios anotados.
- [x] **Criterios de elegibilidad** (§5): en particular, incluir preprints de arXiv y artículos de NeurIPS, ICML e ICLR sin DOI.
- [x] **Seed set** (`protocol/seed_set.csv`, 54 papers en alcance + 8 fundacionales): revisa
      `data/processed/seed_set_resolved.csv` y anota los papers clave que falten o sobren.
      Es la vara con la que se medirá la búsqueda, así que debe fijarse **antes** de buscar.
- [x] **Codebook v1 y rúbrica de calidad Q1–Q8** (`protocol/codebook.yaml`): definiciones claras y sin categorías faltantes.
- [x] **Diseño del benchmark** (§9): 4 EDPs × 6 métodos × 5 semillas, reglas de igual presupuesto, hipótesis H5a y H5b.

## B. Datos operativos

- [ ] Autores, afiliaciones y ORCID (IJACSA los recomienda).
- [ ] Segundo codificador humano para el 20 % (colega o asesor).
- [ ] Cómputo: solo CPU (≈15–25 h de benchmark) o GPU gratuita (Kaggle o Colab).
- [ ] Manuscrito en LaTeX o en la plantilla Word de SAI.
- [ ] OpenAlex: ¿usar tu correo (`OPENALEX_MAILTO`) o una API key? Es opcional; sin ellos va más lento.

## C. Acciones que tienes que hacer tú

- [ ] Crear el repositorio en GitHub (privado por ahora) para que la CI corra; después hago el push.
- [ ] Crear la cuenta en OSF y enviar el registro. Yo preparo el texto; registrar es una acción tuya.
