# RQ1 findings (draft for the authors; numbers from `data/processed/snapshots/rq1_v3.json`)

Interpretation is provisional until Gate G2. All statements hold in both robustness subsets unless noted.

1. **Growth.** 53 papers (2017) → 5 540 (2025). Year-on-year growth has stayed at ~60 % for 2023–2025;
   exponential doubling time 1.41 years (95 % CI 1.36–1.43). Gompertz has the lowest AIC, but the margins
   (ΔAIC 3.2 logistic, 4.4 exponential, from 9 points) are weak evidence, and 2025 may still be incompletely
   indexed. **Conclusion: sustained rapid growth; saturation is not supported by the data.**
2. **Sources (Bradford).** 2 642 sources. With preprints included (21 % of papers; arXiv alone 15 %), zones of
   14 / 241 / 2 387 sources give non-constant multipliers (17.2, 9.9): preprint servers inflate the core.
   **Journals only: 53 / 332 / 2 245, multipliers 6.3 and 6.8 — consistent with Bradford's law.** Core journals:
   J. Comput. Phys., CMAME, Physics of Fluids, LNCS, NeurIPS, EAAI.
3. **Authors (Lotka).** 30 387 author identifiers; 73.7 % with a single paper (classic Lotka with α = 2 predicts
   60.8 %). Discrete power-law MLE α = 2.54 (x_min = 1), but the bootstrap KS test rejects it (p < 0.01) and a
   lognormal fits better (normalised LLR R = −8.1, p < 10⁻¹⁵). The intersection subset gives α = 2.95 with
   x_min = 3 (p = 0.06). **Conclusion: Lotka-type concentration with a lighter-than-power-law tail.**
4. **Countries.** 86.7 % of papers have affiliation countries. China 35.2 %, United States 28.5 %, Germany 6.5 %,
   United Kingdom 5.7 %, India 5.4 %, Saudi Arabia 4.2 % (full counting). China overtook the United States in
   2023 and doubled its yearly output by 2025. International co-authorship: 27.3 % of papers.

Caveats: author disambiguation relies on OpenAlex author ids; country coverage is 86.7 %; 2025 counts may rise
with indexing lag.
