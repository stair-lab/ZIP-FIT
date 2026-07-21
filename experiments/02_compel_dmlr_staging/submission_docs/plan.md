# DMLR Submission Plan — Compel

**TLDR:** Requirement-by-requirement mapping of the DMLR rules (fetched fresh from data.mlr.press on 2026-07-17) to the `paper_latex/DMLR_2026_Compel/` package, plus the improvement log vs. the Dec-2025 double-blind review PDF it was rebuilt from.

## DMLR requirements → how satisfied

| # | DMLR requirement | How satisfied |
|---|---|---|
| 1 | Official `dmlr2e.sty` template; violations = desk rejection | `00_dmlr_compel.tex`: `\documentclass[twoside,11pt]{article}` + `\usepackage{dmlr2e}` (files verbatim from github.com/JmlrOrg/dmlr-style-file) |
| 2 | Single-blind — authors NOT anonymized | Page 1 carries the six authors + Stanford affiliation + emails (was "Anonymous authors" in the review PDF) |
| 3 | No page limit; appendix after references | 20 pp; no appendix (the review PDF's "appendix" was an unmodified ICLR template placeholder — dropped) |
| 4 | Scope: primarily data-centric | Pure data-curation paper; abstract, contributions list, and keywords foreground pretraining-data quality |
| 5 | Broader Impact Statement mandatory | `\impact{...}` block: auditability upside + multilingual-filtering bias risk (the bilingual high-CR example) |
| 6 | Extended-submission (≥30% new) applies to prior conference/workshop publications | **N/A** — no prior publication; prior state is an unpublished double-blind manuscript + no arXiv record found. Disclose the review history on the form ⚠ |
| 7 | New dataset/benchmark ⇒ datasheet supplement | N/A — no new dataset/benchmark introduced; corpora (FineWeb, FineWeb-EDU, DCLM, C4, Twitter) and all 13 benchmarks pre-exist |
| 8 | Reproducibility support | Filtering code + distribution-analysis scripts: github.com/eobbad/Compel (`experiments/data_analysis/`); training configs fully specified in §6 |
| 9 | No concurrent archival submission | ⚠ Confirm the Dec-2025 double-blind review concluded before submitting |
| 10 | Reviewer test: claims ↔ evidence aligned | Every number transcribed 1:1 from the Dec-2025 PDF (cross-checked against its text layer); no new empirical claims added |

## Improvement log (DMLR version vs. Dec-2025 review PDF)

1. **New §4 "Theoretical Grounding: Compression, Entropy, and Learnable Structure"** — the Shannon/Kolmogorov/MDL account of why an interior compression-ratio band selects learnable text. Directly addresses the standing reviewer criticism that the method is "conceptually simple / lacks deeper theoretical grounding." New refs: Kolmogorov 1965, Rissanen 1978, Grünwald 2007, Li & Vitányi 2008, Cilibrasi & Vitányi 2005.
2. **New contributions list** at the end of §1, foregrounding the data-centric contributions (DMLR scope check).
3. **Citation repairs** (the review PDF had broken "(et al., 2024b)"-style citations and placeholder bib entries "arXiv:2401.xxxxx"): real entries for Ankner et al. 2024 (2405.20541), Maini et al. 2024 (WRAP, 2401.16380), Marion et al. 2023 (2309.04564); the erroneous duplicate "Xu et al." copy of the C4-documentation entry replaced by the intended Xu et al. 2021 *Detoxifying LMs risks marginalizing minority voices* (2104.06390); T5 deduplicated to one entry.
4. **Figure caption/panel mismatch fixed** (Figure 3 in the review PDF's numbering; Figure 2 in the DMLR version) — the review PDF's caption said "Top left: C4 … Bottom right: Twitter" while the panels actually show Twitter top-left, C4 top-right, FineWeb-EDU bottom-left, DCLM bottom-right. Caption now matches the panels (verified against the embedded plot titles/legends).
5. **Duplicated examples figure removed** — the review PDF showed the identical 3-column examples table twice (its Figs. 2 and 5); it now appears once (§8) and is referenced from §3.
6. **Leftover ICLR template placeholder appendix dropped** (review PDF p. 14 still contained the template's "Optionally include supplemental material…" text).
7. Grammar/register fixes ("experiments would necessary" → "would be necessary"; removed "novel", "seamlessly", "sadly"; "utilize"→"use") and one-sentence-per-line source formatting.
8. Broader Impact statement and keywords added (required by DMLR; absent in the review PDF).
9. Figures recovered losslessly from the PDF's embedded PNGs (original resolutions: pipeline 2048×444, histograms 3000×1800, KDE 2100×1200).
10. Flagged for internal review (not changed): the 8B learning rate 2×10⁻³ transcribed from the PDF looks high vs. LLaMA-3's ~3×10⁻⁴ — `% TODO(internal-review)` comment in `06_experiments.tex`; confirm with Elyas.
11. **Gains range corrected to "0.4–1.1 points"** in abstract/intro/discussion (Mega QA stage-2 fix): the review PDF said "0.5–1.1" but its own Table 1 shows DCLM's macro gain is 0.4 (46.6→47.0) — the corrected range is exactly what the table supports.

## Build verification (2026-07-17)

- `latexmk -pdf` + final `pdflatex`: **0 errors, 0 undefined citations, 0 undefined references, 0 overfull hboxes**; `00_dmlr_compel.pdf`, 20 pages.
- Visual QA on rendered pages: title block, theory section, colored results tables, and the CJK (Chinese) example all render correctly.

## Remaining human steps (Brando / Elyas)

1. Elyas: point to the canonical Compel paper repo; diff/merge this rebuild into it.
2. Confirm the Dec-2025 double-blind review is concluded (no concurrent submission).
3. Confirm author order/emails; ensure all six authors have OpenReview profiles.
4. Fill funding + conflicts; submit at https://openreview.net/group?id=DMLR.
