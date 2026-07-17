# DMLR Submission Plan — ZIP-FIT

**TLDR:** Requirement-by-requirement mapping of the DMLR rules (fetched fresh from data.mlr.press on 2026-07-17: submissions.html, acceptance-criteria.html, reviewer-guidelines.html) to what the `paper_latex/DMLR_2026_ZipFit/` package does, plus the log of improvements made over the arXiv v2 source.

## DMLR requirements → how satisfied

| # | DMLR requirement (data.mlr.press) | How satisfied |
|---|---|---|
| 1 | Official DMLR LaTeX style (`dmlr2e.sty` from github.com/JmlrOrg/dmlr-style-file); format violations are desk-reject grounds | `00_dmlr_zipfit.tex` uses `\documentclass[twoside,11pt]{article}` + `\usepackage{dmlr2e}` with the verbatim style files from the official repo |
| 2 | Single-blind, open review — authors NOT anonymized | Full author names, Stanford affiliations, and emails on page 1 |
| 3 | No page limit; appendix AFTER references | 30 pp; `\appendix` + `\input{99_appendix}` placed after `\bibliography` |
| 4 | Scope: primarily data-centric | The paper *is* data selection (data-centric by construction); abstract/intro lead with data selection and data quality |
| 5 | Broader Impact Statement mandatory | `\impact{...}` block in the wrapper (dmlr2e's native macro) |
| 6 | Extended submissions: ≥30% new content vs. prior conference/workshop publication + disclose and link on the form | Prior pub = ICML 2025 DataWorld workshop poster ("for Code" subset). DMLR version adds AF experiments, efficiency, misalignment study, appendix ⇒ far beyond 30%. Links + change summary drafted in `openreview_checklist.md` |
| 7 | New dataset/benchmark ⇒ separate datasheet supplement PDF | N/A — no new dataset or benchmark is introduced; all corpora/benchmarks are pre-existing public ones. Stated explicitly on the form |
| 8 | Reproducibility: code/data availability encouraged (NeurIPS code guidelines, Pineau checklist) | Code: github.com/stair-lab/ZIP-FIT (public pip-installable repo); experiment scripts in `experiments/` |
| 9 | Concurrent archival submission prohibited (arXiv OK) | arXiv preprint OK. ⚠ Brando must confirm the ICLR OpenReview record (`4JBEpP6eRS`) is concluded/inactive before submitting |
| 10 | Reviewer two-question test: claims supported by evidence; interest to DMLR audience | Every number in the DMLR version traces to the arXiv v2 source; audience fit = data selection methods |

## Improvement log (DMLR version vs. arXiv v2 source)

1. Converted from ICLR-2025 format to the official DMLR `dmlr2e` layout (wrapper `00_dmlr_zipfit.tex`), de-anonymized, with keywords and the mandatory Broader Impact statement.
2. **Repaired a dangling citation** in Related Works (`07_conclusion.tex`): the text-diversity/compression sentence had no citation — now credits `\citet{shaib2024standardizing}` (Shaib et al., 2024, arXiv:2403.00553; entry added to `references_rylan.bib`).
3. Converted four bare `\citep{...}`-as-sentence-subject citations in the Compression related-work block to `\citet{...}` (house LaTeX rule: no orphan citations).
4. Fixed the widest appendix table (Performance/efficiency comparison, was 110 pt overfull) with a `\resizebox` to text width.
5. Review-note macros (`\rylan`, `\fix`, `\new`) suppressed via `\providecommand` no-ops.
6. All post-review (rebuttal-era) experiments from arXiv v2 retained: LESS comparison, QLoRA/4-bit quantized results, compression-algorithm ablations (appendix tables).

## Build verification (2026-07-17)

- `latexmk -pdf` + final `pdflatex` pass: **0 errors, 0 undefined citations, 0 undefined references**; output `00_dmlr_zipfit.pdf`, 30 pages.
- Remaining known-cosmetic: 8 small overfull hboxes inside appendix sample-dump tables (long unbreakable code/URL strings inherited from the published arXiv version).

## Remaining human steps (Brando)

1. Confirm ICLR OpenReview record `4JBEpP6eRS` is inactive (no concurrent archival review) — DMLR prohibits concurrent submissions.
2. Verify author email handles on page 1 and that every author has an OpenReview profile.
3. Fill funding + conflicts fields (see `openreview_checklist.md`).
4. Submit at https://openreview.net/group?id=DMLR and, once the forum exists, put its URL in `\def\openreview{...}` for the camera-ready.
