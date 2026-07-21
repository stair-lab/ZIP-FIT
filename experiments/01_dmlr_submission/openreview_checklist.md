# OpenReview Form Answers — ZIP-FIT → DMLR

**TLDR:** Drafted answers for every field of the DMLR OpenReview submission form (submit at https://openreview.net/group?id=DMLR), with the items only Brando can confirm marked ⚠.

## Title
ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment

## Abstract
Use the abstract from `paper_latex/DMLR_2026_ZipFit/01_abstract.tex` (the active, uncommented block) verbatim.

## Authors (order as in paper — all need OpenReview profiles ⚠)
1. Elyas Obbad (Stanford CS)
2. Iddah Mlauzi (Stanford CS)
3. Brando Miranda (Stanford CS)
4. Rylan Schaeffer (Stanford CS)
5. Kamal Obbad (Stanford Biophysics, School of Medicine)
6. Suhana Bedi (Stanford Biomedical Data Science, School of Medicine)
7. Sanmi Koyejo (Stanford CS)

⚠ Verify each author has an active OpenReview profile before submitting; DMLR requires co-author consent confirmation on the form.

## Data-centric scope statement (1–3 sentences for the form)
ZIP-FIT is a data-selection method: it chooses fine-tuning data by measuring compression-based alignment between candidate data and the target task distribution, with no embeddings or auxiliary models. The contribution is entirely on the data side of ML — what to train on, selected how, at what cost — evaluated by downstream model quality on Autoformalization and code generation.

## Prior publication disclosure (extended-submission rule)
- **Yes, this extends a workshop paper**: "ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment for Code," poster at the ICML 2025 workshop *DataWorld: Unifying data curation frameworks across domains*.
  Link: https://openreview.net/forum?id=WgZ9C4q9nH (ICML virtual page: https://icml.cc/virtual/2025/48702)
- **Changes vs. that version (≥30% new content — comfortably satisfied):** the workshop version covered only the code-generation subset. The DMLR version adds: the Autoformalization experiments and analysis; the interventional/misalignment study (§6); the efficiency and selection-cost analysis (§5); compression-algorithm ablations, LESS comparison, and QLoRA/4-bit results (appendix); plus the full appendix of selected-sample analyses.
- Also on record: arXiv:2410.18194 (allowed, non-archival) and an OpenReview record https://openreview.net/forum?id=4JBEpP6eRS. ✅ Confirmed by Brando (2026-07-21): ZIP-FIT is not under review anywhere — that record's process is concluded, so DMLR's no-concurrent-submission rule is satisfied.

## IRB / human subjects
N/A — no human subjects research; all datasets are pre-existing public corpora and benchmarks.

## New dataset/benchmark supplement
N/A — the paper introduces no new dataset or benchmark, so no datasheet supplementary PDF is required.

## Code availability
https://github.com/stair-lab/ZIP-FIT (public; pip-installable; experiment scripts under `experiments/`).

## Funding ⚠
To be filled by Brando (e.g., Stanford Trustworthy AI Research funding sources; any NSF/ONR/industry grants supporting the authors).

## Competing interests ⚠
To be filled by Brando (default: none beyond employment/affiliations listed).

## Conflicts of interest (DMLR: personal conflicts + institutional/domain conflicts, last 3 years)
- Domain conflicts (institutions, last 3 years): **stanford.edu** (all authors). ⚠ Add any author's second affiliation/internship hosts from the last 3 years (e.g., industry labs) — per author.
- Personal conflicts: advisor/advisee relations within the author list are internal and fine; name recent (≤3 y) external collaborators of any author who are DMLR action editors ⚠ (check the AE list at submission time).

## Keywords (form field)
data selection, data quality, gzip compression, normalized compression distance, task-aware fine-tuning, autoformalization, code generation, large language models

## After submission
- Fill `\def\openreview{https://openreview.net/forum?id=<assigned-id>}` in `00_dmlr_zipfit.tex` at camera-ready.
