# OpenReview Form Answers — Compel → DMLR

**TLDR:** Drafted answers for the DMLR OpenReview submission form (https://openreview.net/group?id=DMLR), with ⚠ marking the items only Brando/Elyas can confirm.

## Title
Curating High Quality Pretraining Data for Language Models via Compression Ratios

## Abstract
Use the abstract from `paper_latex/DMLR_2026_Compel/00_dmlr_compel.tex` verbatim (matches the Dec-2025 PDF abstract, except: the DMLR-scope closing sentence, and the gains range corrected to "0.4--1.1 points" to match Table 1).

## Authors (order from Brando's CV — ⚠ confirm with Elyas; all need OpenReview profiles)
1. Elyas Obbad (Stanford CS)
2. Brando Miranda (Stanford CS)
3. David L. W. Hall (Stanford CS / CRFM)
4. Rylan Schaeffer (Stanford CS)
5. Sanmi Koyejo (Stanford CS)
6. Percy Liang (Stanford CS)

## Data-centric scope statement (for the form)
Compel is a pretraining-data curation method: it filters web-scale corpora using a model-free compression-ratio band as an information-density signal, improving downstream accuracy on FineWeb, FineWeb-EDU, and DCLM at negligible cost. The contribution is entirely data-centric — a quality signal, its distributional analysis across corpora, its theoretical grounding, and its causal validation via pretraining runs.

## Prior publication disclosure
- **Not an extension of a published conference/workshop paper** → the ≥30%-new-content rule does not apply.
- The manuscript went through one double-blind review round (Dec 2025 version). ✅ Confirmed by Brando (2026-07-21): that round is concluded and Compel is not under review anywhere, so DMLR's no-concurrent-submission policy is satisfied. ⚠ If the form asks for prior-review history, Elyas/Brando should name the venue of that Dec-2025 round. No arXiv record exists yet (posting to arXiv is allowed and non-archival).

## IRB / human subjects
N/A — no human-subjects research; all corpora and benchmarks are pre-existing public resources.

## New dataset/benchmark supplement
N/A — no new dataset or benchmark is introduced (existing corpora are filtered; the filter is the contribution). Stated explicitly for the AE.

## Code availability
https://github.com/eobbad/Compel — compression-ratio computation and distribution-analysis scripts (`experiments/data_analysis/`: the per-corpus histogram pipeline). ⚠ Add the pretraining/filtering pipeline repo if it lives elsewhere (Elyas / Marin infra).

## Funding ⚠
To be filled (Stanford CS / CRFM / STAIR lab funding; any grants supporting the six authors).

## Competing interests ⚠
To be filled (default: none beyond listed affiliations).

## Conflicts of interest (last 3 years)
- Domain conflicts: **stanford.edu** (all six authors). ⚠ Add other institutions from the last 3 years per author (internships, prior employers — e.g., any industry affiliations of D. Hall / R. Schaeffer).
- Personal conflicts: ⚠ check the DMLR action-editor list for recent collaborators of any author (the author list spans STAIR + CRFM, so several Stanford-adjacent AEs may conflict).

## Keywords
pretraining data curation, data quality, data filtering, compression ratio, LZ4, information density, minimum description length, large language models, data-centric machine learning

## After submission
- Put the assigned forum URL into `\def\openreview{...}` in `00_dmlr_compel.tex` at camera-ready.
- If desired, post to arXiv (allowed by DMLR) and add the arXiv number to the CV entry.
