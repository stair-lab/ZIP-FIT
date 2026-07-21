# 01_dmlr_submission — DMLR 2026 Submission Package for Compel

**TLDR:** Everything needed to submit *Curating High Quality Pretraining Data for Language Models via Compression Ratios* (Compel) to DMLR. The DMLR-formatted paper lives at `paper_latex/DMLR_2026_Compel/` (20 pp, official `dmlr2e.sty`, 0 errors / 0 undefined refs / 0 overfull); this folder holds the plan, the OpenReview form answers, the fetch-failure log, and the QA report.

## ⚠ Provenance — read first

No LaTeX source for the Compel paper exists anywhere on this machine (verified twice, including
GitHub-wide search in a prior session). The canonical paper repo location is **pending Elyas's
reply** (Brando's Slack message, 2026-07-17). This package was therefore **rebuilt faithfully from
the double-blind review PDF** `~/Documents/Research Documents/Research Papers Reference/compel.pdf`
(created 2025-12-01, 14 pp), then de-anonymized per DMLR's single-blind policy and improved
(see `plan.md` § Improvement log). When Elyas surfaces the original repo, diff this version
against it and merge whichever is newer — the numbers here match the Dec-2025 PDF exactly.

Local `~/Compel` (github.com/eobbad/Compel) is the *code* repo — compression-ratio analysis
scripts and plots under `experiments/data_analysis/` (the per-corpus compression-ratio histograms;
Figure 2 in the DMLR version's numbering, Figure 3 in the review PDF's), no paper.
That is likely the repo Brando was trying to remember; it is under Elyas's personal GitHub, not
stair-lab.

## Contents

| File | What it is |
|---|---|
| `plan.md` | DMLR requirement mapping + improvement log vs. the Dec-2025 review PDF |
| `openreview_checklist.md` | Drafted OpenReview form answers + items Brando must confirm |
| `fetch_attempts.md` | Log of unreachable external resources (review texts) |
| `qa_report.md` | Mega QA chain results (written at the end of the QA pass) |

## Paper location and build

```bash
cd paper_latex/DMLR_2026_Compel
latexmk -pdf 00_dmlr_compel.tex   # output: 00_dmlr_compel.pdf (20 pages)
```

## Key submission facts

- **Results (verbatim from the Dec-2025 PDF):** 1.4B models: macro-avg 42.4→43.5 (FineWeb),
  44.2→44.9 (FineWeb-EDU), 46.6→47.0 (DCLM); 8B FineWeb: macro 0.570→0.572, micro 0.587→0.593.
- **Prior venue**: none published — the paper is an unpublished manuscript that was under
  double-blind review (venue per the Dec-2025 PDF unknown locally). CV lists it as "Preprint 2026".
  ⇒ DMLR's ≥30%-new-content extended-submission rule is **N/A** (it applies only to prior
  conference/workshop *publications*). The DMLR version still adds substantial new content
  (theory section §4, contributions list, repaired citations).
- **⚠ Dual-submission check**: DMLR prohibits concurrent archival submissions. Brando/Elyas must
  confirm the earlier double-blind review has concluded before submitting.
- **No new dataset/benchmark introduced** → no datasheet supplement required.
- **Authors (single-blind, from Brando's CV)**: Elyas Obbad, Brando Miranda, David L. W. Hall,
  Rylan Schaeffer, Sanmi Koyejo, Percy Liang — Stanford. ⚠ Confirm order + emails with Elyas.
