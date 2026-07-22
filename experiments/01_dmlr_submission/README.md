# 01_dmlr_submission — DMLR 2026 Submission Package for ZIP-FIT

> **⚠ SUPERSEDED as primary venue (2026-07-21):** ZIP-FIT now targets **TMLR** — see
> `experiments/03_tmlr_submission/` and `paper_latex/TMLR_2026_ZipFit/`. This DMLR package is
> retained fully submission-ready as the **fallback venue**. Do not submit it to DMLR while the
> TMLR review is active (both venues prohibit concurrent submissions).

**TLDR:** Everything needed to submit ZIP-FIT to DMLR (Journal of Data-centric Machine Learning Research). The DMLR-formatted paper lives at `paper_latex/DMLR_2026_ZipFit/` (30 pp, official `dmlr2e.sty`, compiles with 0 errors / 0 undefined refs); this folder holds the submission plan, the drafted OpenReview form answers, the external-fetch failure log, and the QA report.

## Contents

| File | What it is |
|---|---|
| `plan.md` | How every DMLR requirement (from data.mlr.press, fetched 2026-07-17) is satisfied, plus the improvement log vs. the arXiv v2 source |
| `openreview_checklist.md` | Drafted answers for the OpenReview submission form + items only Brando can confirm |
| `fetch_attempts.md` | Log of external resources that could not be fetched (OpenReview review records) |
| `qa_report.md` | Mega QA chain results (written at the end of the QA pass) |

## Paper location and build

```bash
cd paper_latex/DMLR_2026_ZipFit
latexmk -pdf 00_dmlr_zipfit.tex   # TeX Live 2024; output: 00_dmlr_zipfit.pdf (30 pages)
```

Source of truth for content: arXiv 2410.18194 v2 e-print (latest public version, includes the
post-review experiments: LESS comparison, QLoRA/4-bit results, compression-algorithm ablations).
Rebuttal-era experiment code lives in `experiments/rebuttals/zipfit/ft_and_eval.py` and
`experiments/AF/` — its results are already in the paper's appendix tables.

## Key submission facts

- **Prior venue**: "ZIP-FIT: … for Code" appeared as a **poster at the ICML 2025 workshop
  "DataWorld: Unifying data curation frameworks across domains"** (icml.cc/virtual/2025/48702).
  Concluded conference review records (recovered 2026-07-21, full texts in `or_reviews/`):
  ICLR 2025 full paper `4JBEpP6eRS` (Reject, 8/6/3/1) and ICLR 2026 code version
  `WgZ9C4q9nH` (2/4/6/6/4). The full paper is an arXiv preprint (2410.18194).
- **DMLR extended-submission rule**: applies to the workshop version → the DMLR version must
  contain ≥30% new content vs. it. Satisfied trivially: the workshop paper covered only the
  code-generation subset; the DMLR version adds Autoformalization (§4), efficiency analysis (§5),
  the misalignment study (§6), and the full appendix.
- **No new dataset/benchmark introduced** → no datasheet supplement required (stated on the form).
- **Authors (single-blind, de-anonymized)**: Elyas Obbad, Iddah Mlauzi, Brando Miranda,
  Rylan Schaeffer, Kamal Obbad, Suhana Bedi, Sanmi Koyejo — Stanford.
