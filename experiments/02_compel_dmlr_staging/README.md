# 02_compel_dmlr_staging — TEMPORARY: Compel DMLR Package Staged in ZIP-FIT

**TLDR:** Temporary staging copy of the complete, submission-ready Compel DMLR package (paper source + PDF + submission docs), parked here because the canonical Compel *paper* repo is unknown until Elyas replies (asked via Slack 2026-07-17 and email 2026-07-21). Delete this folder once the real repo surfaces and the package is merged there.

## Why this exists (provenance)

- No LaTeX source for the Compel paper exists anywhere on this machine or on GitHub (verified
  twice). The paper was rebuilt number-for-number from the Dec-2025 double-blind review PDF
  (`~/Documents/Research Documents/Research Papers Reference/compel.pdf`) and Mega-QA'd
  (11/11 gate items PASS — see `submission_docs/qa_report.md`).
- Primary copies of this same package:
  - `eobbad/Compel` branch **`dmlr-2026-submission`** @ `55c2083` (github.com/eobbad/Compel)
  - local: `~/Compel/paper_latex/DMLR_2026_Compel/` + `~/Compel/experiments/01_dmlr_submission/`
- This ZIP-FIT copy exists only so the whole DMLR submission (both papers) lives in one repo
  Brando controls until the canonical Compel repo is known.

## Contents

| Path | What |
|---|---|
| `DMLR_2026_Compel/` | Full paper source: `00_dmlr_compel.tex` + 10 section files, `compel_refs.bib`, `dmlr2e.sty`, `figures/` (6 PNGs), compiled `00_dmlr_compel.pdf` (20 pp) |
| `submission_docs/` | `plan.md`, `openreview_checklist.md`, `fetch_attempts.md`, `qa_report.md` (copies of `~/Compel/experiments/01_dmlr_submission/`) |
| `DMLR_2026_Compel_source.tar.gz` | Upload-ready source bundle |

## Build

```bash
cd DMLR_2026_Compel && latexmk -pdf 00_dmlr_compel.tex   # 20 pp, 0 errors / 0 undefined / 0 overfull
```

## Exit criterion

When Elyas points to the canonical Compel paper repo: diff `DMLR_2026_Compel/` against his source,
merge whichever is newer into that repo, submit from there, and **delete this folder** (its content
is fully duplicated on the `eobbad/Compel:dmlr-2026-submission` branch).
