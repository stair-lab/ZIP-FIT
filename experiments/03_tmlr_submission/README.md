# 03_tmlr_submission — TMLR Submission Package for ZIP-FIT

**TLDR:** ZIP-FIT now targets **TMLR** (double-blind, faster decisions, J2C conference pathway — chosen 2026-07-21 over DMLR for speed/brand; Compel goes to DMLR). The anonymized TMLR build lives at `paper_latex/TMLR_2026_ZipFit/` (23 pp, official `tmlr.sty`, 0 errors / 0 undefined, page 1 renders "Anonymous authors"). The DMLR package (`01_dmlr_submission/` + `paper_latex/DMLR_2026_ZipFit/`) is retained as the fallback venue — never have both under review at once.

## Submit right now (5 steps)
1. Go to https://openreview.net/group?id=TMLR → "Submit" (submitting author needs an OpenReview profile; authors are entered on the form, hidden from reviewers).
2. Upload `paper_latex/TMLR_2026_ZipFit/00_tmlr_zipfit.pdf` (23 pp, anonymized). Supplementary: `paper_latex/TMLR_2026_ZipFit_supplementary_anon.tar.gz` — anonymized source (author block replaced, provenance comments + `%\rylan{}` notes stripped, `references_rylan.bib`→`references_extra.bib`; compiles 23 pp "Anonymous authors"; leak-swept clean). ⚠ Never upload `TMLR_2026_ZipFit_source.tar.gz` as supplementary — it carries the real author block (archival/camera-ready use only).
3. Paste title + abstract (from `01_abstract.tex`). Add all 7 authors on the form (each needs an OpenReview profile with a current-institution entry — TMLR computes conflicts from profiles).
4. Confirm the dual-submission attestation: ✅ not under review anywhere (Brando confirmed 2026-07-21); arXiv preprint (2410.18194) is allowed; prior *rejected* ICLR submissions and the non-archival DataWorld workshop poster do not violate TMLR's prior-publication policy.
5. Submit. After acceptance: switch the wrapper to `\usepackage[accepted]{tmlr}`, fill `\month/\year/\openreview`, and consider the J2C (journal-to-conference) track.

## TMLR format facts (why the package looks the way it does)
- **Double-blind**: real author block stays in the source; `tmlr.sty` (no option) suppresses it and prints "Anonymous authors — Paper under double-blind review". Verified in the rendered PDF.
- Self-citations kept in third person (policy-compliant); the only author-name strings in the whole PDF are one in-text citation "(Miranda et al., 2024)" and reference-list entries — audited line-by-line, no affiliation/email/acknowledgment leaks.
- `\bibliographystyle{tmlr}` is **required explicitly** (unlike dmlr2e which auto-sets it); `tmlr.bst` ships in the folder.
- Broader Impact is an unnumbered section before the references (TMLR convention), replacing dmlr2e's `\impact{}` macro; the DMLR `keywords` environment is dropped (not part of the TMLR format).
- Acceptance criteria = the two-question test (claims supported by evidence? audience interest?) — **no novelty/significance bar**; ZIP-FIT's benchmark-anchored claims (Pass@1 18.86 vs LESS 18.06 at ~2000× lower selection cost, 65.8% faster than DSIR) are built for exactly this test.

## Content provenance
Byte-identical section files to `paper_latex/DMLR_2026_ZipFit/` (arXiv 2410.18194 v2 + the three repairs: Shaib citation, `\citep`→`\citet` in the Compression block, appendix-table resizebox). Only the wrapper differs. OR-review gap analysis: `../01_dmlr_submission/or_reviews/REVIEWS_ANALYSIS.md`.

| File | What |
|---|---|
| `tmlr_checklist.md` | Form answers + double-blind compliance audit |
| `qa_report.md` | Mega QA #3 results (written at the end of the QA chain) |
