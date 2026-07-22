# TMLR Form Answers + Double-Blind Compliance — ZIP-FIT

**TLDR:** Everything needed for the TMLR OpenReview form (https://openreview.net/group?id=TMLR), plus the anonymity audit. ⚠ marks human-only items.

## Title
ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment

## Abstract
Verbatim from `paper_latex/TMLR_2026_ZipFit/01_abstract.tex` (active block).

## Authors (entered on the form, hidden from reviewers; order as in source)
Elyas Obbad, Brando Miranda, Iddah Mlauzi, Rylan Schaeffer, Kamal Obbad, Suhana Bedi, Sanmi Koyejo — Stanford.
⚠ Each needs an OpenReview profile with current institution + DBLP/homepage (TMLR derives conflicts from profiles; no separate conflicts field).

## Double-blind compliance audit (2026-07-21)
- Page 1: "Anonymous authors — Paper under double-blind review"; running head "Under review as submission to TMLR" ✓ (verified via pdftotext on the built PDF).
- Full-PDF sweep for author surnames / "Stanford": 7 hits, all policy-compliant — 1 third-person in-text citation "(Miranda et al., 2024)" + 6 reference-list author lines (Gerstgrasser et al., Kazdan et al., Lee et al., Miranda et al. entries). No affiliations, emails, acknowledgments, or grant numbers anywhere in the rendered text ✓.
- No GitHub/repo links in the body ✓ (code URL goes on the form / camera-ready only).

## Dual-submission attestation
✅ Not under review anywhere (Brando, 2026-07-21). Prior history — all permitted at TMLR: arXiv:2410.18194 (preprints allowed); ICLR 2025 `4JBEpP6eRS` (rejected, concluded); ICLR 2026 code-version `WgZ9C4q9nH` (concluded); ICML 2025 DataWorld workshop poster (non-archival).
⚠ Rule going forward: the DMLR fallback package must NOT be submitted while TMLR review is active.

## Code availability (form field / camera-ready)
https://github.com/stair-lab/ZIP-FIT — omit from the reviewed PDF; provide on the form if asked (OpenReview keeps it from reviewers per venue config) or at camera-ready.

## Certifications to consider post-acceptance
J2C (journal-to-conference) track; Reproducibility certification if the pip package + scripts are highlighted.

## After acceptance
`\usepackage[accepted]{tmlr}`, fill `\def\month`, `\def\year`, `\def\openreview{...forum id...}`, restore code link, add acknowledgments.
