# decision_submission.md — Venue Decision: ZIP-FIT → TMLR, Compel → DMLR

**TLDR:** On 2026-07-21 we split the venues: **ZIP-FIT submits to TMLR** (its benchmark-anchored claims survive generalist review, and TMLR's speed + brand + journal-to-conference pathway help most — especially for Elyas's PhD applications) and **Compel submits to DMLR** (a data-centric journal whose criteria, reviewer priors, and theory canon neutralize exactly the objections that rejected it at ICLR). One venue per paper at a time; the unused venue is each paper's ready-built fallback.

## The decision

| Paper | Venue | Package | Status |
|---|---|---|---|
| ZIP-FIT | **TMLR** (openreview.net/group?id=TMLR, double-blind) | `paper_latex/TMLR_2026_ZipFit/` (23 pp, anonymized) | Mega QA #3 FINAL GATE PASS; merged PR #7 |
| ZIP-FIT (fallback) | DMLR | `paper_latex/DMLR_2026_ZipFit/` (30 pp) | Submission-ready; banner'd "fallback only" |
| Compel | **DMLR** (openreview.net/group?id=DMLR, single-blind) | `~/compel/paper_latex/DMLR_2026_Compel/` (20 pp) | Mega QA #2 FINAL GATE PASS; merged PR #1 |

Hard rule: never have the same paper under review at two venues (both venues prohibit concurrent submissions; attested clean 2026-07-21 — neither paper is under review anywhere).

## Context that shaped the decision

- Both papers' conference rounds concluded in rejection: ZIP-FIT ICLR 2025 (`4JBEpP6eRS`, 8/6/3/1), ZIP-FIT code version ICLR 2026 (`WgZ9C4q9nH`, 2/4/6/6/4), Compel ICLR 2026 (`KFafeqE5fe`, 4/6/2/4). Full texts + gap analyses: `../01_dmlr_submission/or_reviews/`.
- The fatal complaints were **bar mismatches, not claim failures** — and the claim-level items are fixed (ZIP-FIT: pass@k benchmarks + LESS baseline already in v2; Compel: new theory §4, 0.4–1.1 range, hedged scale claim, caption token counts).
- Elyas (first author on both) applies to PhDs this cycle → decision speed and venue name recognition carry real weight.

## Why ZIP-FIT → TMLR

1. **Claims robust to any reviewer pool**: objective benchmark wins (HumanEval Pass@1 18.86 vs LESS 18.06 at ~2000× lower selection cost; 65.8% faster than DSIR; compressor ablations) fit TMLR's two-question test (claims supported? audience interested?) with no novelty bar to trip on.
2. **Speed**: TMLR typically decides in ~2–4 months (large editor pool, hard deadlines) vs DMLR's stated 4–6 (up to 10) — a decision can exist before PhD applications are read.
3. **Brand + pathway**: TMLR is the journal admissions readers recognize, and its J2C track can put an accepted paper back on an ICLR/ICML/NeurIPS stage.
4. **Lab precedent**: CertJudge went to TMLR on 2026-07-17 — the ops (style file, process) were proven days earlier; the TMLR package reused its verified `tmlr.sty`/`tmlr.bst`.
5. **Cost was small**: double-blind conversion took ~1 hour (anonymized wrapper; content byte-identical to the DMLR package; page-1 "Anonymous authors" and a full-PDF leak sweep verified).

## Why Compel → DMLR

1. **Contribution-type fit**: DMLR exists to publish work whose artifact is the data pipeline — filters, quality signals, corpus analyses. Compel is exactly that; at general venues it competes on a novelty axis it doesn't need.
2. **Its reviews died on bars DMLR doesn't apply**: the decisive 2-rating called it "a nice proof-of-concept but not a finished paper" — an ICLR completeness judgment. The claims-vs-evidence items from that review are all fixed; the rest are "do a bigger paper" scope asks.
3. **Evidence style is the field's standard**: nobody multi-seeds 8B pretraining; consistency across 3 corpora × 13 tasks × 2 scales is how FineWeb/DCLM-style papers argue. Data-centric reviewers read "+0.4 on DCLM, for ~free" as improving the gold-standard pipeline; generalists read it as noise.
4. **The theory fix lands in the venue's canon**: §4 plugs Compel into the Shannon → NCD → LM-as-compression lineage that community treats as core, converting "conceptually simple" into "operationally simple, theoretically grounded."
5. **Single-blind helps**: visible Stanford/CRFM authorship and TPU v4-128 infrastructure lend credibility to the expensive experiments; double-blind would hide that context.
6. **Portfolio logic**: the sensitive paper goes where its acceptance odds are highest; the robust paper spends the fast/branded venue; splitting avoids concentrating both papers in one editorial pipeline; each keeps the other venue untouched as fallback.

## Fallback flow (if a rejection happens)

- ZIP-FIT ← TMLR reject → submit the retained DMLR package (`01_dmlr_submission/`, update the two-venue banner first).
- Compel ← DMLR reject → convert to TMLR (mechanical: mirror the ZIP-FIT tmlr wrapper; content is parity-tracked). During any review cycle, the two CPU-cheap strengthening analyses (compressor-robustness of the CR distributions; CR→classifier AUROC) can be added in revision.

**TLDR (bottom):** ZIP-FIT → TMLR because its claims are generalist-proof and TMLR's speed/brand/J2C maximize value (notably for Elyas's applications) at a one-hour conversion cost; Compel → DMLR because that venue's purpose, reviewer priors, evidence norms, and theory canon eliminate precisely the ICLR objections it faced; one venue per paper at a time, with the other venue pre-built as fallback — both packages are QA-gated PASS and merged.
