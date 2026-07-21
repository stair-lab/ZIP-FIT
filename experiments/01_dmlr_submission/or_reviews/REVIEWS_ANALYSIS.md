# ZIP-FIT — OpenReview Complaints vs. the DMLR Version

**TLDR:** Both concluded review rounds recovered (ICLR 2025 full paper `4JBEpP6eRS`: 8/6/3/1 → Reject; ICLR 2026 code version `WgZ9C4q9nH`: 2/4/6/6/4). The two **[Major]** ICLR-2025 complaints — benchmark-based evaluation and stronger baselines — are substantially addressed in the arXiv-v2 content the DMLR version is built on (HumanEval Pass@1/Pass@10, LESS/D4/DSIR comparisons, compression-algorithm ablations). The recurring open item across ALL nine reviewers is domain breadth (only Autoformalization + code). Raw review JSON: `or_zipfit_iclr2025.json`, `or_zipfit_code_iclr2026.json` (recovered via the public ICLR review scrapes `smallari/openreview-iclr-peer-reviews` and `davidheineman/iclr-2026` on Hugging Face, 2026-07-21).

## ICLR 2025 (full paper, Reject, ratings 8/6/3/1)

| Complaint (reviewer, rating) | Status in DMLR version |
|---|---|
| **[Major]** Only test-loss results; need HumanEval/MBPP pass@k (R3:3, R4:1) | ✅ **Addressed for code**: HumanEval Pass@1/Pass@10 tables (Full-FT 18.86% vs LESS 18.06% vs DSIR 17.98%; QLoRA 12.19% vs DSIR 9.14% vs D4 6.09%; base 15.24%). ⚠ Still open: functional metric for Autoformalization (miniF2F-style proof rates) and MBPP. |
| **[Major]** Inadequate baselines — LESS, Alpagasus, Cherry, Instruction Mining, embedding-based (R4:1) | 🟡 **Partial**: LESS (gradient-based, the strongest named) added and beaten at ~2000× lower selection cost. Embedding-based and LLM-judge selection baselines still absent. |
| **[Major]** Method "very simple / standard" (R3:3) | 🟡 Reframed, not "fixed": R1 (rating 8) explicitly praised the conceptual simplicity; DMLR's data-centric scope treats a cheap, effective data method as the contribution itself. |
| No evidence gzip captures syntax/structure (R4) | 🟡 Partial: alignment-vs-downstream-loss correlation analyses (R²≈0.9) in the paper; the MDL/Kolmogorov theory section written for Compel's DMLR version makes the general argument and could be ported in one paragraph. |
| Compressor choice — would others work? (R2:6) | ✅ Addressed: compression-algorithm/level ablation (LZ4 level-0 12.19% vs gzip 11.58% Pass@1). |
| Why was D4 excluded from code-gen? (R4) | ✅ Addressed: D4 included in the QLoRA comparison. |
| Target-set size $n$ ablation (R1:8) | ❌ Open (cheap-ish: selection is CPU-only; needs small finetunes per $n$). |
| More domains beyond AF/code (R1, R2) | ❌ Open — see below; the most repeated complaint across both rounds. |
| Figure 2 quality / rushed writing (R3) | ✅ v2 figures + the DMLR pass (0 errors / 0 undefined; dangling citation repaired). |

## ICLR 2026 (code version, ratings 2/4/6/6/4)

| Complaint | Status in DMLR version |
|---|---|
| Only program-synthesis domains; generalization unclear (all 5 reviewers; "a regex could find Python code" — R:4; MultiPL-E suggested) | ❌ Open experimentally. The DMLR version is the FULL paper (AF + code + misalignment study), which widens scope vs. the code-only manuscript they reviewed, but adds no new domain. |
| Only Gemma-2-2B on HumanEval; unclear zero-shot vs FT settings in tables | 🟡 Partial: interventional/misalignment sections use additional models; the HumanEval table is single-model. Baseline rows are labeled "Pre-trained" in the DMLR version. |
| No finetune-on-all reference curve; embedding-based baselines missing | ❌ Open (overlaps ICLR-2025 baseline complaint). |
| Diversity-of-target-set influence not quantified | ❌ Open. |
| "Table ??", missing citations, broken appendix formatting, typos | ✅ Not present in the DMLR build (compiles with 0 undefined refs; those flaws were specific to the ICLR-2026 manuscript). |
| ProofNet syntax regularity may inflate R²=0.9 (R:4) | ❌ Open (would need a lexically-diverse target set replication). |

## Bottom line for the DMLR decision
The complaints that sank ZIP-FIT at ICLR 2025 (loss-only evaluation, missing LESS-class baselines) are the ones arXiv v2 — and therefore the DMLR version — actually fixed. What remains open (domain breadth, embedding-baseline head-to-heads, $n$-ablation) is real but is "more scope" rather than "claims unsupported": every claim in the DMLR version is backed by its tables. DMLR's acceptance test is claims⇔evidence + data-centric interest, not novelty-wow.
