# cowork_submission_prompt.md — Dispatch Prompt: Submit ZIP-FIT→TMLR + Compel→DMLR

**TLDR:** Paste the fenced prompt below into a fresh Claude session (cowork mode, with Brando at the browser) to execute both OpenReview submissions using the already-QA'd packages. The agent drives every step it can, hands identity-bound steps (login, profiles, funding/conflicts, the final Submit click) to Brando, and records the resulting forum links back into the repos.

```
We are submitting two QA-gated papers to OpenReview TODAY, coworking: you drive, I (Brando) handle credentials and final clicks. Everything is prepared and verified — your job is faithful execution, recording, and verification. Do NOT edit any paper content.

LOGIN MODEL: you drive my real browser, which is likely ALREADY logged into openreview.net — check first and proceed directly if so (no login needed). If you hit a login wall, leave that tab as-is and continue all other prep; the sign-in becomes an item at the final checkpoint (never type/read my passwords; 2FA is mine).

INTERACTION CONTRACT — be maximally automatic, ask NOTHING until one final checkpoint:
- Do not ask me questions mid-run. Accumulate every open item (login wall, missing OpenReview profiles, funding, conflicts, consent boxes) into a list.
- Front-load everything automatable for BOTH submissions in parallel: pre-flight PDF checks, open both forms, fill every field you can verbatim from the checklists, attach the uploads.
- File uploads are YOURS: drive the picker or set the file input yourself — never hand me a "click Choose File" step (ac Trigger Rule 34).
- FINAL CHECKPOINT (the only interaction): show me a compact summary — each form's filled fields, plus the accumulated items only I can do. I complete those, review, and click Submit on both. Then resume fully automatic for the closeout (record, push, verify, email).
- Note: a folder-access dialog for ~/ZIP-FIT and ~/compel at session start is expected — that grant is the session's, not a question for the run.

READ FIRST (ground truth, in this order):
1. ~/ZIP-FIT/experiments/03_tmlr_submission/decision_submission.md   (venue decision + fallback rules)
2. ~/ZIP-FIT/experiments/03_tmlr_submission/{README.md, tmlr_checklist.md, qa_report.md}
3. ~/compel/experiments/01_dmlr_submission/{openreview_checklist.md, qa_report.md}

SUBMISSION A — ZIP-FIT → TMLR (https://openreview.net/group?id=TMLR):
1. Pre-flight (do yourself, in terminal): pdftotext page 1 of ~/ZIP-FIT/paper_latex/TMLR_2026_ZipFit/00_tmlr_zipfit.pdf and CONFIRM it reads "Anonymous authors" + "Paper under double-blind review". If it does not, STOP — wrong file.
2. Open the TMLR submission form and fill it field by field from tmlr_checklist.md: title, abstract, all 7 authors in order (Obbad, Mlauzi, Miranda, Schaeffer, Obbad, Bedi, Koyejo), upload the anonymized PDF, and upload the supplementary tarball ~/ZIP-FIT/paper_latex/TMLR_2026_ZipFit_supplementary_anon.tar.gz (leak-audited anonymized source — NEVER upload TMLR_2026_ZipFit_source.tar.gz, which contains the real author block).
3. If any author's OpenReview profile is missing/incomplete, pause and tell me exactly which and what's needed — do not guess emails.
4. Dual-submission attestation: answer truthfully per the checklist (nothing under review; arXiv + concluded rejected rounds + non-archival workshop are permitted).
5. I click Submit. You record the forum URL + submission number.

SUBMISSION B — Compel → DMLR (https://openreview.net/group?id=DMLR):
1. Pre-flight: pdftotext page 1 of ~/compel/paper_latex/DMLR_2026_Compel/00_dmlr_compel.pdf and CONFIRM it shows the six named authors (Obbad, Miranda, Hall, Schaeffer, Koyejo, Liang) — DMLR is single-blind; if it says "Anonymous", STOP — wrong file.
2. Guide me through the DMLR form using openreview_checklist.md verbatim: title, abstract, keywords, data-centric scope statement, 6 authors, prior-review disclosure (not an extension; 30% rule N/A; prior double-blind round concluded), IRB N/A, datasheet N/A, code links (github.com/eobbad/Compel + Marin note).
3. Funding + conflicts are MINE to fill — prompt me for them (3-year institutions per author; check the DMLR action-editor list against recent collaborators) and wait.
4. I click Submit. You record the forum URL + submission number.

HARD GUARDRAILS:
- One venue per paper, ever, at a time. Never touch ~/ZIP-FIT/paper_latex/DMLR_2026_ZipFit (that is the FALLBACK package — submitting it now would be a concurrent-submission violation).
- Never swap the PDFs: TMLR gets the anonymized one, DMLR gets the authored one. Re-verify at upload time, not from memory.
- Anything requiring my identity or judgment (logins, profile creation, funding, conflicts, consent boxes, the Submit button) — hand to me explicitly and wait.

AFTER BOTH SUBMISSIONS:
1. Write <forum URL, submission number, timestamp, venue> into ~/ZIP-FIT/experiments/03_tmlr_submission/submission_record.md and ~/compel/experiments/01_dmlr_submission/submission_record.md (Title + TLDR header per house rules). Note in each: fill \def\openreview{...} in the wrapper at camera-ready.
2. Commit + push both repos (main).
3. Verification pass: have me open my OpenReview author console and confirm both submissions are listed; screenshot or transcribe the two entries into the submission_record files.
4. Email a one-paragraph confirmation with both forum links to brando.science@gmail.com (send via uutils.emailing.send_email_smtp, smtp_user brandojazz@gmail.com, smtp_pass_file ~/keys/gmail_app_password.txt, no CC).
5. Print a final PASS/FAIL line per paper: submitted, recorded, pushed, emailed.
```

**TLDR (bottom):** Dispatch the fenced prompt in a cowork session to execute both submissions: it pre-verifies the right PDF goes to the right venue (anonymity check for TMLR, author-block check for DMLR), walks Brando through each form from the drafted checklists, refuses to touch the fallback package, then records forum links in both repos, pushes, and emails confirmation.
