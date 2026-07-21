# fetch_attempts.md — External Resources That Could Not Be Fetched (Compel DMLR prep)

**TLDR:** The Compel review texts (the source of the "conceptually simple / lacks deeper theoretical grounding" criticism Brando quoted) could not be located or fetched — OpenReview is bot-walled from this machine and no local copy of the reviews exists. The criticism was addressed as stated by Brando; nothing was fabricated.

| When (2026-07-17) | Resource | Method | Result |
|---|---|---|---|
| ~15:05 | OpenReview search API v2 for the paper title | urllib GET | 0 hits (double-blind record not indexed or search blocked) |
| ~15:05 | OpenReview search API v1 for the paper title | urllib GET | 200 but unranked/garbage results — paper not surfaced |
| ~15:06–15:10 | OpenReview forum/API endpoints generally | urllib + curl with browser headers, WebFetch | **HTTP 403** / browser-verification interstitial (Cloudflare bot wall) |
| ~15:07 | Local search for review texts: `grep` over `~/.claude/projects/*`, `mdfind` (goldilocks, compel), Cursor history, `~/Downloads` | local | Only the review *PDF* of the paper found (`compel.pdf`); no reviewer comments anywhere local |

## What was used instead
- `~/Documents/Research Documents/Research Papers Reference/compel.pdf` (Dec-2025 double-blind
  version, 14 pp) — full content ground truth; all numbers transcribed from it.
- Brando's CV (`~/brandomiranda/professional_documents/cvs/cv_long.tex` line ~450) — author list.
- The reviewer criticism as quoted by Brando ("conceptually simple / lacks deeper theoretical
  grounding") — addressed with the new §4 theory section.

## What Brando/Elyas should supply if desired
- The actual review texts (venue + forum link) for a point-by-point rebuttal-style pass.
- The original paper LaTeX repo (pending Elyas's Slack reply) to diff against this rebuild.
