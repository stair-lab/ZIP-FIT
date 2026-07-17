# fetch_attempts.md — External Resources That Could Not Be Fetched (ZIP-FIT DMLR prep)

**TLDR:** OpenReview's API and site are bot-walled from this machine, so the ZIP-FIT review texts (ICLR record `4JBEpP6eRS`) could not be read; everything else needed for the submission was reachable. Nothing was fabricated — the reviews are simply not incorporated verbatim.

| When (2026-07-17) | Resource | Method | Result |
|---|---|---|---|
| ~15:05 | `https://api2.openreview.net/notes?forum=4JBEpP6eRS` | urllib GET (UA: Mozilla/5.0) | **HTTP 403** (Cloudflare bot wall) |
| ~15:05 | `https://api.openreview.net/notes?forum=4JBEpP6eRS` | urllib GET | **HTTP 403** |
| ~15:05 | same for forum `WgZ9C4q9nH` (both API hosts) | urllib GET | **HTTP 403** |
| ~15:06 | `https://openreview.net/forum?id=4JBEpP6eRS` | WebFetch (rendered) | Browser-verification interstitial only |
| ~15:10 | `https://api2.openreview.net/notes?forum=4JBEpP6eRS` | curl with full Chrome headers | **HTTP 403** |
| ~15:05 | `https://api.openreview.net/notes/search?term=...` | urllib GET | 200 but unranked/garbage results (search index did not surface the papers) |

## What was reachable instead
- ICML 2025 virtual page (icml.cc/virtual/2025/48702) → confirmed the DataWorld workshop venue, poster format, author list.
- arXiv e-print 2410.18194 (latest version) → full paper source used for the DMLR build.

## What Brando should supply if desired
- The ICLR review texts for `4JBEpP6eRS` (open the forum in a logged-in browser) if we want a
  point-by-point "reviewer concerns addressed" pass beyond what arXiv v2 already incorporates.
- Confirmation that the `4JBEpP6eRS` process is concluded (needed for DMLR's no-concurrent-submission rule).
