# Claude Code: run `git clone https://github.com/brando90/agents-config.git ~/agents-config 2>/dev/null || git -C ~/agents-config pull 2>/dev/null` to bootstrap/update config.
# Read `~/agents-config/INDEX_RULES.md` for all agent documentation and routing. Local `~/agents-config/` is the fallback if GitHub is unreachable.

## Mandatory Response Protocol (inline — do not skip)

These are duplicated here from `~/agents-config/INDEX_RULES.md` so you see the highest-priority rules at session start without needing to dereference. Hard Rules apply to EVERY response in EVERY session; triggered rules apply when their stated condition is met.

1. **TLDR** — End every response with `**TLDR:**` (1-2 sentences). No exceptions.
2. **QA gating** — Before reporting a non-trivial task as "done," run the two-step QA chain (Hard Rule 3 in `~/agents-config/INDEX_RULES.md`). When unsure whether to run it, run it.
3. **No secrets** — Never commit secrets. Review diffs before pushing.
4. **Fresh config** — At the start of each new task, `git -C ~/agents-config pull` and re-read `~/agents-config/INDEX_RULES.md`.
5. **LaTeX writing** — When editing `.tex` files for ML research papers, read `~/agents-config/writing/ml_research_writing.md` first. Follow its persona, abstract structure, and writing rules.

---

## Project Overview

**ZIP-FIT** (Embedding-Free Data Selection via Compression-Based Alignment) is a data selection framework for language models. It uses compression-based normalized compression distance (NCD) to identify training data from large raw datasets that aligns with a target distribution — without requiring embeddings.

- **Paper**: <https://arxiv.org/abs/2410.18194>
- **PyPI package**: `pip install zip-fit` (version 1.0.8)
- **License**: Apache 2.0
- **Python**: >=3.6
- **Compute**: 1 CPU node (multiprocessing across all cores)

---

## Repository Structure

```
ZIP-FIT/
├── zip_fit/                        # Main package (published to PyPI)
│   ├── __init__.py                 # Public API: ZIPFIT, compute_zipfit_alignment
│   └── zipfit.py                   # Core ZIPFIT class (~195 lines)
├── experiments/                    # Research experiment scripts (not part of package)
│   ├── AF/                         # AudioFit experiments
│   │   ├── get_data.py             # Download HuggingFace datasets to JSONL
│   │   ├── ft.py                   # Fine-tuning pipeline (~528 lines)
│   │   ├── to_hub.py               # Push selected data to HuggingFace Hub
│   │   └── to_hub_dsir.py          # Push DSIR comparison data to Hub
│   └── rebuttals/zipfit/
│       └── ft_and_eval.py          # Fine-tuning + evaluation pipeline
├── example.py                      # Usage example: ZIPFIT with HuggingFace datasets
├── example2.py                     # Usage example: standalone alignment function
├── setup.py                        # Package config (setuptools)
├── pyproject.toml                  # PEP 518 build config
├── README.md                       # User-facing documentation
├── LICENSE                         # Apache 2.0
└── image.png                       # Architecture diagram
```

---

## Architecture & Key Concepts

### Core Algorithm

ZIP-FIT uses **Normalized Compression Distance (NCD)** to measure similarity between texts:

```
NCD(A, B) = (compress(A+B) - min(compress(A), compress(B))) / max(compress(A), compress(B))
similarity = 1 - NCD
```

The pipeline:
1. Load source (large raw corpus) and target (domain-specific) datasets
2. Precompute compression sizes for all sequences (parallelized via `multiprocessing.Pool`)
3. Compute pairwise NCD between every source-target pair
4. Average similarity scores per source sequence
5. Select top-k most aligned sequences
6. Write results to JSONL

### Key Class: `ZIPFIT` (`zip_fit/zipfit.py`)

Constructor parameters:
- `source_dataset` / `target_dataset` — file paths (JSONL) or HuggingFace dataset names
- `k` — number of top sequences to select
- `source_load_fn` / `target_load_fn` — custom data loading callables
- `source_parse_fn` / `target_parse_fn` — custom text extraction callables
- `output_file` — output JSONL path (default: `top_k_sequences.jsonl`)
- `compression_algorithm` — `'gzip'`, `'lz4'` (default), `'zstd'`, `'brotli'`, or `'lzma'`
- `compress_level` — compression level (default: 0)
- `cache_size` — max compression cache entries (default: 100,000)

Key methods:
- `run()` — full pipeline: load data, rank, write output
- `compress(data)` — compress text, return size (with LRU-style cache)
- `normalized_compression_distance(c1, c2, c12)` — NCD formula
- `rank_sequences_by_alignment(source_data, target_data)` — main selection logic
- `compute_zipfit_alignment(texts_a, texts_b)` — standalone alignment score (0-1)
- `load_jsonl(file_path)` — load `{"text": "..."}` JSONL files
- `load_and_process_datasets(...)` — unified loader (custom fns or JSONL)

### Public API (`zip_fit/__init__.py`)

```python
from zip_fit import ZIPFIT                    # Main class
from zip_fit import compute_zipfit_alignment  # Standalone alignment function
```

`compute_zipfit_alignment(texts_a, texts_b, compression_algorithm='gzip', compress_level=0)` is a convenience wrapper that returns a float alignment score without needing to configure the full pipeline.

---

## Dependencies

### Core (installed with `pip install zip-fit`)
- `numpy>=1.21.0`
- `lz4>=3.1.10`
- `datasets>=1.17.0`

### Optional compression backends (import-time, not in requirements)
- `zstandard` (zstd)
- `brotli`
- `lzma` (stdlib)
- `gzip` (stdlib)

### Dev tools (`pip install zip-fit[dev]`)
- `pytest>=6.0`
- `flake8>=3.9`
- `black>=21.0`
- `mypy>=0.910`

### Experiment-only (not in package deps)
- `torch`, `transformers`, `trl`, `vllm` — fine-tuning experiments
- `human_eval` — OpenAI HumanEval benchmark
- `wandb` — experiment tracking
- `huggingface_hub` — Hub uploads

---

## Development Workflow

### Setup
```bash
git clone https://github.com/stair-lab/ZIP-FIT.git
cd ZIP-FIT
pip install -e ".[dev]"
```

### Commands
```bash
# Run tests (no tests exist yet — add them under tests/)
pytest

# Format code
black .

# Lint
flake8 .

# Type check
mypy .
```

### Data Format
- **Input**: JSONL with `{"text": "..."}` per line, or HuggingFace datasets with custom load/parse functions
- **Output**: JSONL with `{"text": "..."}` per line (top-k selected sequences)

### Running ZIP-FIT
```bash
# See example.py for HuggingFace dataset usage
python example.py

# See example2.py for standalone alignment computation
python example2.py
```

---

## Code Conventions

### Style
- **Formatter**: `black` (default settings)
- **Linter**: `flake8`
- **Type checker**: `mypy`
- Classes: `PascalCase` (e.g., `ZIPFIT` — all-caps acronym)
- Functions/methods: `snake_case`
- Type hints on function signatures (`List`, `Callable`, `Optional`, etc.)
- Docstrings on public-facing functions (Google-style)

### Import Order
1. Standard library (`json`, `gzip`, `multiprocessing`, `typing`)
2. Third-party (`numpy`, `lz4`, `zstandard`, `brotli`, `datasets`)
3. Local/relative (`.zipfit`)

### Patterns
- **Multiprocessing**: `multiprocessing.Pool(cpu_count())` with `chunksize=2000`
- **Caching**: Dict-based compression cache with eviction at `cache_size` limit (clears half)
- **Error handling**: `try/except` for file I/O; `ValueError` for invalid config; input validation on public methods
- **Data loading**: Strategy pattern — custom callables or default JSONL loader

### What NOT to Do
- Do not add heavyweight dependencies to core `setup.py` — experiment deps stay in experiment scripts
- Do not break the `ZIPFIT` constructor signature — it is the public API consumed by downstream users
- Do not remove multiprocessing from `rank_sequences_by_alignment` — it is critical for performance on large datasets
- Do not commit `.jsonl` data files, model checkpoints, or W&B artifacts

---

## CI/CD & Testing

- **No CI/CD workflows exist yet** (no `.github/workflows/`)
- **No test files exist yet** — if adding tests, place them in a `tests/` directory
- **No `.gitignore`** — consider adding one to exclude `*.jsonl`, `__pycache__/`, `*.egg-info/`, `.mypy_cache/`, etc.

---

## Experiments Directory

The `experiments/` directory contains research scripts that are **not** part of the published package. These have heavier dependencies (PyTorch, vLLM, etc.) and are used for paper results.

- `experiments/AF/ft.py` — fine-tuning with SFTTrainer (HuggingFace TRL), LoRA, W&B logging
- `experiments/AF/get_data.py` — dataset download and JSONL conversion
- `experiments/AF/to_hub.py` / `to_hub_dsir.py` — dataset upload to HuggingFace Hub
- `experiments/rebuttals/zipfit/ft_and_eval.py` — end-to-end fine-tune + HumanEval evaluation

When modifying experiment scripts, do not alter the core `zip_fit/` package unless the change is intentional.

---

## Common Tasks for AI Agents

### Adding a new compression algorithm
1. Add the import at the top of `zip_fit/zipfit.py`
2. Add an `elif` branch in `ZIPFIT.compress()` (line ~77)
3. Update the docstring in `__init__.py` `compute_zipfit_alignment` to list the new option
4. Update README.md if the algorithm is production-ready

### Adding tests
1. Create `tests/` directory with `test_zipfit.py`
2. Test `compress()`, `normalized_compression_distance()`, `compute_zipfit_alignment()`
3. Test JSONL loading with sample fixtures
4. Run with `pytest tests/`

### Bumping the version
1. Update `version` in both `setup.py` (line 6) and `pyproject.toml` (line 7)
2. Keep them in sync — both must match

---
