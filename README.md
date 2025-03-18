# Data Selection for Language Models via Compression
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2305.10429-00ff00.svg)](https://arxiv.org/abs/2410.18194)

This repository hosts the [ZIP-FIT](https://arxiv.org/abs/2410.18194) data selection framework, designed to effectively and efficiently select relevant training data for language models from any data source based on a specified target dataset.

ZIP-FIT is optimized for:
- Rapid, large-scale data selection from extensive raw text datasets.
- Identifying data that closely aligns with the distribution of a given target dataset (e.g., domain-specific data, HumanEval, etc.).

Compute needed:
- 1 CPU node

![ZIP-FIT figure](image.png)

## Quickstart

Install with pip:
```
pip install zip-fit
```

To select data, simply initialize a `ZIPFIT` object and call the following functions:
```python
from zip_fit import ZIPFIT

source_dataset = <path>
target_dataset = <path>
top_k = 10000

zipfit = ZIPFIT(source_dataset, target_dataset, k=top_k, output_file="top_k_sequences.jsonl")
zipfit.run()
```
Executing this process will generate a jsonl file named 'top_k_sequences.jsonl', containing 10,000 documents. For optimal performance, it is recommended to use uncompressed jsonl files stored on local file storage for all data paths, and to utilize as many CPU cores as possible. You can provide custom functions for reading the data paths and extracting the text field from each example using the {source,target}_load_dataset_fn and {source,target}_parse_example_fn parameters in the constructor.
 

## Examples

HuggingFace datasets can also be used in either `source_dataset` or `target_dataset`. However, please note that streaming a large raw dataset directly may result in slow performance; this approach is better suited for target datasets:

```python
from zip_fit import ZIPFIT
from datasets import load_dataset

source_dataset = f'/path/to/source.jsonl'
target_dataset = 'openai/openai_humaneval'

# Define the function to load the target dataset
def target_load_dataset_fn(dataset):
    ds = load_dataset(dataset, split='test', trust_remote_code=True)
    return ds

# Define the function to parse examples from the target dataset
def target_parse_example_fn(ex):
    text = f"Problem description: {ex['prompt']} \nCanonical solution: {ex['canonical_solution']}"
    return text

# Create an instance of ZIPFIT
zip_fit_instance = ZIPFIT(
    source_dataset=source_dataset,
    target_dataset=target_dataset,
    target_load_fn=target_load_dataset_fn,
    target_parse_fn=target_parse_example_fn,
    k=100000,  
    output_file="top_k_sequences.jsonl",
    compression_algorithm='gzip'  # Change to 'lz4' if desired
)

# Run the ZIPFIT process
zip_fit_instance.run()
```
You can specify different compression algorithms. The ZIP-FIT paper uses gzip, however other compression algorithms like lz4 are faster. 

## Dev Install: ZIP-FIT + PyPantograph + Mathlib4 + Lean Setup
Below are comprehensive instructions for setting up everything in a **conda** environment named `zip_fit`, ensuring that:

- **Lean** is installed (via elan),
- **PyPantograph** is installed (via Poetry) and matches the same Lean version,
- **Mathlib4** is checked out at the corresponding Lean version,
- **ZIP-FIT** and optional vLLM are also installed, all in one place.

## 1. Create & Activate the `zip_fit` Conda Environment

```bash
conda create -n zip_fit python=3.11
conda activate zip_fit
conda activate zip_fit
```

### Install ZIP-FIT

If you have the ZIP-FIT repo at `~/ZIP-FIT`, install it in editable mode:

```bash
git clone git@github.com:stair-lab/ZIP-FIT.git
cd ZIP-FIT
git fetch origin
git branch -a
git checkout -b bm_dev origin/bm_dev
git branch -vv

pip install -e ~/ZIP-FIT
```

## 2. Install vLLM + EleutherAI Harness

If you want vLLM for flash attention, you can install it via `lm_eval[vllm]`:

```bash
pip install lm_eval[vllm]
# If you find version issues, pin it:
pip install vllm==0.6.4.post1

pip install antlr4-python3-runtime==4.11

# Quick check
pip list | grep lm_eval
pip list | grep vllm
pip list | grep antlr4
```

## 3. Install Lean (Via `elan`)

```bash
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y

# If ~/.bashrc doesn't have ~/.elan, add it:
cat ~/.bashrc | grep .elan
# If no output, do:
export PATH="$HOME/.elan/bin:$PATH"
echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Check versions
elan --version
lean --version
lake --version
```

**Note**: This sets up Lean for your user via elan, which will manage multiple Lean versions as needed.

## 4. Prepare PyPantograph
Note: **you need Lean4, Mathlib4 and PyPantograph/Pantograph to all agree on Lean version** e.g., **4.15.0** at the time of this writing.

<!-- ```bash
# Get the PyPantograph repo submodule if not present already:
if [ ! -d "PyPantograph" ] || ! grep -q "PyPantograph" .gitmodules || ! grep -q "submodule.*PyPantograph" .git/config; then
   # Adds the submodule (creates/updates .gitmodules automatically)
   git submodule add git@github.com:lenianiva/PyPantograph.git
   # Reads .gitmodules and registers submodules in .git/config (does not clone/update them)
   git submodule init  
   # Fetches latest commits from remote for submodules, including nested ones
   git submodule update --recursive --remote 
else
    # If it's already a submodule, just update all submodules recursively
    git submodule update --recursive --remote
fi
```

If the previous fails (eg git submodules are complicated),
then you can instead git clone it: -->
```bash
git clone --recurse-submodules git@github.com:lenianiva/PyPantograph.git
cd PyPantograph
# Initialize and update all submodules (including nested ones) recursively
git submodule update --init --recursive
git pull
```

### 4A. Ensure PyPantograph & Submodule Are Lean 4.15.0

PyPantograph has a `src/` submodule that also pins a Lean version. Confirm it’s `4.15.0`:

```bash
# either cd ~/ZIP-FIT/PyPantograph or cd ~/PyPantograph
cd PyPantograph
cat src/lean-toolchain
# Expect: leanprover/lean4:v4.15.0
```

If it’s correct, proceed. If not, pull the latest or check out the branch that uses 4.15.0:

```bash
git pull
git submodule update --init --recursive
# Then re-check src/lean-toolchain
cat src/lean-toolchain
```

Test PyPantograph server:
```bash
python -m pantograph.server
```

## 5. Install Poetry

Check if you have poetry:
```bash
poetry
```

### (Optional)

We’ll install Poetry **inside** the `zip_fit` environment (so we don’t leave conda):

```bash
pip install poetry
which poetry
poetry --version
```
If you prefer a separate Python env just for Poetry, see the commented lines below, but typically you can keep it simple by installing Poetry in `zip_fit`.

#### Install Poetry in it's seperate python env outside your current env
Remark: Installing Poetry in a separate Python environment prevents conflicts between Poetry’s dependencies and those of your Conda or project environments, ensuring a stable and isolated package management experience.

5. Create a Separate Python Env Just for Poetry
From any shell (you can leave zip_fit or open a new terminal):

```bash
mkdir -p $HOME/.virtualenvs
export VENV_PATH=$HOME/.virtualenvs/venv_for_poetry
export PATH="$VENV_PATH/bin:$PATH"

python3 -m venv $VENV_PATH
$VENV_PATH/bin/pip install -U pip setuptools
$VENV_PATH/bin/pip install poetry

poetry

# Make it permanent
cat ~/.bashrc | grep poetry
echo 'export VENV_PATH=$HOME/.virtualenvs/venv_for_poetry' >> ~/.bashrc
echo 'export PATH="$VENV_PATH/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```
Note: After this, which poetry may show `$HOME/.virtualenvs/venv_for_poetry/bin/poetry`.
If it hijacks your shell’s Python, you can open a new shell and re-activate zip_fit when needed.

## 6. Install PyPantograph into `zip_fit`

1. **Stay** in the [`PyPantograph](https://github.com/stanford-centaur/PyPantograph)` folder (and in the `zip_fit` env).  
2. **Configure Poetry** so it doesn’t create an extra venv:
   ```bash
   cd ~/PyPantograph
   poetry config virtualenvs.create false
   ```
3. **Install**:
   ```bash
   poetry build
   poetry install
   ```
4. **Check** you’re still in `zip_fit`:
   ```bash
   which python
   # Should be something like ~/miniconda/envs/zip_fit/bin/python
   ```
5. **Verify**:
   ```bash
   poetry show
   # or
   pip list | grep pantograph
   ```

## 7. Ensure Mathlib4 Matches Lean 4.15.0

PyPantograph’s submodule is pinned to Lean 4.15.0. If you want to import `Mathlib` inside PyPantograph, your local Mathlib4 **must** be the same Lean version.

1. **Clone or go to** your `mathlib4` folder:
   ```bash
   cd ~
   git clone https://github.com/leanprover-community/mathlib4.git
   cd mathlib4
   ```
2. **Check out** the branch for Lean 4.15.0:
   ```bash
   git fetch --all
   git checkout releases/v4.15.0
   cat lean-toolchain
   # → leanprover/lean4:v4.15.0
   ```
3. ** Speed Up** with cache:
   ```bash
   lake exe cache get
   ```
   This fetches pre-built .olean files for that commit if they exist.

4. (*optional*) If needed, do a **local build**:
   ```bash
   lake clean
   rm -rf .lake/build
   lake update
   lake build
   ```

Now, Mathlib4 is on Lean 4.15.0, matching PyPantograph.

## 8. Final Verification

### 8A. Check Lean Versions

```bash
poetry show
# or
pip list | grep pantograph

lean --version
# Should say 4.15.0 if you're in a folder overridden by elan or if conda isn't overshadowing anything.

cd ~/mathlib4
cat lean-toolchain
# → 4.15.0

cd ~/PyPantograph/src
cat lean-toolchain
# → 4.15.0

elan --version
lean --version
lake --version

python -m pantograph.server
```

All must match for a successful import.

### 8B. Minimal PyPantograph Test

```bash
cd ~
python -c "from pantograph import Server;import os; \
           s = Server(imports=['Mathlib'], project_path=os.path.expanduser('~/mathlib4')); \
           print('Pantograph server started!')"

python -m pantograph.server
```
- If you see **“Pantograph server started!”** and no “invalid header” errors, you’re good!

## 9. (Optional) Re-check ZIP-FIT

Return to your ZIP-FIT repo:

```bash
cd ~/ZIP-FIT
pip install -e .

python -c "import zip_fit; print('zip_fit imported successfully!')"
```
ref, o1 pro:https://chatgpt.com/g/g-p-6789a51d52308191917d7bc04225a117-zip-fit/c/67996aa2-4d28-8001-a095-b54f4555676a?model=o1-pro

## Citation Information
Paper: <https://arxiv.org/abs/2410.18194>
```
@article{obbad2024zipfit,
  author = {Elyas Obbad and Iddah Mlauzi and Brando Miranda and Rylan Schaeffer and Kamal Obbad and Suhana Bedi and Sanmi Koyejo},
  title = {ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment},
  year = {2024},
  journal = {arXiv preprint arXiv:2410.18194},
}
```
