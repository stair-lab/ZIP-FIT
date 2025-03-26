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
pip install --upgrade pip
```

### Install ZIP-FIT

If you have the ZIP-FIT repo at `~/ZIP-FIT`, install it in editable mode:

```bash
# Clone the ZIP-FIT repository (SSH) and recursively initialize any submodules
git clone --recurse-submodules git@github.com:stair-lab/ZIP-FIT.git
# Change directory into the newly cloned ZIP-FIT repo
cd ZIP-FIT
# Fetch the latest changes and branches from the remote
git fetch origin
# Show all local and remote branches
git branch -a
# Create and switch to the 'bm_dev' branch, tracking remote 'origin/bm_dev'
git checkout -b bm_dev origin/bm_dev
# Display branch details, including local/remote tracking and commit differences
git branch -vv
# Install ZIP-FIT in editable mode, so local changes are reflected immediately
conda activate zip_fit
pip install -e ~/ZIP-FIT
```

## 2. Install vLLM + EleutherAI Harness

If you want vLLM for flash attention, you can install it via `lm_eval[vllm]`:

```bash
# If the bellow are not already there:
pip install lm_eval[vllm]
# If you find version issues, pin it:
pip install vllm==0.6.4.post1

pip install antlr4-python3-runtime==4.11

# Quick check
pip list | grep lm_eval
pip list | grep vllm
pip list | grep antlr4
# Expected output:
# lm_eval                           0.4.8
# vllm                              0.6.4.post1
# antlr4-python3-runtime            4.11.0
```

## 3. Install Lean (Via `elan`) & Mathlib4 with version you need for PyPantograph

Useful chat: https://chatgpt.com/c/67da423c-e384-8001-a934-eadf6aca7b11 

Note: PyPantograph's submodule is pinned to Lean 4.15.0. 
If you want to import `Mathlib` inside PyPantograph, 
your local Mathlib4 **must** be the same Lean version.

**Note**: This sets up Lean for your user via elan, which will manage multiple Lean versions as needed.
```bash
# Install elan (Lean’s toolchain manager, manages diff Lean version) which will install Lean and Lake.
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y

# If ~/.bashrc doesn't have ~/.elan, add it:
cat ~/.bashrc | grep .elan
# If no output, do:
export PATH="$HOME/.elan/bin:$PATH"
echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# (elan is Lean's toolchain manager for installing and managing Lean versions) 
elan --version
# elan 4.0.0 (bb75b50d2 2025-01-30)

# Install the 4.15.0 toolchain via elan
elan toolchain install leanprover/lean4:v4.15.0

# Set your default toolchain to 4.15.0 (so you don’t re-download another version later)
elan default leanprover/lean4:v4.15.0
# TODO: how robust this elan default

# Verify the installed Lean version
lean --version  
# Lean (version 4.15.0, x86_64-unknown-linux-gnu, commit 11651562caae, Release)

# Check Lake version (Lake is Lean’s build system and package manager) "Lean's pip"
lake --version
# Lake version 5.0.0-1165156 (Lean version 4.15.0)

# Run Lean within the Lake environment (which sets the search path to include mathlib) to test that mathlib4 is installed:
echo 'def main : IO Unit := IO.println "Lean works!"' | lake env lean --run --stdin
## doesn't work idk why echo 'def main : IO Unit := IO.println "Lean works!"' | lean --run --stdin
# Sample output: Lean works!


# Ensure Mathlib4 Matches Lean 4.15.0
# Clone or go to your mathlib4 or mathlib4_15_0 folder
cd ~

# Download mathlib4 name it mathlib4_15_0 repository from GitHub
git clone --branch releases/v4.15.0 https://github.com/leanprover-community/mathlib4.git mathlib4_15_0

# Change directory into the cloned mathlib4_15_0 repository
cd ~/mathlib4_15_0
# mv ~/mathlib4_15_0 $DFS/mathlib4_15_0
# ln -s $DFS/mathlib4_15_0 $HOME/mathlib4_15_0

# Check out the branch for Lean 4.15.0:
# Fetch all remote references from every configured remote
git fetch --all

# Switch to the release tag matching PyPantograph
git checkout releases/v4.15.0
git brach # check branch
cat lean-toolchain
# leanprover/lean4:v4.15.0

# Fetch cache .olean files for the current commit to speed up the build process
lake exe cache get
# No files to download
# Decompressing 5826 file(s)
# Unpacked in 8795 ms
# Completed successfully!

# Run the mathlib4 test suite in the mathlib4_15_0 directory to verify everything works correctly
# Warning: Very slow
(cd ~/mathlib4_15_0 && lake test)
# ⣿ [?/?] Computing build job
cd ~/mathlib4_15_0 && lake build
# Build completed successfully.
# real    0m16.570s
# user    0m6.518s
# sys     0m12.284s

# TODO: one liner test for mathlib
#   --> https://github.com/stanford-centaur/PyPantograph/issues/89
#   --> https://proofassistants.stackexchange.com/questions/4848/quick-one-liner-to-verify-mathlib4-installation-eg-with-lean-4-15-0
#   --> https://chatgpt.com/c/67da423c-e384-8001-a934-eadf6aca7b11
# since there is no lean project specified here, this cannot be tested but it woulb like:
cd ~/veribench/lean_src_proj
echo -e 'import Mathlib.Topology.Basic\n#check TopologicalSpace' | lake env lean --stdin
# Output: TopologicalSpace.{u} (X : Type u) : Type u

# For a pypantograph test go bellow, search Server( in this file
```

## 4. Prepare PyPantograph before Pip Installing it
Note: **you need Lean4, Mathlib4 and PyPantograph/Pantograph to all agree on Lean version** e.g., **4.15.0** at the time of this writing.

<!-- ```bash
# Get the PyPantograph repo submodule if not present already:
if [ ! -d "PyPantograph" ] || ! grep -q "PyPantograph" .gitmodules || ! grep -q "submodule.*PyPantograph" .git/config; then
   # Adds the PyPantograph submodule on the right branch (updates .gitmodules automatically)
   git submodule add -b main git@github.com:lenianiva/PyPantograph.git
   # Reads .gitmodules and registers submodules in .git/config (does not clone/update them)
   git submodule init  
   # Fetches latest commits from remote for submodules, including nested ones (always gets latest submodule commits)
   git submodule update --init --recursive --remote 
   # Inits & updates all submodules (including nested ones) to their tracked commits, not the latest remote, (reproducible builds, common case)
   # git submodule update --init --recursive --remot
else
    # If it's already a submodule, just update all submodules recursively
    git submodule update --recursive --remote
fi
```
Note: Leni suggests the `main` branch for PyPantograph: https://github.com/stanford-centaur/PyPantograph/issues/84 but the `dev` branch for `Pantograph` (a depedency of PyPantograph, though that isn't something we need to hopefully worry about).
Note: If the previous fails (eg git submodules are complicated),
you can instead git clone it and see if it works: -->

We recommend using a direct clone of PyPantograph rather than adding it as a submodule to avoid git submodule complexity:
```bash
# Clones the repository and fetches submodules recursively
cd ~
git clone --recurse-submodules git@github.com:lenianiva/PyPantograph.git
# Enters the repository directory
cd ~/PyPantograph
# Pulls the latest code changes from the repository's remote branch
git pull
# Initializes and updates every level of submodules from their remote sources
git submodule update --init --recursive --remote
```
## 5. Install Poetry
We need Poetry to install PyPantograph because it uses it to build it.

Check if you have poetry:
```bash
# Check if you have poetry
which poetry
# Example output: /lfs/skampere1/0/brando9/.virtualenvs/venv_for_poetry/bin/poetry
```

### (Optional)

We'll install Poetry **inside** the `zip_fit` environment (so we don't leave conda):
```bash
# [Optional] Install Poetry within the current conda environment
pip install poetry
# [Optional] Check which Poetry is being used
which poetry
# [Optional] Check Poetry version
poetry --version
```
If you prefer a separate Python env just for Poetry, see the commented lines below, but typically you can keep it simple by installing Poetry in `zip_fit`.

#### Install Poetry in it's seperate python env outside your current env
Remark: Installing Poetry in a separate Python environment prevents conflicts between Poetry's dependencies and those of your Conda or project environments, ensuring a stable and isolated package management experience.

5. Create a Separate Python Env Just for Poetry
From any shell (you can leave zip_fit or open a new terminal):

```bash
mkdir -p $HOME/.virtualenvs
export VENV_PATH=$HOME/.virtualenvs/venv_for_poetry
export PATH="$VENV_PATH/bin:$PATH"

python3 -m venv $VENV_PATH
$VENV_PATH/bin/pip install -U pip setuptools
$VENV_PATH/bin/pip install poetry

which poetry

# Make it permanent
cat ~/.bashrc | grep poetry
echo 'export VENV_PATH=$HOME/.virtualenvs/venv_for_poetry' >> ~/.bashrc
echo 'export PATH="$VENV_PATH/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```
Note: After this, which poetry may show `$HOME/.virtualenvs/venv_for_poetry/bin/poetry`.
If it hijacks your shell's Python, you can open a new shell and re-activate zip_fit when needed.

## 6. Install PyPantograph into `zip_fit`

1. **Stay** in the [`PyPantograph`](https://github.com/stanford-centaur/PyPantograph)` folder (and in the `zip_fit` env).  
2. **Configure Poetry & Install PyPantograph** so it doesn't create an extra venv:
```bash
# Change PyPantograph dir & check lean-toolchain, if wrong git pull & update submodules recursively & remote
conda activate zip_fit
cd ~/PyPantograph
cat src/lean-toolchain
# Expect: leanprover/lean4:v4.15.0

# Configure Poetry to install packages in the current environment instead of creating a new virtual environment
poetry config virtualenvs.create false

# Install PyPantograph
# Check you have somewhere poetry
which poetry
# Build the distributable package for PyPantograph using Poetry
poetry build
# Install the package in the current environment using Poetry
poetry install

# **Verify Install Worked**:
# List all installed packages via Poetry to confirm PyPantograph is installed
poetry show
# Alternatively, list packages filtered by 'pantograph' using pip
pip list | grep pantograph

# Run the PyPantograph server
pip install pexpect # force install pexpect (sometimes needed)
python3 -m pantograph.server
# pip install pexpect # - hack if prev fails maybe this works
# (zip_fit) brando9@mercury2~ $ python -m pantograph.server
# <frozen runpy>:128: RuntimeWarning: 'pantograph.server' found in sys.modules after import of package 'pantograph', but prior to execution of 'pantograph.server'; this may result in unpredictable behaviour
# ...........
# ----------------------------------------------------------------------
# Ran 11 tests in 7.224s

# OK

# Basic PyPantograph import
python -c "from pantograph import Server; server = Server(imports=['Init']); print(server)"

# PyPantograph Import with local Mathlib
python -c "import os; from pantograph import Server; server = Server(imports=['Mathlib', 'Init'], project_path=os.path.expanduser('~/mathlib_4_15_0_lfs'), timeout=300); print(server)"

# Run basic lean4 test with mathlib4
python ~/ZIP-FIT/zip_fit/metrics/tests/basic_lean4_tests.py
```

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

## 11. Troubleshooting

If you encounter version mismatches or PyPantograph errors:

1. **Version Mismatch**: Ensure all three components (PyPantograph, Pantograph submodule, mathlib4_15_0) are using Lean v4.15.0
```bash
# Check PyPantograph's submodule version
cd ~/PyPantograph/src && cat lean-toolchain
# Check mathlib4_15_0 version
cd ~/mathlib4_15_0 && cat lean-toolchain
```

<!-- 2. **PyPantograph Import Errors**: If you can't import `pantograph`, reinstall it:
```bash
cd ~/PyPantograph
pip install -e .
``` -->

3. **Server Initialization Errors**: If the server fails to start, try rebuilding the components:
```bash
# Rebuild PyPantograph's Lean components
cd ~/PyPantograph/src
lake clean
lake build
# Rebuild PyPantograph Python package
cd ~/PyPantograph
poetry build
poetry install
```

## (*Optional*) Clean (TODO needs checking and clarifications)
```bash
# Clean previous build artifacts using lake's built-in clean command
lake clean
# Remove the entire .lake/build directory to ensure a fresh start
rm -rf .lake/build
# Update the lake project, fetching any new dependencies or configuration changes
lake update
# Build the project using lake to compile the Lean files and generate olean files
lake build
```
Now, Mathlib4 is on Lean 4.15.0, matching PyPantograph.

## To deinitialize the PyPantograph git submodule

```bash
cd ~/ZIP-FIT

# 2) Deinitialize the submodule (removes it from .git/config)
git submodule deinit -f PyPantograph

# 3) Remove the submodule from Git's index (this unregisters the submodule)
git rm -f PyPantograph

# 4) Remove the actual submodule directory from disk
rm -rf PyPantograph/

# 5) Delete leftover metadata for the submodule
rm -rf .git/modules/PyPantograph

# 6) Remove the entire .gitmodules file (if ZIP-FIT is your only submodule)
rm -f .gitmodules

# 7) Commit the changes so the repository no longer references the submodule
git commit -am "Completely remove PyPantograph submodule"
```