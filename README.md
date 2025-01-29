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

## Dev Install
ref: chat with install: https://chatgpt.com/share/67996e5c-9948-8001-bc44-9faed3fa3cf8

```bash
conda create -n zip_fit python=3.11
conda activate zip_fit
pip install -e ~/ZIP-FIT
```

Install vLLM (intalling it by installing lm-harness seems to work well with good flash attn):
```bash
# Install lm-harness (https://github.com/EleutherAI/lm-evaluation-harness)
pip install lm_eval[vllm]
# pip install -e ".[vllm]"
pip install antlr4-python3-runtime==4.11
# to check installs worked do (versions and paths should appear)
pip list | grep lm_eval
pip list | grep vllm
pip list | grep antlr4
```

Install Lean:
```bash
# install elan
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y

# Cat .bashrc to see if to change path for elan
cat ~/.bashrc | grep .elan
# if you don't see .elan then run the code bellow to add it to your .bashrc
export PATH="$HOME/.elan/bin:$PATH"
echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Test the installation
elan --version
lean --version
lake --version
```

Install PyPantograph (our Python Interface to Lean 4):
```bash
git clone --recurse-submodules git@github.com:lenianiva/PyPantograph.git
cd PyPantograph
git submodule update --init --recursive
```

Install poetry: 
```bash
# Option1: instal poetry in your zip_fit
# Instead of creating a separate Poetry venv (like the official Poetry docs often do), we’ll simply put Poetry in the zip_fit environment so that we never leave it.
pip install poetry
which poetry
poetry --version

# Option2: in a seperate Python env
mkdir -p $HOME/.virtualenvs
export VENV_PATH=$HOME/.virtualenvs/venv_for_poetry
export PATH="$VENV_PATH/bin:$PATH"
python3 -m venv $VENV_PATH
$VENV_PATH/bin/pip install -U pip setuptools
$VENV_PATH/bin/pip install poetry
# Only if not in your .bashrc already
bash
echo 'export VENV_PATH=$HOME/.virtualenvs/venv_for_poetry' >> ~/.bashrc
echo 'export PATH="$VENV_PATH/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
which poetry
# You might need to kill your current bash session and restart it if suddenly your in the poetry env
which python
# if it's poetry then kill bash and start a new one and then reactivate zip_fit
conda activate zip_fit
```

Install PyPantograph to current conda `zip_fit` env without breaking things:
```bash
cd PyPantograph
# Configure Poetry to install to the current environment
poetry config virtualenvs.create false
# Now install PyPantograph
poetry install

# Confirm you are in zip_fit for sure
which python

# Show you install PyPantograph
poetry show
# or
pip list | grep pantograph

# Check PyPantograph works
lean --version
lake --version
python -m pantograph.server

# Make sure Zip fit dependencies work 
cd ~/ZIP-FIT
pip install -e .

# Check zip-fit & pantograph
python -c "import zip_fit; print('zip_fit is installed')"
python -c "import pantograph; print('PyPantograph imported')"
```
ref: https://chatgpt.com/c/67996aa2-4d28-8001-a095-b54f4555676a

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
