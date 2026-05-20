---
dataset_info:
  features:
  - name: year
    dtype: string
  - name: id
    dtype: string
  - name: problem
    dtype: string
  - name: solution
    dtype: string
  - name: answer_type
    dtype: string
  - name: source
    dtype: string
  - name: type
    dtype: string
  - name: original_problem
    dtype: string
  - name: original_solution
    dtype: string
  - name: variation
    dtype: int64
  splits:
  - name: full_eval
    num_examples: 522
  - name: test
    num_examples: 372
  - name: val
    num_examples: 150
  download_size: 560892
  dataset_size: 1184885
configs:
- config_name: default
extra_gated_prompt: 'By requesting access to this dataset, you agree to cite the following
  works in any publications or projects that utilize this data:

  - Putnam-AXIOM dataset: @article{putnam_axiom2025, title={Putnam-AXIOM: A Functional
  and Static Benchmark for Measuring Higher Level Mathematical Reasoning}, author={Aryan
  Gulati and Brando Miranda and Eric Chen and Emily Xia and Kai Fronsdal and Bruno
  de Moraes Dumont and Sanmi Koyejo}, journal={39th International Conference on Machine Learning (ICML 2025)}, year={2025}, 
  note={Preprint available at: https://openreview.net/pdf?id=YXnwlZe0yf}} '
---

# Putnam AXIOM Dataset (ICML 2025 Version)

**Note: for questions, feedback, bugs, etc. please [open a Huggingface discussion here](https://huggingface.co/datasets/Putnam-AXIOM/putnam-axiom-dataset-ICML-2025-522/discussions).**

## Dataset Summary

The [**Putnam AXIOM**](https://openreview.net/pdf?id=YXnwlZe0yf) dataset is designed for evaluating large language models (LLMs) on advanced mathematical reasoning skills. It is based on challenging problems from the Putnam Mathematical Competition. This version contains 522 original problems prepared for the ICML 2025 submission.

This dataset includes:
- **Full Evaluation Set (522 problems)**: Complete set of problems
- **Test Set (372 problems)**: Set used for testing
- **Validation Set (150 problems)**: Set used for validation/development

Each problem includes:
- Problem statement
- Solution
- Original problem (where applicable)
- Answer type (e.g., numerical, proof)
- Source and type of problem (e.g., Algebra, Calculus, Geometry)
- Year (extracted from problem ID)

## Supported Tasks and Leaderboards

- **Mathematical Reasoning**: Evaluate mathematical reasoning and problem-solving skills.
- **Language Model Benchmarking**: Use this dataset to benchmark performance of language models on advanced mathematical questions.

## Languages

The dataset is presented in **English**.

## Dataset Structure

### Data Fields

- **year**: The year of the competition (extracted from the problem ID).
- **id**: Unique identifier for each problem.
- **problem**: The problem statement.
- **solution**: The solution or explanation for the problem.
- **answer_type**: The expected type of answer (e.g., numerical, proof).
- **source**: The origin of the problem (Putnam).
- **type**: A description of the problem's mathematical topic (e.g., "Algebra Geometry").
- **original_problem**: Original form of the problem, where applicable.
- **original_solution**: Original solution to the problem, where applicable.
- **variation**: Flag for variations (0 for all problems in this dataset as these are not variations).

### Splits

| Split       | Description                            | Number of Problems |
|-------------|----------------------------------------|--------------------|
| `full_eval` | Complete set of 522 problems           | 522                |
| `test`      | Test split                             | 372                |
| `val`       | Validation/development split           | 150                |

## Dataset Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Putnam-AXIOM/putnam-axiom-dataset-ICML-2025-522")

# Access each split
full_eval = dataset["full_eval"]
test = dataset["test"]
val = dataset["val"]

# Example usage: print the first problem from the full evaluation set
print(full_eval[0])
```
    
## Citation
If you use this dataset, please cite it as follows:

```bibtex
@article{putnam_axiom2025,
  title={Putnam-AXIOM: A Functional and Static Benchmark for Measuring Higher Level Mathematical Reasoning},
  author={Aryan Gulati and Brando Miranda and Eric Chen and Emily Xia and Kai Fronsdal and Bruno de Moraes Dumont and Sanmi Koyejo},
  journal={39th International Conference on Machine Learning (ICML 2025)},
  year={2025},
  note={Preprint available at: https://openreview.net/pdf?id=YXnwlZe0yf}
}
```

## License

This dataset is licensed under the Apache 2.0.

Last updated: May 22, 2024 