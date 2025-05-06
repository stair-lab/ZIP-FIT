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
  - name: validation
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
  note={Preprint available at: https://openreview.net/pdf?id=YXnwlZe0yf}}
  
  - ZIP-FIT: @article{obbad2024zipfit, title={ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment}, 
  author={Elyas Obbad and Iddah Mlauzi and Brando Miranda and Rylan Schaeffer and Kamal Obbad and Suhana Bedi and Sanmi Koyejo}, 
  journal={arXiv preprint arXiv:2410.18194}, year={2024}}'
---

# Putnam-AXIOM Dataset for ZIP-FIT Experiments

This dataset contains splits of the Putnam-AXIOM dataset specifically created for experiments with the ZIP-FIT framework. These splits were derived from the original [Putnam-AXIOM dataset](https://huggingface.co/datasets/Putnam-AXIOM/putnam-axiom-dataset-ICML-2025-522).

## Dataset Summary

This dataset was created for the experiments in the [ZIP-FIT paper](https://arxiv.org/abs/2410.18194). It contains the original 522 Putnam Mathematical Competition problems from the Putnam-AXIOM dataset, divided into validation and test splits for evaluating data selection methods in the context of mathematical reasoning.

Each problem includes:
- Problem statement
- Solution
- Original problem (where applicable)
- Answer type (e.g., numerical, proof)
- Source and type of problem (e.g., Algebra, Calculus, Geometry)
- Year (extracted from problem ID)

## Relationship to ZIP-FIT

The [ZIP-FIT framework](https://github.com/stair-lab/ZIP-FIT) (Embedding-Free Data Selection via Compression-Based Alignment) uses compression techniques to select high-quality training data that aligns well with target distributions. These Putnam-AXIOM splits were used to evaluate the effectiveness of ZIP-FIT for mathematical reasoning tasks.

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

| Split         | Description                            | Number of Problems |
|---------------|----------------------------------------|--------------------|
| `full_eval`   | Complete set of 522 problems           | 522                |
| `test`        | Test split                             | 372                |
| `validation`  | Validation/development split           | 150                |

## Dataset Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("zipfit/Putnam-AXIOM-for-zip-fit-splits")

# Access each split
full_eval = dataset["full_eval"]
test = dataset["test"]
validation = dataset["validation"]

# Example usage: print the first problem from the test set
print(test[0])
```
    
## Citation
If you use this dataset, please cite both the original Putnam-AXIOM paper and the ZIP-FIT paper:

```bibtex
@article{putnam_axiom2025,
  title={Putnam-AXIOM: A Functional and Static Benchmark for Measuring Higher Level Mathematical Reasoning},
  author={Aryan Gulati and Brando Miranda and Eric Chen and Emily Xia and Kai Fronsdal and Bruno de Moraes Dumont and Sanmi Koyejo},
  journal={39th International Conference on Machine Learning (ICML 2025)},
  year={2025},
  note={Preprint available at: https://openreview.net/pdf?id=YXnwlZe0yf}
}

@article{obbad2024zipfit,
  author = {Elyas Obbad and Iddah Mlauzi and Brando Miranda and Rylan Schaeffer and Kamal Obbad and Suhana Bedi and Sanmi Koyejo},
  title = {ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment},
  year = {2024},
  journal = {arXiv preprint arXiv:2410.18194},
}
```

## ZIP-FIT Project

For more information on the ZIP-FIT framework, including code and documentation, visit the [ZIP-FIT GitHub repository](https://github.com/stair-lab/ZIP-FIT).

## License

This dataset is licensed under the Apache 2.0. 