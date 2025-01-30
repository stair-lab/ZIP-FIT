# lean_pass_k_unbiased.py
#!/usr/bin/env python3
"""
Implements the unbiased Pass@k estimator from the Codex paper [Chen et al., 2021],
and references the official OpenAI HumanEval code:
    https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py

--------------------------------------------------------------------------------
According to the Codex paper (and the HumanEval repository),
the unbiased pass@k for a single problem is:

  pass@k = 1 - ( (n-c choose k) / (n choose k) )

where:
 - n: total number of samples generated for a problem
 - c: number of correct samples
 - k: top-k threshold
 - (n choose k) denotes the binomial coefficient

If (n - c) < k, pass@k = 1.0 (since we can't fill all k slots with incorrect).
If c = 0, pass@k = 0.0. If c >= n, pass@k = 1.0.

In the official HumanEval code, they use a product form for numerical stability,
similar to:

  pass@k = 1 - product_{i=0}^{k-1} [ (n-c - i) / (n - i) ]

or equivalently:

  pass@k = 1 - prod(
      1.0 - k / np.arange(n-c+1, n+1)
  )

Both yield the same final result for pass@k.

This script provides:
1) pass_at_k(...) for a single problem,
2) average_pass_at_k(...) for multiple problems,
3) A small example usage section at the bottom.

References:
 - "Evaluating Large Language Models Trained on Code" (Chen et al., 2021)
 - Official code snippet from OpenAI's HumanEval: 
   https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py
"""

import numpy as np
from typing import List

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Computes the unbiased pass@k for a single problem using the
    Codex/HumanEval approach.

    pass@k = 1 - ( (n - c choose k) / (n choose k) ).

    Args:
        n (int): Number of total samples for the problem.
        c (int): Number of correct samples among those n.
        k (int): The 'top-k' threshold.

    Returns:
        float: pass@k in the range [0, 1].

    Edge Cases:
      - If c == 0, pass@k = 0.0
      - If c >= n, pass@k = 1.0
      - If (n - c) < k, pass@k = 1.0 (not enough "incorrect" samples to fill k)
    """
    # 1. No correct samples => pass@k=0
    if c == 0:
        return 0.0
    # 2. All correct => pass@k=1
    if c >= n:
        return 1.0
    # 3. If we can't fill k slots with incorrect, pass@k=1
    if (n - c) < k:
        return 1.0

    # 4. Otherwise, compute the product form
    # pass@k = 1 - ( (n-c choose k)/(n choose k) )
    #        = 1 - Π_{i=0..k-1} [ (n-c-i)/(n-i) ]
    product_term = 1.0
    for i in range(k):
        product_term *= (float(n - c - i) / float(n - i))
    return 1.0 - product_term


def average_pass_at_k(all_counts: List[int], n: int, k: int) -> float:
    """
    Computes the *average* pass@k across multiple problems, each with
    the same 'n' but possibly different 'c' (# of correct solutions).

    Args:
        all_counts (List[int]): A list of c-values (correct sample counts),
                                one for each problem.
        n (int): Number of total samples per problem.
        k (int): The 'top-k' threshold for pass@k.

    Returns:
        float: Mean pass@k over the entire list of problems.
    """
    vals = []
    for c in all_counts:
        vals.append(pass_at_k(n, c, k))
    return float(np.mean(vals)) if vals else 0.0


def _example_usage() -> None:
    """
    Demonstrates pass_at_k(...) on a few scenarios and compares results.
    """
    # Single problem: n=10, c=3, k=5
    val_1 = pass_at_k(n=10, c=3, k=5)
    print(f"(Single) n=10, c=3, k=5 => pass@5 = {val_1:.3f}")

    # Suppose we have 3 problems, each with 10 samples, and their
    # correct counts are: [3, 0, 6]
    c_list = [3, 0, 6]
    val_2 = average_pass_at_k(c_list, 10, 5)
    print(f"(Multi) pass@5 across c_list={c_list} => {val_2:.3f}")

    # Some edge cases
    print("Edge Cases:")
    print("  c=0 =>", pass_at_k(5, 0, 3))
    print("  c=n =>", pass_at_k(5, 5, 3))
    print("  (n-c)<k =>", pass_at_k(5, 2, 4))


if __name__ == "__main__":
    _example_usage()
