#!/usr/bin/env python3
# Filename: _lean_pass_k.py

"""
Demonstration of Pass @ k for Lean 4 compilation using PyPantograph and GPT-2.

Key Concepts:
  - N = number of test problems (docstrings)
  - k = the "top-k" threshold in Pass @ k
  - We generate 'num_samples' solutions for each problem using GPT-2,
    then check how many of those compile in Lean 4.

Prerequisites:
  - Lean 4 + lake installed
  - PyPantograph installed (per instructions)
  - transformers & torch for GPT-2 generation
  - numpy for pass@k combinatorial calculation

Usage:
  1. Ensure Lean and PyPantograph are installed (see instructions).
  2. pip install transformers torch numpy (or poetry add them).
  3. Run: python lean_pass_k.py
"""

import numpy as np
from typing import List
from transformers import pipeline, set_seed
from pantograph.server import Server


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """
    Computes pass@k using the combinatorial formula from the HumanEval approach.

    pass@k = 1 - product_{i=0 to k-1}((num_samples - num_correct - i) / (num_samples - i))

    Args:
        num_samples (int): total completions generated
        num_correct (int): number of completions that are correct (compile)
        k (int): top-k threshold

    Returns:
        float: pass@k value in [0.0, 1.0]
    """
    if num_correct == 0:
        return 0.0
    if num_correct >= num_samples:
        return 1.0

    prob_none_correct = 1.0
    for i in range(k):
        numerator = (num_samples - num_correct - i)
        denominator = (num_samples - i)
        if numerator < 0:
            # means we've run out of 'incorrect' samples to fill
            return 1.0
        prob_none_correct *= (numerator / denominator)

    return 1.0 - prob_none_correct


def generate_snippets(prompt: str, num_samples: int = 5, max_length: int = 64) -> List[str]:
    """
    Generates GPT-2 completions (toy Lean 4 code) from a textual prompt.

    Args:
        prompt (str): The user prompt or docstring
        num_samples (int): Number of completions to generate
        max_length (int): Max token length of each completion

    Returns:
        List[str]: List of code snippet strings
    """
    set_seed(42)  # for reproducibility in this toy example
    generator = pipeline("text-generation", model="gpt2")
    outputs = generator(
        prompt,
        num_return_sequences=num_samples,
        max_length=max_length,
        do_sample=True,
        temperature=1.0,
    )
    return [o["generated_text"] for o in outputs]


def check_lean_compiles(snippet: str, server: Server) -> bool:
    """
    Checks if a snippet compiles in Lean 4 via Pantograph's load_sorry.

    Args:
        snippet (str): Proposed Lean 4 code snippet
        server (Server): PyPantograph server instance

    Returns:
        bool: True if snippet compiles, False otherwise
    """
    try:
        server.load_sorry(snippet)
        return True
    except Exception:
        return False


def run_pass_k_eval(
    docstrings: List[str],
    server: Server,
    k: int = 5,
    num_samples: int = 10
) -> float:
    """
    Runs pass@k evaluation for the given docstrings, generating `num_samples` completions each.

    Args:
        docstrings (List[str]): The N test problem statements
        server (Server): PyPantograph server
        k (int): top-k threshold for pass@k
        num_samples (int): how many completions to generate per docstring

    Returns:
        float: the average pass@k across the docstrings
    """
    pass_vals = []
    for idx, statement in enumerate(docstrings):
        completions = generate_snippets(prompt=statement, num_samples=num_samples)
        num_correct = sum(check_lean_compiles(c, server) for c in completions)
        p_at_k = estimate_pass_at_k(num_samples, num_correct, k)
        pass_vals.append(p_at_k)

        print(f"\n[{idx+1}/{len(docstrings)}] Problem: {repr(statement)}")
        print(f"  #Generated = {num_samples}, #Correct = {num_correct}, pass@{k} = {p_at_k:.3f}")

    return float(np.mean(pass_vals)) if pass_vals else 0.0


def main() -> None:
    """
    Main function to demonstrate pass@k with N=10 docstrings and k=5.

    GPT-2 won't produce valid Lean code, but we illustrate the pipeline:
      docstrings -> GPT-2 -> compile check -> pass@k calculation
    """
    # 1) Create a PyPantograph server with default Lean environment
    server = Server()

    # 2) N=10 docstrings (toy statements). In practice, these would be formal or near-formal tasks.
    #    Real Lean tasks might already have "theorem ..." etc.
    docstrings = [
        f"theorem lemma{i} : 2 + {i} = {i} + 2 := by"
        for i in range(1, 11)
    ]
    # docstrings now has length N=10

    # 3) Evaluate pass@k with k=5, generating 10 completions per docstring
    pass_k_value = run_pass_k_eval(docstrings, server, k=5, num_samples=10)

    print(f"\n==== Final Average Pass@5 over N=10 tasks: {pass_k_value:.3f} ====\n")


if __name__ == "__main__":
    main()
