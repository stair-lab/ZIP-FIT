# lean_pass_k_unbiased.py
#!/usr/bin/env python3
"""
Implements the unbiased Pass@k estimator from the Codex paper [Chen et al., 2021],
plus a demonstration of using PyPantograph to generate Lean 4 code (via GPT-2)
and evaluate how many snippets compile. We then compute Pass@k on these results.

References:
 - Official OpenAI HumanEval code, which uses the same binomial formula:
   https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py
 - "Evaluating Large Language Models Trained on Code" (Chen et al., 2021)

--------------------------------------------------------------------------------
Unbiased Pass@k:
~~~~~~~~~~~~~~~~
For a single problem, if we generate 'n' code completions and find 'c' correct,
the unbiased pass@k is:

    pass@k = 1 - ( (n-c choose k) / (n choose k) )

Edge Cases:
 - If c = 0 => pass@k=0
 - If c >= n => pass@k=1
 - If (n - c) < k => pass@k=1 (can't fill all k picks with incorrect samples)

Implementation uses a product form for numerical stability:
  pass@k = 1 - ∏_{i=0 to k-1} [ (n-c-i) / (n-i) ]

--------------------------------------------------------------------------------
Lean 4 Use Case:
~~~~~~~~~~~~~~~~
We integrate with PyPantograph to:
1. Prompt GPT-2 for Lean 4 code snippets (docstring => code).
2. Check if each snippet compiles in Lean 4 via `server.load_sorry`.
3. Count how many are correct (compilable).
4. Compute pass@k with the unbiased formula.

Prerequisites:
  - Lean 4 + lake installed
  - PyPantograph installed
  - transformers & torch for GPT-2 generation
  - numpy for pass@k combinatorial calculation

Usage:
  1. Ensure the above prerequisites are installed.
  2. Run: python lean_pass_k_unbiased.py
"""

import numpy as np
from typing import List
from transformers import pipeline, set_seed
from pantograph.server import Server


def seed_everything(seed: int = 42):
    """
    Seed Python, NumPy, and PyTorch for reproducibility.
    """
    import random
    import numpy as np
    from transformers import set_seed as hf_set_seed

    print(f"Setting random seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        hf_set_seed(seed)
    else:
        print("Warning: Transformers is only fully deterministic on GPU")


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute the unbiased pass@k for a single problem using the
    Codex/HumanEval approach.

    pass@k = 1 - ((n - c choose k)/(n choose k))
            = 1 - ∏_{i=0 to k-1}[(n-c-i)/(n-i)]

    Args:
        n (int): Total number of code snippets generated.
        c (int): Number of correct (compilable) snippets.
        k (int): "top-k" threshold.

    Returns:
        float: pass@k ∈ [0,1].
    """
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    if (n - c) < k:
        return 1.0

    product_term = 1.0
    for i in range(k):
        numerator = (n - c - i)
        denominator = (n - i)
        if numerator < 0:
            # We can't fill these k picks with incorrect
            return 1.0
        product_term *= (numerator / denominator)

    return 1.0 - product_term


def generate_snippets(prompt: str, num_samples: int = 5, max_length: int = 64) -> List[str]:
    """
    Generates Lean 4 code snippets from a textual prompt using GPT-2 (toy example).

    Args:
        prompt (str): The user prompt or docstring.
        num_samples (int): Number of completions to generate.
        max_length (int): Max token length of each completion.

    Returns:
        List[str]: List of code snippet strings.
    """
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
    Checks if the given Lean 4 code snippet compiles via PyPantograph.

    Args:
        snippet (str): Potential Lean 4 code snippet.
        server (Server): PyPantograph server instance.

    Returns:
        bool: True if snippet compiles successfully; else False.
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
    Runs pass@k evaluation over a list of "docstrings" (Lean 4 tasks).

    Steps:
      1) Generate 'num_samples' Lean code snippets via GPT-2 for each docstring.
      2) Check how many snippets compile in Lean 4.
      3) Compute pass@k for that docstring with pass_at_k(...).
      4) Return the *average* pass@k across all docstrings.

    Args:
        docstrings (List[str]): The set of Lean tasks or docstrings.
        server (Server): PyPantograph server instance for Lean 4 checks.
        k (int): The top-k threshold for pass@k.
        num_samples (int): # completions to generate per docstring.

    Returns:
        float: The average pass@k across all docstrings.
    """
    pass_vals = []

    for idx, statement in enumerate(docstrings):
        # 1. Generate completions
        completions = generate_snippets(prompt=statement, num_samples=num_samples)
        # 2. Count how many compile
        num_correct = sum(check_lean_compiles(c, server) for c in completions)
        # 3. pass@k for this docstring
        this_pass_k = pass_at_k(num_samples, num_correct, k)
        pass_vals.append(this_pass_k)

        print(f"\n[{idx+1}/{len(docstrings)}] Problem: {repr(statement)}")
        print(f"  #Generated={num_samples}, #Correct={num_correct}, pass@{k}={this_pass_k:.3f}")

    # 4. Return average
    return float(np.mean(pass_vals)) if pass_vals else 0.0


def main() -> None:
    """
    Demonstration:
      - N=10 docstrings,
      - k=5 => pass@5,
      - num_samples=10 completions per docstring.

    GPT-2 typically won't yield valid Lean code, but we illustrate the pipeline:
      docstrings => generate_snippets => check_lean_compiles => pass_at_k => average pass@k
    """
    seed_everything(42)  # For reproducibility

    # Create a default PyPantograph server to check Lean 4 code
    server = Server()

    # Example docstrings: toy "theorem" statements
    docstrings = [
        f"theorem lemma{i} : 2 + {i} = {i} + 2 := by"
        for i in range(1, 11)
    ]

    # Evaluate pass@5 with 10 completions per docstring
    k = 5
    num_samples = 10
    avg_pass_k = run_pass_k_eval(docstrings, server, k=k, num_samples=num_samples)

    print(f"\n==== Final Average Pass@{k} across N={len(docstrings)} tasks: {avg_pass_k:.3f} ====\n")


if __name__ == "__main__":
    main()
