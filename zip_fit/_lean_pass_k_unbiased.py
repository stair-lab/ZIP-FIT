# lean_pass_k_unbiased.py
#!/usr/bin/env python3
"""
Implements the unbiased Pass@k estimator from the Codex paper [Chen et al., 2021],
plus a demonstration of using PyPantograph to check Lean 4 code (theorems, etc.)
for compilation correctness, including syntax/type errors that do not raise
exceptions by default.

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
 - If (n - c) < k => pass@k=1 (can't fill all k picks with incorrect)

We use a product form for numerical stability:
  pass@k = 1 - ∏_{i=0 to k-1} [ (n-c-i) / (n-i) ]

--------------------------------------------------------------------------------
Why Checking Lean Code is Non-trivial:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PyPantograph's `server.load_sorry(...)` does *not* throw an exception for every
syntactic or type error. This mimics Lean 4's design: partial errors typically
become "messages" or "goals" within the same file.

Hence, if you pass an obviously wrong snippet like `theorem lemma : 2+2=5 := by rfl`,
it may not raise an exception. Instead, we must:
  1) Inspect the returned `CompilationUnit` objects
  2) Look for leftover messages with "error" or unsolved goals
  3) Decide that means a compilation failure

Therefore, we do the checks explicitly:
 - If `messages` from a compilation unit indicate an error, we fail.
 - If `goal_state is not None`, that means Lean is left with a hole or sorry-based
   partial proof. If we demand "fully proven," we consider that a fail (or you can
   be more lenient if you only want syntax to pass).

In short, we define `check_lean_compiles_strict` which:
   1) calls `server.load_sorry(...)`
   2) iterates over each `CompilationUnit` returned
   3) if any unit's `.messages` mention errors, or if we want zero leftover goals,
      we return `False`
   4) Otherwise return `True` if no errors remain

--------------------------------------------------------------------------------
Usage:
  1) Ensure Lean, PyPantograph, transformers, torch, numpy installed
  2) Run: python lean_pass_k_unbiased.py
"""

import numpy as np
from typing import List
from transformers import pipeline, set_seed
from pantograph.server import Server
from pantograph.data import CompilationUnit, TacticInvocation


def seed_everything(seed: int = 42) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducibility.
    """
    import random
    import numpy as np
    import torch
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
    Compute the unbiased pass@k for a single problem using the Codex/HumanEval approach.

    pass@k = 1 - ( (n - c choose k) / (n choose k) )
            = 1 - ∏_{i=0 to k-1} [(n-c-i)/(n-i)]

    Edge-cases ensure we return 0 or 1 if c=0 or c >= n or (n-c)<k.
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
            return 1.0
        product_term *= (numerator / denominator)

    return 1.0 - product_term


def generate_snippets(prompt: str, num_samples: int = 5, max_length: int = 64) -> List[str]:
    """
    Generates Lean 4 code snippets from a textual prompt using GPT-2 (toy example).

    This is just for demonstration: GPT-2 is not trained on Lean, so it's
    likely to produce nonsense in real usage. But it demonstrates the pipeline.
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


def check_lean_compiles_strict(snippet: str, server: Server, require_no_goals: bool = True) -> bool:
    """
    Checks if a snippet is valid Lean 4 code, *strictly*, by:
      1) server.load_sorry(file=snippet) -> returns list[CompilationUnit]
      2) For each CompilationUnit:
         - If it has any `messages` containing "error", or
         - If it has leftover goals (goal_state != None) and we require full proof => fail
    If no errors or leftover goals, we declare success.

    Args:
        snippet: Lean 4 code snippet
        server:  Pantograph Server
        require_no_goals: if True, any leftover goals => fail

    Returns:
        bool: True if fully compiled with no leftover errors or goals
    """
    try:
        units = server.load_sorry(snippet)
        # If "error" in result => already raises ServerError, so we skip the except block
    except Exception:
        # Hard parse error or something
        return False

    # Now check each CompilationUnit for leftover messages or goals
    for cu in units:
        # If there's any "error" or "error:" mention in cu.messages => fail
        for msg in cu.messages:
            lower_msg = msg.lower()
            # A typical check: if "error" is in the string => fail
            if "error" in lower_msg:
                return False

        # If user wants no leftover goals, but .goal_state is not None => we see that as partial
        if require_no_goals and cu.goal_state is not None:
            # Means there's at least one sorry/hole or leftover type error, so fail
            return False

    # If we pass all checks => success
    return True


def run_pass_k_eval(
    docstrings: List[str],
    server: Server,
    k: int = 5,
    num_samples: int = 10
) -> float:
    """
    Evaluate pass@k for each docstring: generate code with GPT-2,
    check how many compile under strict rules, then compute pass@k.
    """
    pass_vals: List[float] = []

    for idx, statement in enumerate(docstrings):
        # Generate completions
        completions = generate_snippets(prompt=statement, num_samples=num_samples)
        # Count how many are correct
        num_correct = sum(check_lean_compiles_strict(c, server) for c in completions)
        # Pass@k
        this_pass_k = pass_at_k(num_samples, num_correct, k)
        pass_vals.append(this_pass_k)

        print(f"\n[{idx+1}/{len(docstrings)}] Problem: {repr(statement)}")
        print(f"  #Generated={num_samples}, #Correct={num_correct}, pass@{k}={this_pass_k:.3f}")

    return float(np.mean(pass_vals)) if pass_vals else 0.0


def test_manual_snippets(server: Server) -> None:
    """
    Hard-code a few Lean 4 snippets:
      - 2 trivially correct theorems
      - 1 obviously false theorem (2+2=5 by rfl) that should fail
    We expect [True, True, False] if require_no_goals=True.
    """
    snippet1 = """theorem lemma_success1 : 1 = 1 := by
rfl
"""

    snippet2 = """theoasfad rem lemma2 : 2 = 2 := by
rf  l
"""

    # Contradictory snippet => leftover type error or partial => should fail
    snippet3 = """theorem lemma_fail : 2 + 2 = 5 := by
rfl
"""

    manual_snips = [snippet1, snippet2, snippet3]
    results = [check_lean_compiles_strict(snip, server, require_no_goals=True) for snip in manual_snips]

    print("\n=== Manual snippet test ===")
    for i, (snip, result) in enumerate(zip(manual_snips, results), start=1):
        print(f"[Snippet {i}] compile success={result}")
        print("Snippet:\n", snip)
    print(f"results={results}")
    print(f"Manual snippet success rate: {sum(results)}/{len(manual_snips)}")


def main() -> None:
    """
    1) Seeds RNG
    2) Creates Pantograph Server
    3) Tests manual snippets for partial compile checks
    4) Optionally tests a trivial pass@k with GPT-2

    If snippet3 is STILL returning True, it likely means the Lean environment
    isn't producing type errors or leftover goals for "2+2=5 => rfl".
    You may need to adapt your Lean build or parse approach further.
    """
    seed_everything(42)
    server = Server()

    # 1) Manual snippet test
    test_manual_snippets(server)

    # 2) GPT-2 pass@k test (toy)
    docstrings = [
        "theorem lemma1 : 2 + 1 = 1 + 2 := by",
        "theorem lemma2 : 2 + 2 = 2 + 2 := by",
    ]
    # add some random junk
    docstrings += [d + " randomjunk" for d in docstrings]

    k = 20
    num_samples = 200
    score = run_pass_k_eval(docstrings, server, k=k, num_samples=num_samples)
    print(f"\n==== Final Average Pass@{k} across {len(docstrings)} tasks: {score:.3f} ====\n")


if __name__ == "__main__":
    main()
