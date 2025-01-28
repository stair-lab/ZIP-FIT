# lean_pass_k_unbiased.py
#!/usr/bin/env python3
"""
Implements the unbiased Pass@k estimator from the Codex paper [Chen et al., 2021],
plus a demonstration of using **vLLM** to generate Lean 4 code and measure syntactic
or "fully correct" compilation success with PyPantograph.

We replace the old GPT-2 pipeline approach with a vLLM approach:
    from vllm import LLM, SamplingParams
    # generate completions for each prompt

Then we apply our pass@k logic to these completions, checking each snippet's
Lean 4 compilation success via PyPantograph. We do *strict* checks or syntax-only
checks, depending on your preference (see the check_lean_* function).

References:
 - Official OpenAI HumanEval code, for pass@k formula:
   https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py
 - "Evaluating Large Language Models Trained on Code" (Chen et al., 2021)
 - vLLM docs: https://github.com/vllm-project/vllm
 - PyPantograph docs: https://github.com/stanford-centaur/PyPantograph

--------------------------------------------------------------------------------
Unbiased Pass@k:
~~~~~~~~~~~~~~~~
For a single problem, if we generate n completions and find c that pass,
the unbiased pass@k is:
   pass@k = 1 - ( (n-c choose k)/(n choose k) )
We handle edge cases c=0, c>=n, or (n-c)<k => pass@k=1 or 0.

--------------------------------------------------------------------------------
Usage Steps:
  1) pip install vllm
  2) pip install PyPantograph
  3) Ensure Lean 4 is installed
  4) python lean_pass_k_unbiased.py --model /path/to/model [--max-n 5] etc.

(We provide a basic example of how to gather completions from a vLLM model
and compute pass@k. You can adapt for your own code.)
"""

import argparse
import numpy as np
from typing import List

from pantograph.server import Server
from pantograph.data import CompilationUnit

# The user must install vllm from https://github.com/vllm-project/vllm
# e.g. pip install vllm
from vllm import LLM, SamplingParams

# ------------------------------------------------------------------------
# Seeding
# ------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """
    Seed Python, NumPy, and Torch. 
    (vLLM's random seed is set via the sampling_params or 
     environment variables, but we'll do a global seed too.)
    """
    import random
    import torch
    from transformers import set_seed as hf_set_seed

    print(f"Setting random seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)  # ensure HF transformers reproducibility, if used

# ------------------------------------------------------------------------
# Pass@k function
# ------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute unbiased pass@k from the Codex/HumanEval approach:
      pass@k = 1 - ( (n-c choose k)/(n choose k) )
              = 1 - Π_{i=0..k-1}[ (n-c-i)/(n-i) ]
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

# ------------------------------------------------------------------------
# Checking Lean code with PyPantograph
# ------------------------------------------------------------------------
def check_lean_compiles_strict(snippet: str, server: Server, require_no_goals: bool = True) -> bool:
    """
    Strict check: 
      - Use server.load_sorry(...) to parse the snippet as a file of Lean code.
      - If any compilation unit has messages containing 'error', fail.
      - If require_no_goals=True and there's leftover goals => fail.
    => Return True if no error or leftover goals remain.
    """
    try:
        units = server.load_sorry(snippet)
    except Exception:
        return False  # Parse or server error => fail

    for cu in units:
        # If any message has 'error', fail
        for msg in cu.messages:
            if 'error' in msg.lower():
                return False
        # If leftover goals => fail if require_no_goals
        if require_no_goals and cu.goal_state is not None:
            return False
    return True

# ------------------------------------------------------------------------
# vLLM code generation
# ------------------------------------------------------------------------
def generate_lean_snippets_with_vllm(
    prompts: List[str],
    llm: LLM,
    n: int,
    sampling_params: SamplingParams,
) -> List[List[str]]:
    """
    For each prompt in 'prompts', we request 'n' completions from vLLM,
    then gather them as strings in a list-of-lists.

    Returns:
      completions[i] => the list of 'n' code snippets for prompts[i].
    """
    all_completions: List[List[str]] = []
    # vLLM's LLM.generate returns a list of BatchOutput, each with .outputs
    # We'll do it in single-batch mode if the user only has a few prompts,
    # or you can chunk them for large-scale usage.

    # We'll just do them one prompt at a time for clarity:
    for prompt in prompts:
        output = llm.generate([prompt], sampling_params)  # single-prompt
        # output is a list w/ length=1 for single-prompt
        # output[0].outputs is a list of n completions
        completions_for_prompt = []
        if len(output) > 0:
            for i, outcomp in enumerate(output[0].outputs):
                completions_for_prompt.append(outcomp.text)
        else:
            # no completions => empty
            pass
        all_completions.append(completions_for_prompt)

    return all_completions

# ------------------------------------------------------------------------
# Evaluate pass@k
# ------------------------------------------------------------------------
def run_pass_k_eval_vllm(
    prompts: List[str],
    server: Server,
    llm: LLM,
    sampling_params: SamplingParams,
    k: int = 5,
    n: int = 10
) -> float:
    """
    1) For each prompt in 'prompts', get n completions from vLLM
    2) Check compilation success (# correct)
    3) Compute pass@k for each prompt, then average
    """
    all_completions = generate_lean_snippets_with_vllm(prompts, llm, n, sampling_params)

    pass_values: List[float] = []
    for idx, (prompt, completions_for_prompt) in enumerate(zip(prompts, all_completions)):
        # Count how many compile
        num_correct = sum(check_lean_compiles_strict(c, server) for c in completions_for_prompt)
        p_k = pass_at_k(n, num_correct, k)
        pass_values.append(p_k)

        print(f"\nPrompt #{idx+1}: {repr(prompt)}")
        print(f"  #Generated={n}, #Correct={num_correct}, pass@{k}={p_k:.3f}")

    return float(np.mean(pass_values)) if pass_values else 0.0

# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main() -> None:
    """
    Example script:
      1) parse CLI args
      2) set seed
      3) create a Pantograph server
      4) create a vLLM model from the user-provided path
      5) run pass@k on a few toy Lean docstrings
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or name of your vLLM model. E.g. /path/to/gemma or hf:google/gemma-2-2b",
    )
    parser.add_argument("--k", type=int, default=5, help="top-k for pass@k")
    parser.add_argument("--n", type=int, default=10, help="completions to sample per prompt")
    parser.add_argument("--seed", type=int, default=42, help="global random seed")
    args = parser.parse_args()

    # 1) Seeding
    seed_everything(args.seed)

    # 2) Start Pantograph server
    server = Server()

    # 3) Create vLLM model
    print(f"Loading vLLM model from: {args.model}")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        download_dir="vllm_model_cache"  # Or your desired cache dir
    )

    # 4) sampling params
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=256,
        n=args.n,
    )

    # 5) docstrings to test
    docstrings = [
        "theorem lemma1 : 2 + 1 = 1 + 2 := by",
        "theorem lemma2 : 2 + 2 = 2 + 2 := by",
    ]

    # Evaluate pass@k
    final_score = run_pass_k_eval_vllm(
        prompts=docstrings,
        server=server,
        llm=llm,
        sampling_params=sampling_params,
        k=args.k,
        n=args.n
    )

    print(f"\n=== Final Average Pass@{args.k} across {len(docstrings)} prompts: {final_score:.3f} ===")

if __name__ == "__main__":
    main()
