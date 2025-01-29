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

import os
import numpy as np
from typing import List
from transformers import pipeline, set_seed
from pantograph.server import Server
from pantograph.data import CompilationUnit, TacticInvocation
import tqdm

from vllm import LLM, SamplingParams
from vllm import RequestOutput

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
    try:
        from vllm import set_seed as vllm_set_seed
        vllm_set_seed(seed)
    except ImportError:
        print("vLLM not installed or vllm set seed has a bug, skipping vLLM seed setting.")


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Computes the unbiased pass@k metric from the Codex/HumanEval paper [Chen et al., 2021],
    using a numerically stable product formulation:

        pass@k = 1 - ( (n-c choose k) / (n choose k) )
                = 1 - ∏(i=0 to k-1) [ (n - c - i) / (n - i) ]

    Explanation & Implementation Details:
    -------------------------------------
    1) **Combinatorial Definition**:
       The original definition in the Codex paper is:
         pass@k = 1 - [ (n-c choose k) / (n choose k) ].
       This means: given `n` generated completions, and `c` of them are correct,
       the probability that "all top k" picks are incorrect is `(n-c choose k) / (n choose k)`.
       So pass@k is `1 -` that probability.

    2) **Why a Product Form?**:
       Directly computing binomial coefficients `(n-c choose k)` and `(n choose k)` can lead to
       large factorial terms or floating-point overflow for bigger n, c, k. The product form:

          ( (n-c choose k) / (n choose k) )
          = Π (i=0..k-1)  ( (n-c-i) / (n-i) )

       avoids computing large factorials. We multiply smaller ratios that stay
       closer to 1.0 numerically. 

    3) **Edge Cases**:
       - If `c=0`, no correct completions => pass@k=0.
       - If `c >= n`, all completions are correct => pass@k=1.
       - If `(n - c) < k`, we can't fill k slots with incorrect => pass@k=1.

    4) **Numeric Stability**:
       Because we multiply a sequence of terms `(n-c-i)/(n-i)` each roughly in [0,1], 
       we avoid factorial explosion or large intermediate sums. Multiplying these 
       fractions is more stable for up to typical values of n. 
       Finally, pass@k is `1 - product_of_those_fractions`.

    Complexity:
    -----------
    - Time complexity: O(k) for the loop, which is usually small (k <= 100).
    - No large factorial or gamma function calls, so it’s efficient & stable for typical n.

    Args:
        n (int): Total number of code completions generated for a problem.
        c (int): Number of correct completions among those n.
        k (int): "Top-k" threshold for pass@k.

    Returns:
        float: pass@k in [0.0, 1.0], capturing the fraction of problems for
               which at least one of the top-k completions is correct.
    """
    # Edge cases for immediate short-circuit:
    if c == 0:        # no correct completions
        return 0.0
    if c >= n:        # all completions are correct
        return 1.0
    if (n - c) < k:   # not enough incorrect to fill k picks
        return 1.0

    # Numerically stable product form:
    product_term = 1.0
    for i in range(k):
        numerator = (n - c - i)
        denominator = (n - i)
        if numerator < 0:
            # If for some reason numerator < 0, we can't fill the top k with all incorrect
            return 1.0
        product_term *= (numerator / denominator)

    return 1.0 - product_term


def generate_snippets_hf_pipeline(prompt: str, num_samples: int = 5, max_length: int = 512, model: str = 'gpt2') -> List[str]:
    """
    Generates Lean 4 code snippets from a textual prompt using GPT-2 (toy example).

    This is just for demonstration: GPT-2 is not trained on Lean, so it's
    likely to produce nonsense in real usage. But it demonstrates the pipeline.
    """
    generator = pipeline("text-generation", model=model)
    outputs = generator(
        prompt,
        num_return_sequences=num_samples,  # e.g. 5 completions
        max_length=max_length,            # e.g. up to 64 tokens total
        do_sample=True,                   # random sampling
        temperature=1.0,                  # sampling "temperature", 1.0 => fairly random
    )
    return [o["generated_text"] for o in outputs]


def check_lean_compiles_strict(snippet: str, server: Server, require_no_goals: bool = True) -> bool:
    """
    Strictly checks whether a Lean 4 snippet “fully compiles” according to PyPantograph,
    by analyzing the returned CompilationUnits from server.load_sorry(...).

    Steps & Rationale
    -----------------
    1) **Load the snippet**:
       We call `server.load_sorry(snippet)`, which parses the snippet as if it were
       a Lean 4 source file. Note that Lean can accept multiple definitions/
       theorems in a single snippet, each mapped to a “compilation unit.”

       - If a parse-level or “fatal” server error occurs, `load_sorry` can raise
         an exception. We catch that and return False.

    2) **Check each CompilationUnit**:
       - Each “compilation unit” corresponds to an area in the snippet that Lean
         recognized (like “theorem lemma1 : p -> p := by ...”).
       - If `cu.messages` contains strings with “error” in them, we assume a
         compilation error was detected. We return False.

    3) **Leftover goals**:
       - If `require_no_goals` is True, we also fail if `cu.goal_state` is not None.
         This typically means Lean recognized a “sorry” or leftover proof hole,
         or some type error that was turned into a goal. 
         e.g. “theorem lemma_fail : 2 + 2 = 5 := by rfl” might produce leftover
         type mismatch goals if Lean tries to unify 4 with 5.
       - If `require_no_goals` is False, we only fail on “hard” errors, ignoring
         partial or logically incorrect proofs as long as they parse.

    4) **Return**:
       - True if we never encountered a parse error, no messages with “error,”
         and (optionally) no leftover goals (if `require_no_goals=True`).
       - False otherwise.

    Why This Matters
    ----------------
    - Lean 4’s design allows partial type errors or leftover subgoals to
      accumulate without halting compilation. So code might “parse” but still
      be incomplete or contradictory. For a “strict” notion of success, we
      treat leftover goals as a fail, but for “syntactic-only” we can ignore
      them. 
    - This approach ensures you can systematically measure how many LLM
      completions produce truly “fully compiled” Lean 4 code.

    Args:
        snippet (str):
          A string containing what we’d treat as top-level Lean 4 code.
        server (Server):
          A PyPantograph server instance, e.g. `Server()`.
        require_no_goals (bool):
          If True, leftover subgoals => fail. If False, leftover subgoals do not
          matter for success/failure.

    Returns:
        bool:
          True if no parse error, no “error” messages, and (if `require_no_goals=True`)
          no leftover goals. Otherwise False.

    Example Usage
    -------------
    >>> from pantograph.server import Server
    >>> server = Server()
    >>> snippet = '''theorem lemma_success : 2 = 2 := by rfl'''
    >>> check_lean_compiles_strict(snippet, server)
    True

    >>> snippet2 = '''theorem lemma_fail : 2 + 2 = 5 := by rfl'''
    >>> check_lean_compiles_strict(snippet2, server)
    False  # leftover goals or error messages
    """
    try:
        # Attempt to parse and gather compilation units from snippet
        units = server.load_sorry(snippet)
    except Exception:
        # If the server threw an exception, it's likely a parse or fatal error
        return False

    # Inspect each compilation unit for error messages or leftover goals
    for cu in units:
        # If any message includes 'error', we fail immediately
        for msg in cu.messages:
            if 'error' in msg.lower():
                return False

        # If we require no leftover goals, check if goal_state is present
        if require_no_goals and cu.goal_state is not None:
            return False

    # If we get here, no blocking errors or leftover goals => success
    return True


def _run_pass_k_eval(
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
        completions = generate_snippets_hf_pipeline(prompt=statement, num_samples=num_samples)
        # completions = generate_snippets_vllm(prompt=statement, num_samples=num_samples)
        # Count how many are correct
        num_correct = sum(check_lean_compiles_strict(c, server) for c in completions)
        # Pass@k
        this_pass_k = pass_at_k(num_samples, num_correct, k)
        pass_vals.append(this_pass_k)

        print(f"\n[{idx+1}/{len(docstrings)}] Problem: {repr(statement)}")
        print(f"  #Generated={num_samples}, #Correct={num_correct}, pass@{k}={this_pass_k:.3f}")

    return float(np.mean(pass_vals)) if pass_vals else 0.0

def run_pass_k_eval(
    prompts: List[str],
    server: Server,
    model: str, 
    k: int = 5,
    num_samples: int = 30,
    dtype="bfloat16",
    eval_batch_size: int = 32,
    max_tokens: int = 512,
    top_p: float = 0.95,       # Nucleus sampling: limits token choices to the smallest set whose cumulative probability is >= top_p (e.g., 95%).
    top_k: float = 50,         # Top-k sampling: limits token choices to the top-k most likely tokens at each step (e.g., top 50 tokens).
    temperature: float = 0.7,
) -> float:
    """
    Evaluate pass@k for each docstring: generate code with GPT-2,
    check how many compile under strict rules, then compute pass@k.
    """
    llm = LLM(
        model=model,                         # create vLLM model from path or HF name
        dtype=dtype,                         # specify float precision
        trust_remote_code=True,              # allow custom model code
    )
    sampling_params = SamplingParams(
        temperature=temperature,# Controls randomness: higher values (e.g., 1.0) increase randomness; lower values (e.g., 0.1) make outputs more deterministic.
        top_p=top_p,            # Nucleus sampling: limits token choices to the smallest set whose cumulative probability is >= top_p (e.g., 95%).
        top_k=top_k,            # Top-k sampling: limits token choices to the top-k most likely tokens at each step (e.g., top 50 tokens).
        max_tokens=max_tokens,  # Maximum number of tokens (including prompt + generated tokens) for each completion.
        n=num_samples,          # Number of completions to generate per prompt.
    )

    # For each batch of prompts, call llm.generate for paralellization
    batched_prompts = [prompts[i:i + eval_batch_size] for i in range(0, len(prompts), eval_batch_size)]
    outputs: List[RequestOutput] = []

    # For each batch of prompts, call llm.generate
    for batch_prompts in tqdm.tqdm(batched_prompts, desc="Generating completions"):
        batch_outputs = llm.generate(batch_prompts, sampling_params)  # one generate call
        outputs.extend(batch_outputs)  # accumulate results for each batch

    # Each element in outputs is a RequestOutput with up to n completions in .outputs
    # We want a final structure: text_outputs[i] = list of all completions for prompts[i]
    # So we gather them below:
    text_outputs: List[List[str]] = []
    for request_out in outputs:  
        # request_out.outputs is a list of n completions
        # We extract the .text from each one
        completions = [completion.text for completion in request_out.outputs]
        text_outputs.append(completions)
    assert len(text_outputs) == len(prompts), ( f"Mismatch in number of outputs ({len(text_outputs)}) vs. prompts ({len(prompts)})")

    # Now compute pass@k for each prompt individually
    pass_vals: List[float] = []
    for completions_for_prompt in text_outputs:
        # Check which completions compile => c = sum of successes
        successes = [check_lean_compiles_strict(c, server) for c in completions_for_prompt]
        c = sum(successes)
        pass_val = pass_at_k(num_samples, c, k)
        pass_vals.append(pass_val)
    # Finally, average pass@k across all prompts
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # choose GPU
    server = Server()

    # 1) Manual snippet test
    # test_manual_snippets(server)
    print()

    # 2) Log In
    from huggingface_hub import create_repo, upload_file, login, whoami
    key_file_path = "~/keys/master_hf_token.txt"
    key_file_path = os.path.abspath(os.path.expanduser(key_file_path))
    with open(key_file_path, "r", encoding="utf-8") as f:
        token = f.read().strip()
    login(token=token)
    os.environ['HUGGINGFACE_TOKEN'] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    user_info = whoami()
    print(f"Currently logged in as: {user_info['name']}\n")

    # 3) Model pass@k test (toy)
    from huggingface_hub import create_repo, upload_file, whoami
    whoami()
    # model = 'gpt2'
    # model = 'google/gemma-2-2b'
    # model = 'UDACA/gemma-2-2b'
    model = 'internlm/internlm2-math-plus-1_8b'
    docstrings = [
        "theorem lemma1 : 2 + 1 = 1 + 2 := by",
        "theorem lemma2 : 2 + 2 = 2 + 2 := by",
    ]
    # add some random junk
    docstrings += [d.replace('e', ' ') + " randomjunk" for d in docstrings]

    k = 2
    num_samples = 5
    score = run_pass_k_eval(docstrings, server, model, k=k, num_samples=num_samples, eval_batch_size=2)
    print(f"\n==== Final Average Pass@{k} across {len(docstrings)} tasks: {score:.3f} ====\n")


if __name__ == "__main__":
    main()
