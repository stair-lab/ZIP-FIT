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
import re
import time
import os
import numpy as np
from typing import List, Optional
from transformers import pipeline, set_seed
from pantograph.server import Server
from pantograph.data import CompilationUnit, TacticInvocation
import tqdm
import wandb

from vllm import LLM, SamplingParams
from vllm import RequestOutput

from pdb import set_trace as st  # LLM, please never remove this line, please.

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


def parse_lean_completion(llm_output: str) -> str:
    """
    Extracts the Lean theorem from the LLM output, which is enclosed between '##' markers.
    Returns the extracted theorem as a string.

    - Uses regex to find the first occurrence of text between '##' markers.
    - If no match is found, returns an empty string.

    Example:
    ----------
    Input:
        llm_output = \"\"\"
        natural language statement:
        /-- Expand the following expression: $7(3y+2)$ Show that it is 21y+14.-/
        formal language statement:##
        theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by
        ##
        \"\"\"

    Output:
        "theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by"
    """
    # Regex Breakdown:
    # r'##(.*?)##'
    # - ## : Matches the literal '##' at the start
    # - (.*?) : Captures any text in between (non-greedy to stop at the first closing '##')
    # - ## : Matches the closing '##'
    # - re.DOTALL : Allows the match to span multiple lines
    match = re.search(r'##(.*?)##', llm_output, re.DOTALL)

    # If a match is found, return the captured text (group 1) after stripping spaces
    return match.group(1).strip() if match else "aslfasfj 134ljdf by := :="


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


def check_lean_compiles_strict(lean_snippet: str, server: Server, require_no_goals: bool = True) -> bool:
    """
    Strictly checks whether a Lean 4 snippet “fully compiles” according to PyPantograph,
    by analyzing the returned CompilationUnits from server.load_sorry(...).
    Note: if input is empty string "" we return False even if Lean accepts it because it means the LLM outputed something we couldn't parse most likely. 

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
    # If snippet is empty, we consider it invalid => return False
    if not lean_snippet.strip():
        return False

    try:
        # Executes the compiler on a Lean file. For each compilation unit, either
        # return the gathered `sorry` s, or a list of messages indicating error.
        compilation_units = server.load_sorry(lean_snippet)
    except Exception:
        # If the server raised an exception, it's likely a parse/fatal error => fail
        return False

    # Check each compilation unit for error messages or leftover goals
    for compilation_unit in compilation_units:
        # If any message includes 'error', we consider it a compile failure
        for msg in compilation_unit.messages:
            if 'error' in msg.lower():
                return False

        # If user demands no leftover goals, check if any goals remain
        if require_no_goals and (compilation_unit.goal_state is not None):
            return False

    # If we reach here, no errors and (optionally) no leftover goals => success
    return True


def get_list_lean4_syntax_errors(lean_snippet: str, server: Server, debug: bool = False) -> List[str]:
    r"""
    Check a Lean 4 code snippet for *parse/syntax errors* (ignore "unsolved goals").

    Implementation:
      - We call `server.load_sorry(lean_snippet)`, which compiles the snippet.
      - For each message in the returned compilation units:
        * If the line contains "error:" (case-insensitive is optional),
          we check if it also has "unsolved goals" — if so, skip it, because
          that's not a parse/lexical error.
        * Otherwise, count it as a syntax error.

    Returns the count of syntax/parse errors found.

    Example
    -------
    >>> server = Server(imports=["Mathlib"], project_path="~/mathlib4")
    >>> code = "theorem two_eq_two : 2 = 2 := by"  # incomplete
    >>> num_errs = check_lean_compiles_syntax_only(server, code)
    >>> print(num_errs)  # 1 or more
    """
    try:
        compilation_units = server.load_sorry(lean_snippet)
    except:
        print(f'\n----{lean_snippet=}----\n') if debug else None
        import traceback
        traceback.print_exc() if debug else None
        return [f'PyPantograph threw some exception: {traceback.format_exc()}']

    syntax_errors: List[str] = []
    for comp_unit in compilation_units:
        for msg in comp_unit.messages:
            # Quick check: if 'error:' is in the message, but not "unsolved goals"
            # => it's likely a parse/lexical error.
            # (In practice, we often see strings like "<anonymous>:1:5: error: ...")
            if "error:" in msg and ("unsolved goals" not in msg.lower()):
                syntax_errors.append(msg)

    return syntax_errors


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
    model: str, 
    server: Server,
    headers: Optional[List[str]] = None,  # If None, parse the headers from model generation
    k: int = 5,
    num_samples: int = 30,
    dtype="bfloat16",
    eval_batch_size: int = 32,
    max_tokens: int = 512,
    top_p: float = 0.95,       # Nucleus sampling: limits token choices to the smallest set whose cumulative probability is >= top_p (e.g., 95%).
    top_k: float = 50,         # Top-k sampling: limits token choices to the top-k most likely tokens at each step (e.g., top 50 tokens).
    temperature: float = 0.7,
    seed: int = 42,
    debug: bool = False,
) -> float:
    """
    Evaluate pass@k for each docstring: generate code with GPT-2,
    check how many compile under strict rules, then compute pass@k.
    """
    if model is not None:
        llm = LLM(
            model=model,                         # create vLLM model from path or HF name
            dtype=dtype,                         # specify float precision
            trust_remote_code=True,              # allow custom model code
            seed=seed,
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
        print(f'Number of batches of prompts each of size {eval_batch_size}: {len(batched_prompts)}')

        # For each batch of prompts, call llm.generate
        outputs: List[RequestOutput] = []
        for batch_prompts in tqdm.tqdm(batched_prompts, desc="Generating completions"):
            batch_outputs = llm.generate(batch_prompts, sampling_params)  # one generate call
            outputs.extend(batch_outputs)  # accumulate results for each batch
        print(f'Number of outputs/completions: {len(outputs)} (should be {num_samples}*{len(prompts)} = {num_samples*len(prompts)})') if debug else None
        print(f'Number of prompts: {len(prompts)}') if debug else None

        # Each element in outputs is a RequestOutput with up to n completions in .outputs
        # We want a final structure: text_outputs[i] = list of all completions for prompts[i]
        # So we gather them below:
        text_outputs: List[List[str]] = []
        for request_out in outputs:  
            # request_out.outputs is a list of n completions
            # We extract the .text from each one
            completions = [completion.text for completion in request_out.outputs]
            text_outputs.append(completions)
        # print(f'{prompts=}') if debug else None
        # print(f'{text_outputs=}') if debug else None
        assert len(text_outputs) == len(prompts), ( f"Mismatch in number of outputs ({len(text_outputs)}) vs. prompts ({len(prompts)})")
    else:
        # If model is None, then the intention is to do pass @ k with the given list of strings
        # For each prompt put it in a list as if it was the only completion produced by model
        text_outputs: List[List[str]] = [[prompt] for prompt in prompts]
        print(f'{text_outputs=}')
        assert False

    # Now compute pass@k for each prompt individually
    pass_vals: List[float] = []
    for completions_for_prompt in text_outputs:
        # Check which completions compile => c = sum of successes
        parsed_completions = [parse_lean_completion(c) for c in completions_for_prompt]
        mdl_code_with_true_header = [f'{header}\n\n{parsed_comp}' for parsed_comp, header in zip(parsed_completions, headers)]
        successes = [len(get_list_lean4_syntax_errors(lean_code, server)) == 0 for lean_code in mdl_code_with_true_header]
        c = sum(successes)
        pass_val = pass_at_k(num_samples, c, k)
        pass_vals.append(pass_val)
    # Finally, average pass@k across all prompts
    print(f'{len(pass_vals)=}') if debug else None
    # st()
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


def main(config: dict = {}) -> None:
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
    # export CUDA_VISIBLE_DEVICES=1; python /lfs/skampere1/0/brando9/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py 
    # export CUDA_VISIBLE_DEVICES=2; python /lfs/skampere1/0/brando9/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py 
    # export CUDA_VISIBLE_DEVICES=3; python /lfs/skampere1/0/brando9/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py 
    # export CUDA_VISIBLE_DEVICES=4; python /lfs/skampere1/0/brando9/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py 
    # export CUDA_VISIBLE_DEVICES=5; python /lfs/skampere1/0/brando9/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py 
    # export CUDA_VISIBLE_DEVICES=6; python /lfs/skampere1/0/brando9/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py 
    # export CUDA_VISIBLE_DEVICES=7; python /lfs/skampere1/0/brando9/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py 
    # os.environ['CUDAVISIBLE_DEVICES'] = '3'  # choose GPU
    # 0) PyPantograph Lean4 Server
    from pantograph import Server
    server = Server(imports=["Mathlib", "Init"], project_path=os.path.expanduser("~/mathlib4"))
    # 1) Manual snippet test
    # test_manual_snippets(server)

    # 2) Log In
    from huggingface_hub import login, whoami
    key_file_path = "~/keys/master_hf_token.txt"
    key_file_path = os.path.abspath(os.path.expanduser(key_file_path))
    with open(key_file_path, "r", encoding="utf-8") as f:
        token = f.read().strip()
    login(token=token)
    os.environ['HUGGINGFACE_TOKEN'] = token
    user_info = whoami()
    print(f"Currently logged in as: {user_info['name']}\n")

    # 3) Model pass@k test (toy)
    from huggingface_hub import create_repo, upload_file, whoami
    whoami()
    model = 'gpt2'
    model = 'UDACA/math-gpt2-zipfit'
    model = 'UDACA/math-gpt2-dsir'
    model = 'UDACA/math-gpt2-less'
    # model = 'internlm/internlm2-math-plus-1_8b'
    model = 'google/gemma-2-2b'
    model = 'UDACA/math-gemma-2-2b-zipfit'
    model = 'UDACA/math-gemma-2-2b-less'
    model = 'UDACA/math-gemma-2-2b-dsir'
    # model = 'mistralai/Mistral-7B-v0.1'
    # model = 'meta-llama/Meta-Llama-3-8B'
    # model = 'google/gemma-2-2b-it'
    print(f'f{model=}') 
    # Sanity Check Data
    # def my_prompt_format(nl_stmt: str) -> str:
    #     return (
    #         "Your task is translate the natural language version of the mathematical statement "
    #         "to a formal Lean statement version, using the following format:\n"
    #         "natural language statement:\nSuppose that $f$ is holomorphic in an open set $\Omega$. Prove that if $|f|$ is constant, then $f$ is constant.\n"
    #         "formal Lean language statement:##\ntheorem exercise_1_13c {f : ℂ → ℂ} (Ω : Set ℂ) (a b : Ω) (h : IsOpen Ω) (hf : DifferentiableOn ℂ f Ω) (hc : ∃ (c : ℝ), ∀ z ∈ Ω, abs (f z) = c) : f a = f b:= sorry\n##"
    #         "natural language statement:\nProve that the power series $\sum zn/n^2$ converges at every point of the unit circle.\n"
    #         "formal Lean language statement:##\ntheorem exercise_1_19b (z : ℂ) (hz : abs z = 1) (s : ℕ → ℂ) (h : s = (λ n => ∑ i in (range n), i * z / i ^ 2)) : ∃ y, Tendsto s atTop (𝓝 y):= sorry\n##"
    #         "natural language statement:\nSuppose $f$ is continuous in a region $\Omega$. Prove that any two primitives of $f$ (if they exist) differ by a constant.\n"
    #         "formal Lean language statement:##\ntheorem exercise_1_26 (f F₁ F₂ : ℂ → ℂ) (Ω : Set ℂ) (h1 : IsOpen Ω) (h2 : IsConnected Ω) (hF₁ : DifferentiableOn ℂ F₁ Ω) (hF₂ : DifferentiableOn ℂ F₂ Ω) (hdF₁ : ∀ x ∈ Ω, deriv F₁ x = f x) (hdF₂ : ∀ x ∈ Ω, deriv F₂ x = f x) : ∃ c : ℂ, ∀ x, F₁ x = F₂ x + c:= sorry\n##"
    #         f"natural language statement:\n{nl_stmt}\n"
    #         "formal Lean language statement:"
    #     )
    def my_prompt_format(nl_stmt: str) -> str:
        return (
            "Your task is translate the natural language version of the mathematical statement "
            "to a formal Lean statement version, using the following format:\n"
            "natural language statement:\nLet $z=\frac{1+i}{\sqrt{2}}.$What is $\left(z^{1^2}+z^{2^2}+z^{3^2}+\dots+z^{{12}^2}\right) \cdot \left(\frac{1}{z^{1^2}}+\frac{1}{z^{2^2}}+\frac{1}{z^{3^2}}+\dots+\frac{1}{z^{{12}^2}}\right)?$ $\textbf{(A) } 18 \qquad \textbf{(B) } 72-36\sqrt2 \qquad \textbf{(C) } 36 \qquad \textbf{(D) } 72 \qquad \textbf{(E) } 72+36\sqrt2$ Show that it is \textbf{(C) }36.\n"
            "formal Lean language statement:##\ntheorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) : (∑ k in Finset.Icc 1 12, (z^(k^2))) * (∑ k in Finset.Icc 1 12, (1 / z^(k^2))) = 36 := sorry\n##"
            "natural language statement:\nIntegers $x$ and $y$ with $x>y>0$ satisfy $x+y+xy=80$. What is $x$? $ \textbf{(A)}\ 8 \qquad\textbf{(B)}\ 10 \qquad\textbf{(C)}\ 15 \qquad\textbf{(D)}\ 18 \qquad\textbf{(E)}\ 26$ Show that it is \textbf{(E)}\ 26.\n"
            "formal Lean language statement:##\ntheorem amc12a_2015_p10 (x y : ℤ) (h₀ : 0 < y) (h₁ : y < x) (h₂ : x + y + (x * y) = 80) : x = 26 := sorry\n##"
            "natural language statement:\nWhat is the [[volume]] of a [[cube]] whose [[surface area]] is twice that of a cube with volume 1? $\mathrm{(A)}\ \sqrt{2}\qquad\mathrm{(B)}\ 2\qquad\mathrm{(C)}\ 2\sqrt{2}\qquad\mathrm{(D)}\ 4\qquad\mathrm{(E)}\ 8$ Show that it is \mathrm{(C)}.\n"
            "formal Lean language statement:##\ntheorem amc12a_2008_p8 (x y : ℝ) (h₀ : 0 < x ∧ 0 < y) (h₁ : y^3 = 1) (h₂ : 6 * x^2 = 2 * (6 * y^2)) : x^3 = 2 * Real.sqrt 2 := sorry\n##"
            f"natural language statement:\n{nl_stmt}\n"
            "formal Lean language statement:"
        )
    from datasets import load_dataset
    ds_test = load_dataset('UDACA/proofnet-v3-lean4', split='test')
    # ds_test = ds_test.select(list(range(10)))

    # Promptify & get Gold Truth Headers
    prompts = [my_prompt_format(row['nl_statement']) for row in ds_test]
    # model = None
    # prompts = [f"##\n{row['formal_statement']}\n##" for row in ds_test]
    # print(f'{prompts=}')
    gold_headers = [row['header_no_import'] for row in ds_test]
    # print(f'{gold_headers}=')
    print(f'Number prompts: {len(prompts)=}')
    print(f'Number of gold headers: {len(gold_headers)=}')

    # Start timer
    global_start_time = time.time()  # Start overall timer

    debug = True
    # debug = False
    eval_batch_size = 32 # for vllm how many prompts to batch for speed
    k = 5
    # num_samples = 20
    # num_samples = 40
    num_samples = 5000
    score = run_pass_k_eval(prompts, model, server, headers=gold_headers, k=k, num_samples=num_samples, eval_batch_size=eval_batch_size, debug=debug)
    print(f"\n==== For {model} Final Average Pass@{k=}N={num_samples} across {len(prompts)} tasks: {score:.3f} ====\n")

    # End overall timer
    global_end_time = time.time()
    total_seconds = global_end_time - global_start_time
    print(f"\nDone. Total run time for all models: {total_seconds:.2f} seconds, {total_seconds/60:.2f} minutes, {total_seconds/3600:.2f} hours.\a")


def main_experiment_pass_k_vs_N_config(config: dict = {}):
    """
    Runs the pass@k experiment for a series of N values defined in the config,
    repeating each experiment a specified number of times to measure variance,
    and then plots:
      (1) Pass@k (with 95% CIs) versus N, and 
      (2) Average evaluation time (with 95% CIs) versus N.

    Requirements:
    -----------------
    1. Read experimental parameters from the config dictionary:
         - 'n_start': Starting value for N (number of completions generated per prompt).
         - 'n_end': Ending value for N.
         - 'num_points': Number of points between n_start and n_end (if not provided, default is 10).
         - 'num_reps': Number of repetitions per N to estimate variance.
         - 'k': The k value for pass@k (e.g., pass@5).
         - 'plot_title': Title for the plots.
         - 'model': The model identifier used for code generation.
         - 'seed': Base seed for random number generators.
    2. Load prompts and gold headers from a dataset.
    3. Initialize a Lean 4 server using PyPantograph.
    4. For each N value (generated via np.linspace and rounded to integer):
         - Run the evaluation num_reps times.
         - For each repetition, update the random seed (base_seed plus an offset) to ensure variability.
         - Measure both the pass@k score and the evaluation time for that repetition.
    5. Compute mean, standard deviation, and 95% confidence intervals (CIs) for:
         - Pass@k scores, and
         - Evaluation times.
    6. Print summary results during the loop and detailed results at the end.
    7. Plot:
         - N (x-axis) versus average pass@k (y-axis) with 95% CI error bars.
         - N (x-axis) versus average evaluation time (y-axis) with 95% CI error bars.
    8. Log the plots to wandb.
    
    Returns:
         None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    from typing import List

    # ---------------------------------------------------------------------
    # Set the random seed from the configuration (default seed = 42)
    seed_everything(config.get('seed', 42))
    # conda activate zip_fit
    # conda activate zip_fit; export CUDA_VISIBLE_DEVICES=0; python ~/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py --model "gpt2" 
    # conda activate zip_fit; export CUDA_VISIBLE_DEVICES=1; python ~/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py --model "Qwen/Qwen2.5-0.5B" 
    # conda activate zip_fit; export CUDA_VISIBLE_DEVICES=2; python ~/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py --model "meta-llama/Llama-3.2-1B" 
    # conda activate zip_fit; export CUDA_VISIBLE_DEVICES=3; python ~/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py --model "google/gemma-2-2b'"
    # onda activate zip_fit; export CUDA_VISIBLE_DEVICES=3; python ~/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py --model "UDACA/math-gemma-2-2b-zipfit"
    # export CUDA_VISIBLE_DEVICES=3; python ~/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py --model "UDACA/math-gemma-2-2b-dsir"
    # export CUDA_VISIBLE_DEVICES=3; python ~/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py --model "UDACA/math-gemma-2-2b-less"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # ---------------------------------------------------------------------
    # Read configuration parameters with defaults
    n_start    = config.get('n_start', 50)          # Starting value for N
    n_end      = config.get('n_end', 500)           # Ending value for N
    num_points = config.get('num_points', 10)      # Number of N values between n_start and n_end
    num_reps   = config.get('num_reps', 5)           # Repetitions per N
    k          = config.get('k', 5)                # The k value for pass@k (e.g., pass@5)
    plot_title = config.get('plot_title', "Pass@5 vs. N and Evaluation Time")
    # model      = config.get('model', 'SmolLM-135M')
    model      = config.get('model', 'gpt2')
    model      = config.get('model', 'Qwen/Qwen2.5-0.5B')
    model      = config.get('model', 'meta-llama/Llama-3.2-1B')
    model      = config.get('model', 'google/gemma-2-2b')
    model      = config.get('model', 'UDACA/math-gemma-2-2b-zipfit')
    model      = config.get('model', 'UDACA/math-gemma-2-2b-less')
    model      = config.get('model', 'UDACA/math-gemma-2-2b-dsir')
    print(f'{model=}')
    
    # ---------------------------------------------------------------------
    # Load prompts and gold headers from a dataset.
    # We assume that our prompt formatting function is defined below.
    def my_prompt_format(nl_stmt: str) -> str:
        return (
            "Your task is translate the natural language version of the mathematical statement "
            "to a formal Lean statement version, using the following format:\n"
            "natural language statement:\nLet $z=\\frac{1+i}{\\sqrt{2}}.$What is $\\left(z^{1^2}+z^{2^2}+z^{3^2}+\\dots+z^{{12}^2}\\right) \\cdot "
            "\\left(\\frac{1}{z^{1^2}}+\\frac{1}{z^{2^2}}+\\frac{1}{z^{3^2}}+\\dots+\\frac{1}{z^{{12}^2}}\\right)?$ "
            "$\\textbf{(A)}\\ 18 \\qquad \\textbf{(B)}\\ 72-36\\sqrt2 \\qquad \\textbf{(C)}\\ 36 \\qquad \\textbf{(D)}\\ 72 \\qquad \\textbf{(E)}\\ 72+36\\sqrt2$ "
            "Show that it is \\textbf{(C)}\\ 36.\n"
            "formal Lean language statement:##\ntheorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) : "
            "(∑ k in Finset.Icc 1 12, (z^(k^2))) * (∑ k in Finset.Icc 1 12, (1 / z^(k^2))) = 36 := sorry\n##"
            "natural language statement:\nIntegers $x$ and $y$ with $x>y>0$ satisfy $x+y+xy=80$. What is $x$? "
            "$\\textbf{(A)}\\ 8 \\qquad\\textbf{(B)}\\ 10 \\qquad\\textbf{(C)}\\ 15 \\qquad\\textbf{(D)}\\ 18 \\qquad\\textbf{(E)}\\ 26$ Show that it is \\textbf{(E)}\\ 26.\n"
            "formal Lean language statement:##\ntheorem amc12a_2015_p10 (x y : ℤ) (h₀ : 0 < y) (h₁ : y < x) "
            "(h₂ : x + y + (x * y) = 80) : x = 26 := sorry\n##"
            "natural language statement:\nWhat is the [[volume]] of a [[cube]] whose [[surface area]] is twice that of a cube with volume 1? "
            "$\\mathrm{(A)}\\ \\sqrt{2}\\qquad\\mathrm{(B)}\\ 2\\qquad\\mathrm{(C)}\\ 2\\sqrt{2}\\qquad\\mathrm{(D)}\\ 4\\qquad\\mathrm{(E)}\\ 8$ Show that it is \\mathrm{(C)}.\n"
            "formal Lean language statement:##\ntheorem amc12a_2008_p8 (x y : ℝ) (h₀ : 0 < x ∧ 0 < y) "
            "(h₁ : y^3 = 1) (h₂ : 6 * x^2 = 2 * (6 * y^2)) : x^3 = 2 * Real.sqrt 2 := sorry\n##"
            f"natural language statement:\n{nl_stmt}\n"
            "formal Lean language statement:"
        )
    from datasets import load_dataset
    ds_test = load_dataset('UDACA/proofnet-v3-lean4', split='test')
    # ds_test = ds_test.select(list(range(185)))  # optionally select a subset
    prompts = [my_prompt_format(row['nl_statement']) for row in ds_test]
    gold_headers = [row['header_no_import'] for row in ds_test]
    print(f'Number prompts: {len(prompts)=}')
    print(f'Number of gold headers: {len(gold_headers)=}')
    
    # ---------------------------------------------------------------------
    # Generate a list of N values.
    # Instead of a fixed step, we generate "num_points" evenly spaced values from n_start to n_end.
    arr_float = np.linspace(n_start, n_end, num_points)
    # Round the floats to the nearest integer.
    N_values = np.rint(arr_float).astype(int)
    N_values = [int(N) for N in N_values]
    
    # ---------------------------------------------------------------------
    # Initialize lists to store statistics across N values:
    mean_passk_per_N = []  # Mean pass@k for each N
    std_passk_per_N = []   # Std dev of pass@k for each N
    ci_passk_array = []    # 95% CI for pass@k for each N
    avg_times = []         # Average evaluation time (sec) for each N
    time_stds = []         # Std dev of evaluation times for each N
    ci_time_array = []     # 95% CI for evaluation time for each N
    
    # ---------------------------------------------------------------------
    # Initialize the Lean 4 server via PyPantograph.
    from pantograph import Server
    server = Server(imports=["Mathlib", "Init"], project_path=os.path.expanduser("~/mathlib4"))
    
    print("Starting experiments...")
    global_start_time = time.time()  # Overall experiment timer
    
    # Get the base seed from the config (default 42)
    base_seed = config.get('seed', 42)
    
    # Loop over each N value (each representing the number of completions generated per prompt)
    for i, N in enumerate(tqdm.tqdm(N_values, desc="Evaluating N values")):
        print(f'{model=}')
        rep_start_time = time.time()  # Timer for all repetitions at this N
        
        # Lists to record pass@k scores and evaluation times for each repetition at this N
        passk_runs = []
        rep_times = []
        
        # Run the experiment num_reps times for the current N to assess variance.
        for rep in range(num_reps):
            # Calculate a new seed for this repetition to ensure variability.
            new_seed = base_seed + (i * num_reps) + rep
            seed_everything(new_seed)  # Set all random seeds
            
            # Record the start time for this repetition.
            start_time = time.time()
            # Call run_pass_k_eval which:
            #   - Generates 'N' completions per prompt,
            #   - Checks each for correctness (e.g., compilation),
            #   - Computes and returns the overall pass@k score.
            score = run_pass_k_eval(
                prompts=prompts,
                model=model,
                server=server,
                headers=gold_headers,
                k=k,
                num_samples=N,
                eval_batch_size=32,
                seed=new_seed,
                debug=False
            )
            # Record the end time and compute the elapsed time for this repetition.
            end_time = time.time()
            rep_time = end_time - start_time
            
            # Append the results from this repetition.
            passk_runs.append(score)
            rep_times.append(rep_time)

        rep_end_time = time.time()
        elapsed_N = rep_end_time - rep_start_time
        
        # Compute statistics for pass@k scores for this N.
        avg_passk = np.mean(passk_runs)
        std_passk = np.std(passk_runs, ddof=1)
        ci_passk = 1.96 * (std_passk / math.sqrt(num_reps))
        
        # Compute statistics for evaluation times for this N.
        avg_time = np.mean(rep_times)
        std_time = np.std(rep_times, ddof=1)
        ci_time = 1.96 * (std_time / math.sqrt(num_reps))
        
        # Save the computed statistics.
        mean_passk_per_N.append(avg_passk)
        std_passk_per_N.append(std_passk)
        ci_passk_array.append(ci_passk)
        avg_times.append(avg_time)
        time_stds.append(std_time)
        ci_time_array.append(ci_time)

        wandb.log({
                "N": N,
                "avg_passk": avg_passk,
                "std_passk": std_passk,
                "ci_passk": ci_passk,
                "avg_time": avg_time,
                "std_time": std_time,
                "ci_time": ci_time,
            })
        
        # Print a summary for the current N.
        print(f"N={N}, Pass@{k} mean={avg_passk:.4f}, stdev={std_passk:.4f}, CI(95%)={ci_passk:.4f}, "
              f"avg time={avg_time:.2f} sec, time CI={ci_time:.2f} sec for {num_reps} reps")
    
    # End overall experiment timer.
    global_end_time = time.time()
    total_seconds = global_end_time - global_start_time
    print(f"\nDone. Total experiment time: {total_seconds:.2f} seconds, "
          f"{total_seconds/60:.2f} minutes, {total_seconds/3600:.2f} hours.")
    
    # Print detailed results for each N value.
    print("\nDetailed Results:")
    for N, mean_val, std_val, ci_val, t_avg, t_ci in zip(N_values, mean_passk_per_N, std_passk_per_N, ci_passk_array, avg_times, ci_time_array):
        print(f"N={N}, Pass@{k} mean={mean_val:.4f}, stdev={std_val:.4f}, CI(95%)={ci_val:.4f}, "
              f"avg time={t_avg:.2f} sec, time CI={t_ci:.2f} sec")
    
    # ---------------------------------------------------------------------
    # Plotting:
    # Create two subplots: one for pass@k and one for evaluation time.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: N vs. Mean pass@k with 95% CI error bars.
    ax1.errorbar(N_values, mean_passk_per_N, yerr=ci_passk_array, fmt='o-', capsize=5, ecolor='red', color='blue')
    ax1.set_title(f"{plot_title} (Pass@{k})")
    ax1.set_xlabel("N (Number of completions generated)")
    ax1.set_ylabel(f"Mean Pass@{k} ± 95% CI")
    ax1.grid(True)
    
    # Plot 2: N vs. Average evaluation time with 95% CI error bars.
    ax2.errorbar(N_values, avg_times, yerr=ci_time_array, fmt='o-', capsize=5, ecolor='red', color='green')
    ax2.set_title("Evaluation Time vs. N")
    ax2.set_xlabel("N (Number of completions generated)")
    ax2.set_ylabel("Average Evaluation Time (sec) ± 95% CI")
    ax2.grid(True)
    
    # Save the figure locally.
    plot_file = "pass_at_k_and_time_plot.png"
    fig.savefig(plot_file)
    print(f"\nPlot saved to {plot_file}")
    
    # Log the plot image to wandb.
    wandb.log({"pass_at_k_and_time_plot": wandb.Image(plot_file)})
    
    # Display the plots.
    plt.show()


def _main(**kwargs):
    from datetime import datetime
    from socket import gethostname
    import wandb
    today = datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss') # eg '2024_m01_d22_t13h_00m_30s'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(kwargs.get('CUDA_VISIBLE_DEVICES', '7'))
    tmux_sess_num = None
    kwargs = kwargs | {'today': today, 'tmux_sess_num': tmux_sess_num, 'hostname': gethostname()}
    run_name = f'{kwargs}' 
    run = wandb.init(mode=kwargs.get('mode', 'online'), project="zip-fit-pass-at-k-af", name=run_name, save_code=True, config=kwargs)
    # run = wandb.init(mode=kwargs.get('mode', 'dryrun'), project="zip-fit-pass-at-k-af", name=run_name, save_code=True, config=kwargs)
    wandb.save(__file__) # save current code now, don't wait to wandb.finish, also useful: wandb.save("*.py") # upload all .py files in current directory
    print(f'Kwargs to run:\n{kwargs}')
    # main(kwargs)
    main_experiment_pass_k_vs_N_config(kwargs)
    run.alert(title="Run Completed", text=f"Run finished, run url: {run.get_url()}")
    print(f'{run.get_url()=}')
    wandb.finish()

if __name__ == "__main__":
    import fire
    fire.Fire(_main)
