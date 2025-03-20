# metrics/lean4_comp_pass_k_unbiased.py
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
import numpy as np
from typing import List, Optional
import tqdm

from vllm import LLM, SamplingParams
from vllm import RequestOutput

from pantograph.server import Server
from pantograph.data import CompilationUnit, TacticInvocation

def parse_lean_completion(llm_output: str) -> str:
    """
    Extracts the Lean theorem from the LLM output, which is enclosed between '##' markers.
    Returns the extracted theorem as a string.

    - Uses regex to find the first occurrence of text between '##' markers.
    - If no match is found, returns an empty string.

    Example:
    ----------
    Input:
        "
        natural language statement:
        /-- Expand the following expression: $7(3y+2)$ Show that it is 21y+14.-/
        formal language statement:##
        theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by
        ##
        "

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
    - No large factorial or gamma function calls, so it's efficient & stable for typical n.

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


def check_lean_compiles_strict(lean_snippet: str, server: Server, require_no_goals: bool = True) -> bool:
    """
    Strictly checks whether a Lean 4 snippet "fully compiles" according to PyPantograph,
    by analyzing the returned CompilationUnits from server.load_sorry(...).
    Note: if input is empty string "" we return False even if Lean accepts it because it means the LLM outputed something we couldn't parse most likely. 

    Steps & Rationale
    -----------------
    1) **Load the snippet**:
       We call `server.load_sorry(snippet)`, which parses the snippet as if it were
       a Lean 4 source file. Note that Lean can accept multiple definitions/
       theorems in a single snippet, each mapped to a "compilation unit."

       - If a parse-level or "fatal" server error occurs, `load_sorry` can raise
         an exception. We catch that and return False.

    2) **Check each CompilationUnit**:
       - Each "compilation unit" corresponds to an area in the snippet that Lean
         recognized (like "theorem lemma1 : p -> p := by ...").
       - If `cu.messages` contains strings with "error" in them, we assume a
         compilation error was detected. We return False.

    3) **Leftover goals**:
       - If `require_no_goals` is True, we also fail if `cu.goal_state` is not None.
         This typically means Lean recognized a "sorry" or leftover proof hole,
         or some type error that was turned into a goal. 
         e.g. "theorem lemma_fail : 2 + 2 = 5 := by rfl" might produce leftover
         type mismatch goals if Lean tries to unify 4 with 5.
       - If `require_no_goals` is False, we only fail on "hard" errors, ignoring
         partial or logically incorrect proofs as long as they parse.

    4) **Return**:
       - True if we never encountered a parse error, no messages with "error,"
         and (optionally) no leftover goals (if `require_no_goals=True`).
       - False otherwise.

    Why This Matters
    ----------------
    - Lean 4's design allows partial type errors or leftover subgoals to
      accumulate without halting compilation. So code might "parse" but still
      be incomplete or contradictory. For a "strict" notion of success, we
      treat leftover goals as a fail, but for "syntactic-only" we can ignore
      them. 
    - This approach ensures you can systematically measure how many LLM
      completions produce truly "fully compiled" Lean 4 code.

    Args:
        snippet (str):
          A string containing what we'd treat as top-level Lean 4 code.
        server (Server):
          A PyPantograph server instance, e.g. `Server()`.
        require_no_goals (bool):
          If True, leftover subgoals => fail. If False, leftover subgoals do not
          matter for success/failure.

    Returns:
        bool:
          True if no parse error, no "error" messages, and (if `require_no_goals=True`)
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

def run_lean4_comp_pass_k_unbiased_eval(
    prompts: List[str],
    model_name: str, 
    server: Server,
    headers: Optional[List[str]] = None,  # If None, parse the headers from model_name generation
    k: int = 5,
    num_samples: int = 30, # the big N in pass@k - total number of completions generated
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
    if model_name is not None:
        llm = LLM(
            model=model_name,                         # create vLLM model_name from path or HF name
            dtype=dtype,                         # specify float precision
            trust_remote_code=True,              # allow custom model_name code
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
        raise ValueError("Model is None, so we can't do pass@k with the given list of strings")

    # Now compute pass@k for each prompt individually
    pass_vals: List[float] = []
    avg_pass_val: float = 0.0
    var_pass_val: float = 0.0
    for completions_for_prompt in text_outputs:
        # Check which completions compile => c = sum of successes
        parsed_completions: List[str] = [parse_lean_completion(c) for c in completions_for_prompt]
        mdl_code_with_true_header: List[str] = [f'{header}\n\n{parsed_comp}' for parsed_comp, header in zip(parsed_completions, headers)]
        successes: List[bool] = [len(get_list_lean4_syntax_errors(lean_code, server)) == 0 for lean_code in mdl_code_with_true_header]
        c: int = sum(successes)
        pass_val: float = pass_at_k(num_samples, c, k)
        # Update running average and variance
        pass_vals.append(pass_val)
        avg_pass_val += pass_val / len(pass_vals)
        var_pass_val = np.var(pass_vals)
        std_pass_val = np.std(pass_vals)
        conf_int_95 = 1.96 * std_pass_val / np.sqrt(len(pass_vals))
        # Log
        print(f'current pass@k(x,p) ({model_name}, {k=}, {num_samples=}): {pass_val=}')
        print(f'avg pass@k(D,p) ({model_name}, {k=}, {num_samples=}): {avg_pass_val=}')
        print(f'var pass@k(D,p) ({model_name}, {k=}, {num_samples=}): {var_pass_val=}')
        print(f'std pass@k(D,p) ({model_name}, {k=}, {num_samples=}): {std_pass_val=}')
        print(f'95% confidence interval pass@k(D,p) ({model_name}, {k=}, {num_samples=}): {conf_int_95=}')
        wandb.log({"pass@k(x,p)": pass_val})
        wandb.log({"avg pass@k(D,p)": avg_pass_val})
        wandb.log({"var pass@k(D,p)": var_pass_val})
        wandb.log({"std pass@k(D,p)": std_pass_val})
        wandb.log({"95% confidence interval pass@k(D,p)": conf_int_95})

    # Finally, average pass@k across all prompts
    print(f'{len(pass_vals)=}') if debug else None
    import gc
    del llm
    gc.collect()
    return float(np.mean(pass_vals)) if pass_vals else 0.0

def run_lean4_comp_pass_k_unbiased_eval_log_per_completion(
    prompts: List[str],
    model_name: str, 
    server: Server,
    headers: Optional[List[str]] = None,  # If None, parse the headers from model_name generation
    k: int = 5,
    num_samples: int = 30, # the big N in pass@k - total number of completions generated
    dtype: str = "bfloat16",
    eval_batch_size: int = 32,
    max_tokens: int = 512,
    top_p: float = 0.95,       # Nucleus sampling: limits token choices to the smallest set whose cumulative probability is >= top_p (e.g., 95%).
    top_k: int = 50,           # Top-k sampling: limits token choices to the top-k most likely tokens at each step (e.g., top 50 tokens).
    temperature: float = 0.7,
    seed: int = 42,
    debug: bool = False,
) -> float:
    """
    Evaluate pass@k for each docstring with per-batch logging: generate code with the model,
    check how many compile under strict rules, then compute pass@k for each batch.
    
    This variant logs metrics after each batch of completions is generated, allowing for
    real-time monitoring of model performance during longer evaluation runs.
    """
    # Check if wandb is available for logging
    try:
        import wandb
        wandb_available: bool = wandb.run is not None
    except ImportError:
        wandb_available: bool = False
        print("wandb not installed or not initialized, logging to console only")
    
    if model_name is not None:
        llm = LLM(
            model=model_name,             # create vLLM model from path or HF name
            dtype=dtype,                  # specify float precision
            trust_remote_code=True,       # allow custom model code
            seed=seed,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            n=num_samples,
        )

        # For each batch of prompts, call llm.generate for paralellization
        batched_prompts: List[List[str]] = [prompts[i:i + eval_batch_size] for i in range(0, len(prompts), eval_batch_size)]
        print(f'Number of batches of prompts each of size {eval_batch_size}: {len(batched_prompts)}')

        # For each batch of prompts, call llm.generate and immediately evaluate
        # batch_num: Tracks which batch is currently being processed. Critical for:
        # 1. Logging: Creates unique identifiers for metrics in wandb to prevent overwriting
        # 2. Header matching: Calculates prompt_idx to correctly pair prompts with their headers
        #   otherwise, the lean server won't get the code with the right header.
        # 3. Progress tracking: Provides meaningful batch numbers in status updates
        all_pass_vals: List[float] = []
        for batch_num, batch_prompts in enumerate(tqdm.tqdm(batched_prompts, desc="Processing batches")):
            # Generate completions for this batch (in parallel)
            batch_outputs: List[RequestOutput] = llm.generate(batch_prompts, sampling_params)
            # Get the pass@k for each completion in the batch
            for i, request_out in enumerate(batch_outputs):
                # Get all completions for this prompt
                completions: List[str] = [completion.text for completion in request_out.outputs]
                # Parse out llm completions
                parsed_completions: List[str] = [parse_lean_completion(c) for c in completions]
                
                # Calculate the absolute prompt index in the original prompts list
                absolute_prompt_idx: int = batch_num * eval_batch_size + i
                # Add headers for each completion so lean doesn't give us errors by default (false negatives)
                if headers is None:
                    prompt_headers: List[str] = [""] * len(parsed_completions)
                else:
                    # Use the header corresponding to this prompt
                    prompt_headers: List[str] = [headers[absolute_prompt_idx]] * len(parsed_completions)
                # Combine headers with parsed completions
                mdl_code_with_header: List[str] = [f'{header}\n\n{parsed_comp}' for parsed_comp, header in zip(parsed_completions, prompt_headers)]
                
                # Check which completions compile
                successes: List[bool] = [len(get_list_lean4_syntax_errors(lean_code, server)) == 0 for lean_code in mdl_code_with_header]
                c: int = sum(successes)
                
                # Calculate pass@k for this prompt
                pass_val: float = pass_at_k(num_samples, c, k)
                all_pass_vals.append(pass_val)
                
                # Calculate running statistics after each prompt
                avg_pass: float = np.mean(all_pass_vals)
                std_pass: float = np.std(all_pass_vals, ddof=1)
                
                # Log per-prompt metrics with running statistics
                prompt_log = {
                    "prompt_idx": absolute_prompt_idx,
                    "pass@k(x,p)": pass_val,
                    "correct_count": c,
                    "avg_pass@k(D,p)": avg_pass,
                    "std_pass@k(D,p)": std_pass,
                }
                if wandb_available:
                    wandb.log(prompt_log)
                if debug:
                    print(f"Prompt {absolute_prompt_idx}: pass@k={pass_val}, correct={c}/{num_samples}")
    else:
        raise ValueError("Model is None, so we can't do pass@k with the given list of strings")
    # Print summary to console
    final_avg_pass_val: float = np.mean(all_pass_vals)
    print(f"\nFinal Results:")
    print(f"Average pass@k: {final_avg_pass:.4f}")
    print(f"Total problems evaluated: {len(all_pass_vals)}")

    # Clean up
    import gc
    del llm
    gc.collect()
    return final_avg_pass_val