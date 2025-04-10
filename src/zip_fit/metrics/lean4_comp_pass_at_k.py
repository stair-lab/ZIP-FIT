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

from lean4_utils import parse_lean_completion, get_list_lean4_syntax_errors

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Computes the unbiased pass@k metric from the Codex/HumanEval paper [Chen et al., 2021],
    using a numerically stable product formulation:

        pass@k = 1 - ( (n-c choose k) / (n choose k) )
                = 1 - ∏(i=0 to k-1) [ (n - c - i) / (n - i) ]
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

def lean4_comp_pass_at_k_unbiased(
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
    Evaluate pass@k with lean4 compilation for each docstring completed with LM.
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

def lean4_comp_pass_at_k_unbiased_log_per_completion(
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
    check how many compile, then compute pass@k for each batch.
    
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