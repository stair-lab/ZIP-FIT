from codeop import Compile
import os
import numpy as np
import torch
from typing import Dict, List, Union
from vllm import LLM, SamplingParams
import gc
from datasets import Dataset

def set_performance_env_vars(n_threads: int = 4):
    """Set environment variables according to Rok asked us to include the following specifications in our code to prevent CPUs from spinning idly:"""
    n_threads_str = str(n_threads)
    os.environ["OMP_NUM_THREADS"] = n_threads_str
    os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
    os.environ["MKL_NUM_THREADS"] = n_threads_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
    os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "True"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For deterministic operations


def compute_log_likelihood(
    model_path: str,
    prompt: str,
    gold_reference: str,
    dtype: str = "bfloat16",
    set_rok_performance_env_vars: bool = False, 
    trust_remote_code: bool = True,
) -> Dict[str, Union[float, List[float]]]:
    """
    Compile log likelihood of a gold reference given a prompt using vLLM.
    
    Args:
        model_path: HuggingFace path or local path to the model
        prompt: The input prompt text
        gold_reference: The gold reference text to compute likelihood for
        dtype: Model precision ("bfloat16", "float16", etc.)
        set_rok_performance_env_vars: Whether to set performance environment variables
        trust_remote_code: Whether to trust remote code when loading the model
        
    Returns:
        Dictionary with token-level and sequence-level log probabilities
    """
    # Set performance environment variables
    if set_rok_performance_env_vars:
        set_performance_env_vars()
    
    # Load model with vLLM
    model = LLM(model=model_path, dtype=dtype, trust_remote_code=trust_remote_code)
    print(f"Loaded model from {model_path}")
    
    # Prepare full text (prompt + gold reference)
    prompt_and_reference = prompt + gold_reference
    
    # Configure sampling parameters for log probability computation
    sampling_params = SamplingParams(
        n=1,                # Number of generated sequences per prompt (just 1 completion needed)
        temperature=1.0,    # Controls randomness: 1.0 means standard sampling without scaling
        max_tokens=1,       # Only generate 1 new token - minimum required by vLLM for logprobs
        logprobs=True,      # Enables calculation of log probabilities for generated tokens
        prompt_logprobs=1,  # Calculate log probabilities for prompt tokens as well
        seed=0,             # Random seed for reproducibility of generation
    )
    
    try:
        # Get log probabilities for prompt only
        prompt_outputs = model.generate(
            prompts=[prompt], sampling_params=sampling_params
        )
        
        # Get log probabilities for prompt + (gold) reference/answer
        full_outputs = model.generate(
            prompts=[prompt_and_reference], sampling_params=sampling_params
        )
        
        # Extract token IDs and logprobs
        prompt_token_ids = prompt_outputs[0].prompt_token_ids[1:-1]
        full_token_ids = full_outputs[0].prompt_token_ids[1:-1]
        
        # Calculate token-level log probabilities
        prompt_logprobs = np.array([
            prompt_outputs[0].prompt_logprobs[token_idx][token_id].logprob
            for token_idx, token_id in enumerate(prompt_token_ids, 1)
        ])
        
        full_logprobs = np.array([
            full_outputs[0].prompt_logprobs[token_idx][token_id].logprob
            for token_idx, token_id in enumerate(full_token_ids, 1)
        ])
        
        # Extract logprobs for just the (gold) reference/answer portion
        reference_logprobs = full_logprobs[len(prompt_logprobs):]
        
        # Calculate overall sequence log probability
        sequence_logprob = float(np.sum(reference_logprobs))
        
        return {
            "token_logprobs": reference_logprobs.tolist(),
            "sequence_logprob": sequence_logprob,
            "per_token_avg_logprob": float(np.mean(reference_logprobs))  # this is what we want -- it's log likelihood of gold reference given prompt.
        }
        
    finally:
        # Clean up resources
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            del model.llm_engine.model_executor.driver_worker
        except:
            pass
        del model
        gc.collect()
        torch.cuda.empty_cache()


def compute_log_likelihood_for_dataset(
    sub_ds: Dataset,
    model_path: str,
    dtype: str = "bfloat16",
    debug: bool = False,
    trust_remote_code: bool = True,
) -> float:
    """
    Process an entire subset of data (sub_ds) and compute the average log likelihood of gold references across all examples.

    Parameters:
      sub_ds: The subset of the dataset (like a HuggingFace 'Dataset' slice) with 'prompt' and 'gold_response' fields.
      model_path: Path to the model on HuggingFace or local path.
      dtype: Model precision ("bfloat16", "float16", etc.)
      debug: Whether to print debug information.
      trust_remote_code: Whether to trust remote code when loading the model

    Returns:
      float: The average per-token log probability across all examples.
    """
    sum_logprob = 0.0
    count = 0

    for i, example in enumerate(sub_ds):
        prompt = example["prompt"]
        gold_response = example["gold_response"]

        # Use the existing compute_log_likelihood function
        result: Dict = compute_log_likelihood(
            model_path=model_path,
            prompt=prompt,
            gold_reference=gold_response,
            dtype=dtype,
            trust_remote_code=trust_remote_code
        )
        
        # Get the average log probability per token
        per_token_avg_logprob = result["per_token_avg_logprob"]
        
        sum_logprob += per_token_avg_logprob
        count += 1

        if debug:
            print(f" Example {i}: Avg LogProb = {per_token_avg_logprob:.4f}")

    return sum_logprob / count if count > 0 else 0.0
