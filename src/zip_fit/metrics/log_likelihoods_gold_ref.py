import os

# Rok asked us to include the following specifications in our code to prevent CPUs from spinning idly:
n_threads_str = "4"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "True"

# This is needed for deterministic to work.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import gc
import numpy as np
import pandas as pd
import pathlib
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List, Optional, Tuple, Union
from vllm import LLM, SamplingParams, RequestOutput
from vllm.distributed.parallel_state import destroy_model_parallel

import src.data


def compute_answer_log_likelihoods_from_model(
    model_nickname: str = "Pythia_160M_300B",
    dataset: str = "math",
    num_prompts_to_use: int = 100,
):
    model_outputs_dir = os.path.join(
        "eval_results",
        dataset,
        model_nickname,
    )
    os.makedirs(model_outputs_dir, exist_ok=True)

    data: Dict[str, List[str]] = src.data.create_prompts_and_answers(
        dataset=dataset,
        num_prompts_to_use=num_prompts_to_use,
    )
    compute_answer_log_likelihoods_from_model_and_write_to_disk(
        dataset=dataset,
        data=data,
        model_nickname=model_nickname,
        model_sampled_outputs_dir=model_outputs_dir,
    )


def compute_answer_log_likelihoods_from_model_and_write_to_disk(
    dataset: str,
    model_sampled_outputs_dir: str,
    data: Dict[str, Union[List[str], List[int]]],
    model_nickname: str,
):
    # Load the model_name_or_path and revision
    models_df = pd.read_csv("metadata/models.csv")
    model_row = models_df[models_df["Model Nickname"] == model_nickname].iloc[0]
    model_name_or_path = model_row["HuggingFace Path"]
    revision = model_row["HuggingFace Revision"]
    kwargs = {
        "model": model_name_or_path,
        "dtype": "bfloat16",
    }
    if ~pd.isna(revision):
        kwargs["revision"] = revision

    # Create output filepath
    model_outputs_filepath = (
        pathlib.Path(model_sampled_outputs_dir) / "log_likelihoods.parquet"
    )

    problems_indices: List[int] = data["problems_indices"]
    prompts: List[str] = data["prompts"]
    levels = data["levels"]
    problem_types = data["problem_types"]
    solutions = data["solutions"]
    prompts_and_solutions = data["prompts_and_solutions"]

    # Skip if file exists and is not empty
    if model_outputs_filepath.exists() and model_outputs_filepath.stat().st_size > 0:
        results_df = pd.read_parquet(model_outputs_filepath)
        unique_existing_problem_indices = set(results_df["prompt_idx"].unique())
        unique_target_problem_indices = set(problems_indices)
        if unique_target_problem_indices == unique_existing_problem_indices:
            print(
                f"File {model_outputs_filepath} already exists. Skipping computation."
            )
            return

    # Load the model
    model = LLM(**kwargs)
    print(f"Loaded model {model_name_or_path} and optional revision {revision}.")

    # Configure sampling parameters for log probability computation.
    # Setting temperature=0 for deterministic logprobs, but requesting 1 token as required
    model_sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        max_tokens=1,  # vLLM requires at least 1 token
        logprobs=True,
        prompt_logprobs=1,
        seed=0,
    )

    # Initialize list to store results
    results_list = []

    for prompt_idx, prompt, solution, prompt_and_solution in zip(
        problems_indices,
        prompts,
        solutions,
        prompts_and_solutions,
    ):
        # Compute log probabilities for both the prompt-only and prompt+solution
        prompt_outputs: List[RequestOutput] = model.generate(
            prompts=[prompt], sampling_params=model_sampling_params
        )
        prompt_and_solution_outputs: List[RequestOutput] = model.generate(
            prompts=[prompt_and_solution], sampling_params=model_sampling_params
        )

        # Exclude the first (unconditioned) token and the last unnecessarily generated token.
        prompt_token_ids = prompt_outputs[0].prompt_token_ids[1:-1]
        prompt_logprobs = np.array(
            [
                prompt_outputs[0].prompt_logprobs[token_idx][token_id].logprob
                # Start counting from 1 because 0 corresponds to the <START> token.
                for token_idx, token_id in enumerate(prompt_token_ids, 1)
            ]
        )
        prompt_and_solution_token_ids = prompt_and_solution_outputs[0].prompt_token_ids[
            1:-1
        ]
        prompt_and_solution_logprobs = np.array(
            [
                prompt_and_solution_outputs[0]
                .prompt_logprobs[token_idx][token_id]
                .logprob
                # Start counting from 1 because 0 corresponds to the <START> token.
                for token_idx, token_id in enumerate(prompt_and_solution_token_ids, 1)
            ]
        )
        # # Sanity check: assert that the leading log probabilities are all the same.
        # # TODO: Debug why vLLM doesn't return consistent log probs for the prefix string.
        # assert np.all(
        #     np.isclose(
        #         prompt_logprobs, prompt_and_solution_logprobs[: len(prompt_logprobs)]
        #     )
        # )

        solution_logprobs = prompt_and_solution_logprobs[len(prompt_logprobs) :]
        token_indices_excluding_prompt = 1 + np.arange(len(solution_logprobs))
        token_indices_including_prompt = (
            len(prompt_token_ids) + token_indices_excluding_prompt
        )

        model_nickname_list = [model_nickname] * len(solution_logprobs)
        prompt_idx_list = [prompt_idx] * len(solution_logprobs)

        # Append result to list
        results_list.append(
            pd.DataFrame.from_dict(
                {
                    "Model Nickname": model_nickname_list,
                    "prompt_idx": prompt_idx_list,
                    "solution_logprobs": solution_logprobs.tolist(),
                    "token_idx_excluding_prompt": token_indices_excluding_prompt.tolist(),
                    "token_idx_including_prompt": token_indices_including_prompt.tolist(),
                }
            )
        )

        # Print progress
        print(f"Processed prompt {prompt_idx}")

        # Clean up memory
        del prompt_outputs, prompt_and_solution_outputs
        gc.collect()

    # Create and save single DataFrame with all results
    results_df = pd.concat(results_list).reset_index(drop=True)
    model_outputs_filepath.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(model_outputs_filepath, index=False)
    print(f"Saved all log likelihoods to {model_outputs_filepath}")

    # Clean up model resources
    destroy_model_parallel()
    del model.llm_engine.model_executor.driver_worker
    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(7)
    print("Finished cleaning up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate outputs from a language model."
    )
    parser.add_argument(
        "--model_nickname",
        type=str,
        required=True,
        help="Path or name of the model to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        # default="gsm8k",
        # default="humaneval",
        default="math",
        help="Dataset to use",
    )
    parser.add_argument(
        "--num_prompts_to_use", type=int, default=96, help="Number of prompts to use"
    )
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    compute_answer_log_likelihoods_from_model(
        model_nickname=args.model_nickname,
        dataset=args.dataset,
        num_prompts_to_use=args.num_prompts_to_use,
    )
    print("Finished compute_answer_log_likelihoods.py!")