# ref: https://chatgpt.com/c/673ea80c-7ce0-8001-a42c-b47891b0f0ed
from datetime import datetime
from transformers import TrainerCallback
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import wandb
import torch
from _tfa import compute_tfa

class TFACallback(TrainerCallback):
    def __init__(self, tokenizer, train_input_texts, eval_input_texts=None):
        """
        A custom callback to compute and log Teacher-Forced Accuracy (TFA) for both training and evaluation datasets.

        Parameters:
            tokenizer: The tokenizer corresponding to the model.
            train_input_texts: List of input texts for computing TFA during training.
            eval_input_texts: List of input texts for computing TFA during evaluation (optional).
        """
        self.tokenizer = tokenizer
        self.train_input_texts = train_input_texts
        self.eval_input_texts = eval_input_texts or train_input_texts  # Use training inputs if evaluation inputs are not provided

    def on_step_end(self, args, state, control, **kwargs):
        """
        Compute and log TFA for training at the end of each step.
        """
        trainer = kwargs["trainer"]
        model = trainer.model
        tfa_score = compute_tfa(model, self.tokenizer, self.train_input_texts)
        
        # Log training TFA to console and W&B
        print(f"Step {state.global_step}: TFA (train) = {tfa_score:.4f}")
        if trainer.is_world_process_zero():
            wandb.log({"TFA (train)": tfa_score, "step": state.global_step})

    def on_evaluate(self, args, state, control, **kwargs):
        """
        Compute and log TFA for evaluation during evaluation phases.
        """
        trainer = kwargs["trainer"]
        model = trainer.model
        tfa_score = compute_tfa(model, self.tokenizer, self.eval_input_texts)
        
        # Log evaluation TFA to console and W&B
        print(f"Evaluation at step {state.global_step}: TFA (eval) = {tfa_score:.4f}")
        if trainer.is_world_process_zero():
            wandb.log({"TFA (eval)": tfa_score, "step": state.global_step})

class GenCallbackHFGen(TrainerCallback):
    # \n\nProblem:\n{$PROBLEM}\n\nSolution:
    def __init__(self, mdl: AutoModelForCausalLM, tok: AutoTokenizer, prompt_template: str = "Problem:\n{$PROBLEM}\n\nSolution:", prompt_nickname: str = 'default_prompt', config = 'not set'):
        """
        Callback for generating predictions using the text-generation pipeline.
        
        Args:
            gen: An instance of HFPipelineGenerator initialized with a Hugging Face pipeline.
            tokenizer: Tokenizer associated with the model.
        """
        self.mdl = mdl
        self.tok = tok
        self.prompt_template = prompt_template
        self.prompt_nickname = prompt_nickname
        self.config = config

    def on_evaluate(self, args, state, control, **kwargs):
        """ Generate predictions using the pipeline during evaluation."""
        # - Ensure we have access to the evaluation dataloader and model
        # eval_dataloader = kwargs.get('eval_dataloader')
        eval_dataloader = kwargs.get('train_dataloader')
        print(f'Dataloader in callback: {eval_dataloader.base_dataloader.dataset.split}')
        if eval_dataloader is None:
            print("Evaluation dataloader not available. Skipping text generation.")
            return
        
        # - Starting Text Generation 
        print()
        for batch in eval_dataloader:
            # Prepare prompts
            input_ids = batch["input_ids"] # has bos, eos, pad since eval loader data has same preprocessing as train loader
            input_texts = self.tok.batch_decode(input_ids, skip_special_tokens=True) # won't have <bos>, <eos>, <pad> etc
            prompt_texts: list[str] = self.get_inference_prompts(input_texts) # TODO maybe more efficient, remove all tok id in tensor frm eos forward. Then only batch decde those
            self.tok.padding_side = "left" # left needed for inference! or it will start trying to generate starting from a pad token!? https://chatgpt.com/c/675b3a77-667c-8001-a087-33c2b4ae96be
            prompt_ids: dict = self.tok(prompt_texts, add_special_tokens=False, return_tensors='pt', truncation=True, padding=True) # not traning and <bos> already added, so no need to add anything else
            self.tok.padding_side = "right" # only fortrain
            # https://chatgpt.com/g/g-p-6760861c4ef881918d7582c9afe11a48-logic-contamination/c/6762553f-37d8-8001-8f42-daae269da2c8
            # truncaton true to max length and padding also ok to that length so the questions are sent to the model and padding is ignroed usually so likely gens won't be affect much? this is a debugging ste anyway
            # Get the device of the embeddings (first submodule) dynamically
            # input_device = self.mdl.get_input_embeddings().weight.device
            # Move the input_ids to the same device as the embeddings
            # prompt_ids = prompt_ids.to(input_device)
            prompt_ids = prompt_ids["input_ids"].to(self.mdl.device)
            # prompt_ids = prompt_ids["input_ids"]
            # print(f'\nChecking size of prompt going into model: {prompt_ids.size()=}\n')
            # Get generations
            gen_ids = self.mdl.generate(prompt_ids, max_new_tokens=512)
            gen_texts = self.tok.batch_decode(gen_ids, skip_special_tokens=True) # no special toks needed during gen for sanity checking & computing diff with true soln
            # Print prompt generation pairs
            batch_size = input_ids.size(0)
            normalized_edit_dist_one_bach: list[float] = []
            for batch_idx in range(batch_size):
                input_text: str = input_texts[batch_idx]
                prompt: str = prompt_texts[batch_idx]
                gen_text: str = gen_texts[batch_idx]
                print(f'\n{"---"*15}\nInput Full Text:\n{input_text}')
                print(f'\n{"---"*15}\nInput Prompt Text:\n{prompt}')
                print(f'\n{"---"*15}\nGeneration:\n{gen_text}')
                # -- Print Gold Soln vs Gen Soln statistics
                soln: str = input_text.split('\n\nSolution:')[-1]
                gen_soln: str = gen_text.split('\n\nSolution:')[-1]
                
                import Levenshtein
                dist = Levenshtein.distance(soln, gen_soln)
                normalized_dist = dist / len(soln)
                normalized_edit_dist_one_bach.append(normalized_dist)

                print()
                print("=="*15 + f'<begin> Solutions Comparison (with Prompt nickname: {self.prompt_nickname})')
                print(f'{"--"*10} Gold Solution:\n{soln}')
                print()
                print(f'{"--"*10} Generated Solution:\n{gen_soln}')
                print(f'\nNormalized edit dist: {normalized_dist}')
                print("=="*15 + f'<end> Solutions Comparison (with Prompt nickname: {self.prompt_nickname})')
                print()

            # log_dict = {f"eval_gen/{self.prompt_nickname}/edit_dist": dist, f"eval_gen/{self.prompt_nickname}/norm_avg_edit_dist": normalized_dist, "step": state.global_step}
            avg_normalized_edit_dist: float = float(torch.tensor(normalized_edit_dist_one_bach).cpu().mean())
            log_dict = {f"eval_gen/{self.prompt_nickname}/norm_avg_edit_dist": avg_normalized_edit_dist, "step": state.global_step}
            print(f'\n{log_dict}')
            wandb.log(log_dict)
            print()
            break
        print()
        return
    
    def get_inference_prompts(self, input_texts: list[str]) -> list[str]:
        prompts: list[str] = []
        for input_text in input_texts:
            problem: str = input_text.split('\n\nSolution:')[0]
            problem = problem.replace('Problem:\n', '') # now new line in front cuz that wan't added to the putnam data in preprocessing
            assert self.tok.bos_token not in problem, f'Shouldnt have bos but it does: {problem=}'
            prompt = self.input_text_to_prompt(problem)
            prompts.append(prompt)
        return prompts
    
    def input_text_to_prompt(self, input_text: str) -> str:
        # Note: only need to add <bos> token at front during inference, other special tokens not needed
        return self.tok.bos_token + self.prompt_template.replace('{$PROBLEM}', input_text)

# class CallBackLmEval(TrainerCallback):
#     # \n\nProblem:\n{$PROBLEM}\n\nSolution:
#     def __init__(self, mdl: AutoModelForCausalLM, 
#                  tok: AutoTokenizer, 
#                  prompt_template: str = "Problem:\n{$PROBLEM}\n\nSolution:", 
#                  prompt_nickname: str = 'default_prompt', 
#                  config = 'not set'
#                  ):
#         self.mdl = mdl
#         self.tok = tok
#         self.prompt_template = prompt_template
#         self.prompt_nickname = prompt_nickname
#         self.config = config

#     def on_evaluate(self, args, state, control, **kwargs):
#         """ Generate predictions using the pipeline during evaluation."""
#         # - Ensure we have access to the evaluation dataloader and model
#         eval_dataloader = kwargs.get('eval_dataloader')
#         if eval_dataloader is None:
#             print("Skipping Lm Eval CallBack.")
#             return

#         # sanity checks

#         # set model to vllm on 2nd gpu
#         from eval_logic_contamination import do_lm_eval, print_and_wandb_log_lm_eval_results

#         results: dict = do_lm_eval(model_name, task, kwargs)
#         print(f"Arguments: {results['samples'][task][0]['arguments'][0][0]=}\nResponses: {results['samples'][task][0]['resps'][0][0]=}")
#         print(f'Keys for Lm Eval: {results.keys()=}\nKeys for Lm Eval Task: {results["results"][task].keys()=}')



        

