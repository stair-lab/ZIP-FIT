from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems
import torch
import json
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from itertools import chain
import os
import tqdm

def load_and_process_datasets(dataset_name: str, config_name: str = None, subset_size: int = 128, text_field: str = 'prompt', split: str = 'test', text_field2: str = "canonical_solution") -> list:
    """Loads a subset of a dataset from Hugging Face and processes it for similarity computation."""
    if config_name:
        dataset = load_dataset(dataset_name, config_name, split=split, streaming=True, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
    text_data = []
    for i, item in enumerate(dataset):
        if i >= subset_size:
            break
        if text_field2:
            text = f'Problem Description {item[text_field]} \n {item[text_field2]}'            
            text_data.append(text)
        else:
            text = item[text_field]
            text_data.append(text)
    return text_data[83:]

class CustomDataset(TorchDataset):
    def __init__(self, tokenized_blocks):
        self.input_ids = tokenized_blocks
        self.labels = tokenized_blocks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def create_blocks(text_data, tokenizer, block_size):
    """Create blocks of tokens from text data."""
    eos_token_id = tokenizer.eos_token_id
    concatenated_tokens = list(chain(*[tokenizer(text)['input_ids'] + [eos_token_id] for text in text_data]))
    
    total_length = len(concatenated_tokens)
    total_length = (total_length // block_size) * block_size
    
    all_tokens = [concatenated_tokens[i: i + block_size] for i in range(0, total_length, block_size)]
    return all_tokens

def fine_tune_model(sequences, model_id, hf_token, output_dir, cache_dir):
    """Fine-tune the model and return the path to the fine-tuned model."""
    print("Initializing model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        attn_implementation='eager',
        token=hf_token,    
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=hf_token, 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Creating training blocks...")
    train_blocks = create_blocks(sequences, tokenizer, block_size=1024)
    train_dataset = CustomDataset(train_blocks)

    test_text_data = load_and_process_datasets('openai/openai_humaneval', split='test')
    test_blocks = create_blocks(test_text_data, tokenizer, block_size=1024)
    test_dataset = CustomDataset(test_blocks)


    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,  # Changed from 2 to 8
        per_device_eval_batch_size=16,  # Added based on config
        gradient_accumulation_steps=16,  # Changed from 8 to 16
        warmup_ratio=0.05,  # Changed from 0.1 to 0.05
        max_grad_norm=1.0,
        num_train_epochs=1,
        learning_rate=8e-6,  # Changed from 2e-6 to 8e-6
        optim="adamw_torch",  # Changed from adamw_8bit to adamw_torch
        lr_scheduler_type="constant_with_warmup",  # Changed from cosine
        #dataloader_drop_last=True,  # Added based on config
        #dataloader_num_workers=4,  # Added based on config
        #dataloader_prefetch_factor=4,  # Added based on config
        logging_steps=1,  # Added based on config
        save_strategy="no",  # Added based on config
        save_total_limit=0,  # Added based on config
        #report_to="wandb",  # Changed from none to wandb
        seed=0,  # Using first seed from config
        torch_compile=True,  # Added based on config
        remove_unused_columns=True,  # Added based on config
        #evaluation_strategy="steps",  # Added based on config
        #eval_steps=5,  # Added based on config
        eval_on_start=True,  # Added based on config
    )

    print("Starting training...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        eval_dataset=test_dataset,
    )
    
    trainer.train()
    
    # Save the model
    finetuned_model_path = os.path.join(output_dir, "finetuned_model")
    model.save_pretrained(finetuned_model_path)
    tokenizer.save_pretrained(finetuned_model_path)
    
    return finetuned_model_path

def evaluate_model(model_path, output_file):
    """Evaluate the model using human-eval."""
    print("Loading problems...")
    problems = read_problems()

    print("Initializing VLLM...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        download_dir=os.path.dirname(model_path)
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        max_tokens=1024,
        n=30
    )

    # Prepare prompts
    task_ids = list(problems.keys())
    prompts = [problems[task_id]["prompt"] for task_id in task_ids]

    # Process in batches
    batch_size = 32
    batched_prompts = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    batched_task_ids = [task_ids[i:i + batch_size] for i in range(0, len(task_ids), batch_size)]

    samples = []
    print(f"Processing {len(prompts)} prompts in {len(batched_prompts)} batches...")

    for batch_prompts, batch_task_ids in tqdm.tqdm(zip(batched_prompts, batched_task_ids)):
        batch_outputs = llm.generate(batch_prompts, sampling_params)
        
        for task_id, output in zip(batch_task_ids, batch_outputs):
            for completion in output.outputs:
                samples.append({
                    "task_id": task_id,
                    "completion": completion.text
                })

    # Write results
    write_jsonl(output_file, samples)
    
    print(f"\nEvaluation complete:")
    print(f"- Number of unique tasks: {len(problems)}")
    print(f"- Total samples generated: {len(samples)}")
    print(f"- Samples per task: {len(samples)/len(problems)}")
    print(f"Results saved to '{output_file}'")

def main():
    # Configuration
    model_id = "google/gemma-2-2b"
    hf_token = "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"
    cache_dir = "/lfs/skampere1/0/eobbad/model_cache"
    output_dir = "/lfs/skampere1/0/eobbad/model_cache/"
    
    # Read and process sequences
    print("Loading training sequences...")
    sequences = []
    selected_sequences = []
    total_tokens = 0
    target_token_count = 1000000

    with open("./top_k_sequences_lz4.jsonl", 'r') as f:
        for line in f:
            entry = json.loads(line)
            sequences.append(entry['text'])

    # Initialize tokenizer for sequence selection
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=hf_token, 
        trust_remote_code=True
    )

    # Select sequences up to target token count
    print("Selecting sequences...")
    for seq in sequences:
        seq_tokens = len(tokenizer(seq)['input_ids'])
        
        if total_tokens + seq_tokens <= target_token_count:
            selected_sequences.append(seq)
            total_tokens += seq_tokens
        else:
            if total_tokens < target_token_count:
                remaining_tokens = target_token_count - total_tokens
                truncated_seq = tokenizer.decode(
                    tokenizer(seq)['input_ids'][:remaining_tokens]
                )
                selected_sequences.append(truncated_seq)
                total_tokens += remaining_tokens
            break

    print(f"Selected {len(selected_sequences)} sequences with {total_tokens} tokens")

    # Fine-tune the model
    print("Starting fine-tuning process...")
    finetuned_model_path = fine_tune_model(
        selected_sequences,
        model_id,
        hf_token,
        output_dir,
        cache_dir
    )

    # Evaluate the fine-tuned model
    print("Starting evaluation process...")
    evaluate_model(
        finetuned_model_path,
        "gemma_solutions.jsonl"
    )

if __name__ == "__main__":
    main()