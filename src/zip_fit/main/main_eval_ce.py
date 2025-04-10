###############################################################################
# 4. Main script: Evaluate cross-entropy via HF Trainer
###############################################################################

def main() -> None:
    """
    Main driver:
     1) Load 'AI4M/gemma2-2b-gpt4-more-5-epochs'
     2) Load 'UDACA/proofnet-v3-lean4', using 'validation' split
     3) Rename 'nl_statement' -> 'text'
     4) Tokenize + group into blocks (like training)
     5) Flatten so each block is a separate example
     6) Create a Trainer with do_eval=True, do_train=False
     7) Call trainer.evaluate() and print the average cross-entropy (eval_loss)
    """
    nltk.download("punkt", quiet=True)
    seed_everything(42)

    MODEL_NAME = "AI4M/gemma2-2b-gpt4-more-5-epochs"
    DATASET_NAME = "UDACA/proofnet-v3-lean4"
    SPLIT = "validation"
    BLOCK_SIZE = 1024
    BATCH_SIZE = 1  # Adjust per GPU memory

    # 1. Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    # 2. Load dataset split
    raw_dataset = load_dataset(DATASET_NAME, split=SPLIT)

    # 3. Rename 'nl_statement' -> 'text' for tokenization
    def rename_columns(example):
        return {"text": example["nl_statement"]}

    raw_dataset = raw_dataset.map(
        rename_columns,
        remove_columns=raw_dataset.column_names,
        batched=True,
    )

    # 4. Tokenize + group into blocks
    proc_dataset = raw_dataset.map(
        lambda batch: tokenize_and_group_texts_via_blocks(
            batch, tokenizer=tokenizer, block_size=BLOCK_SIZE
        ),
        batched=True,
    )

    # 5. Flatten blocks
    eval_dataset = BlocksAsExamplesDataset(proc_dataset)

    # 6. Define TrainingArguments
    #    We do NOT train, only evaluate. 
    #    Trainer will compute the average cross-entropy if 'labels' are present.
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir="./hf_trainer_eval",
        per_device_eval_batch_size=BATCH_SIZE,
        do_train=False,
        do_eval=True,
        evaluation_strategy="no"  # We'll just do a single .evaluate() call
    )

    # 7. Build Trainer. 
    #    We can use the default_data_collator because our dataset already 
    #    has 'input_ids'/'labels' shaped as [block_size].
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,  # Not strictly needed for evaluation, but let's keep for completeness
        data_collator=default_data_collator,
    )

    # 8. Evaluate
    metrics = trainer.evaluate()
    # metrics['eval_loss'] is the average cross-entropy in "natural log" units per token
    eval_loss = metrics["eval_loss"]
    ppl = math.exp(eval_loss)

    print("\n===========================================")
    print(f"Eval Cross-Entropy (nat) = {eval_loss:.4f}")
    print(f"Equivalent Perplexity   = {ppl:.4f}")
    print("===========================================\n")


if __name__ == "__main__":
    main()

