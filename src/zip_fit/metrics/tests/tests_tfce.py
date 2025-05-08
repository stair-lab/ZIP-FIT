def main():
    """
    Like the TFA main: 
      1) seed
      2) load a small portion of proofnet
      3) measure TFCE on some model(s)
      4) prints average results
    """
    start_time = time.time()
    seed_everything(42)

    # 1) load 5 examples from proofnet
    ds = load_dataset("hoskinson-center/proofnet", split="validation")
    def my_prompt_format(prompt: str) -> str:
        return (
            "Translate the natural language version of the mathematical statement "
            f"to a formal Lean version:\n{prompt}\n"
        )
    ds = ds.map(lambda ex: {'prompt': my_prompt_format(ex['nl_statement']), 'gold_response': ex['formal_statement']}, num_proc=24)

    N = 5
    sub_ds = ds.select(range(min(N, len(ds))))

    # 2) some models to test
    model_token_configs = [
        {
            "name": "google/gemma-2-2b",
            "repo": "google/gemma-2-2b",
        },
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3) measure TFCE
    for cfg in model_token_configs:
        name = cfg["name"]
        repo = cfg["repo"]
        print(f"\nEvaluating TFCE for {name} on {N} examples...")

        st = time.time()
        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).to(device)

        avg_loss = compute_tfce_for_subds(
            sub_ds=sub_ds,
            model=model,
            repo=repo,
            device=device,
            reduction="mean"
        )
        ed = time.time()

        print(f" => Average TFCE for {name}: {avg_loss:.4f}")
        print(f" => Time: {(ed - st):.2f} s")

    print(f"\nAll done. total time = {time.time() - start_time:.2f} s")


def minimal_tfce_trainer_test():
    """
    A minimal script that demonstrates using the TfceCallback with 
    the Hugging Face Trainer for a tiny "toy" dataset. 
    It runs for 1 training step and triggers the TfceCallback logic 
    at training begin, evaluation, and train end.
    """
    from transformers import TrainingArguments, Trainer

    seed_everything(42)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # choose GPU
    kwargs = {}
    today = datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss') # eg '2024_m01_d22_t13h_00m_30s'
    tmux_sess_num: str = get_current_tmux_session_number()
    kwargs = kwargs | {'today': today, 'tmux_sess_num': tmux_sess_num, 'hostname': gethostname()}
    run_name = f'{kwargs}' 
    run_name = f'tfce test test proofnet gpt2' + f'{kwargs}' 
    # run_name = 'tfce validaton test proofnet gpt2' + f'{kwargs}' 
    run = wandb.init(mode=kwargs.get('mode', 'online'), project="huggingface", name=run_name, save_code=True, config=kwargs)
    wandb.save(__file__) # save current code now, don't wait to wandb.finish, also useful: wandb.save("*.py") # upload all .py files in current directory
    config = kwargs

    # 1) Load a small model (like GPT-2)
    model_name = "gpt2"
    # model_name = "google/gemma-2-2b"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    # 2) Prepare a dataset for training
    def my_prompt_format(prompt: str) -> str:
        return (
            "Translate the natural language version of the mathematical statement "
            f"to a formal Lean version:\n{prompt}\n"
        )

    ds_train = load_dataset("hoskinson-center/proofnet", split="test")
    ds_train = load_dataset("hoskinson-center/proofnet", split="validation")
    ds_train = ds_train.map(lambda eg: {
        "text": my_prompt_format(eg["nl_statement"])
                 + eg["formal_statement"]
                 + tokenizer.eos_token
    }, num_proc=24)

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], 
            padding="max_length", 
            max_length=256, 
            truncation=True
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    ds_train = ds_train.map(
        tokenize_function,
        batched=True,
        remove_columns=ds_train.column_names,
        num_proc=24
    )

    # 3) Prepare a dataset for TFCE
    ds_eval = load_dataset("hoskinson-center/proofnet", split="test")
    ds_eval = ds_eval.map(lambda ex: {
        "prompt": my_prompt_format(ex["nl_statement"]),
        "gold_response": ex["formal_statement"]
    }, num_proc=24)

    # 3.1) Subsmaple for tfce sanity checking
    n_subsample = 25
    indices = random.sample(range(len(ds_train)), k=n_subsample)
    ds_train = ds_train.select(indices)
    ds_eval = ds_eval.select(indices)

    # 4) minimal training args
    training_args = TrainingArguments(
        output_dir="./test-tfce-output",
        do_train=True,
        do_eval=True,
        # evaluation_strategy="steps",
        # eval_steps=1,
        evaluation_strategy="epoch",
        num_train_epochs=15,
        logging_steps=1,
        per_device_train_batch_size=2,
        remove_unused_columns=False,  # ensure prompt/gold_response remain accessible
        save_strategy="no",
        # gradient_accumulation_steps=1,
        # gradient_checkpointing=config.get('gradient_checkpointing', True), # careful might give issues, but not in skampere1
        learning_rate=config.get('learning_rate', 1e-5),
        # optim=config.get('optim', 'paged_adamw_32bit'),
    )

    # 5) attach TfceCallback
    callback = TfceCallback(
        tfce_dataset=ds_eval,
        repo=model_name,
        n_begin=186,   # X samples at train begin
        n_during=186,  # Y sample at each on_evaluate
        n_end=186,     # Z samples at train end
        device="cuda",
        reduction="mean"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_train,
        callbacks=[callback]
    )

    # 6) train
    trainer.train()

    # 7) End Wandb Run
    run.alert(title="Run Completed", text=f"Run finished, run url: {run.get_url()}")
    print(f'{run.get_url()=}')
    wandb.finish()


if __name__ == "__main__":
    # main()  # or test the direct usage
    minimal_tfce_trainer_test()