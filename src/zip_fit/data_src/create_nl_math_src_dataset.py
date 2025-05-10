# Target set 
# https://huggingface.co/datasets/Putnam-AXIOM/putnam-axiom-dataset-v1/viewer/default/variations_org
# target_settings = [
#     ('openai/gsm8k', 'test', 1319, "GSM8K Test"),
#     ('~/beyond-scale-2-alignment-coeff/data/MATH/test/**/*.json', 'test', 5000, "MATH Test"),
#     ('brando/olympiad-bench-imo-math-boxed-825-v2-21-08-2024', 'test', 125, "IMO Test"),
#     ('brando/putnam-axiom-dataset-text-only', 'func_variations_265_10_30_2024', 265, "Putnam Test (Vars)")
# ]

# Target Eval (Test) settings and datasets
target_eval_setting = [
    # choosing vars as test since it's unlikely models were trained on it
    ('zipfit/Putnam-AXIOM-for-zip-fit-splits', 'test', 222), 
]

# Target (Validation/dev) settings and datasets
target_validation_settings = [
    # choosing vars as test since it's unlikely models were trained on it
    ('zipfit/Putnam-AXIOM-for-zip-fit-splits', 'validation', 150), 
]

# Source datasets from which to do data selection
src_datasets = [
    # Goal: around ~100k total examples or documents

    # ~1-25% really good data
    ('zipfit/Putnam-AXIOM-for-zip-fit-splits', 'train', 150), 
    ('hoskinson-center/proofnet', 'validation', 185),
    ('brando/olympiad-bench-imo-math-boxed-825-v2-21-08-2024', 'train', 700),

    # ~25-50% somewhat related
    ('brando/hendrycks_math', 'train', 5_500),
    ('TIGER-Lab/MathInstruct', 'train', 3_000), # Mammoth = "COT & POP"
    ('TIGER-Lab/WebInstructSub', 'train', 3_000), # Mammoth 2 = "Problem:...Soln:...pairs from CC made into pairs by LLMs"
    ('openai/gsm8k', 'train', 7_473),
    ('brando/small-open-web-math-dataset-v2', 'train', 9_000),

    # ~50-75% unrelated or really bad data
    ('brando/small-c4-dataset', 'train', 10_000), 
    ('brando/small-c4-dataset', 'validation', 10_000), 
    ('brando/small-c4-dataset', 'test', 10_000), 
    ('iamtarun/python_code_instructions_18k_alpaca', 'train', 18_612),
    ('brando/random-all-ascii-dataset', 'train', 5_000)
]