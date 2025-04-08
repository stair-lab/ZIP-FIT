from datasets import load_dataset
from zip_fit.asdf import ZIPFIT

# Define the paths and target dataset
source_dataset = "/path/to/source_data.jsonl"
target_dataset = 'openai/openai_humaneval'

# Define the function to load the target dataset
def target_load_dataset_fn(dataset):
    ds = load_dataset(dataset, split='test', trust_remote_code=True)
    return ds

# Define the function to parse examples from the target dataset
def target_parse_example_fn(ex):
    text = f"Problem description: {ex['prompt']} \nCanonical solution: {ex['canonical_solution']}"
    return text

# Create an instance of ZIPFIT
zip_fit_instance = ZIPFIT(
    source_dataset=source_dataset,
    target_dataset=target_dataset,
    target_load_fn=target_load_dataset_fn,
    target_parse_fn=target_parse_example_fn,
    k=100,
    output_file="top_k_sequences.jsonl",
    compression_algorithm='gzip'  # Change to 'lz4' if desired
)

# Run the ZIPFIT process
zip_fit_instance.run()

