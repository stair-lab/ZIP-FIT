import os
from datasets import load_dataset, DatasetDict
from huggingface_hub import login

# Log in to Hugging Face Hub
key_file_path = "~/keys/master_hf_token.txt"
token = open(os.path.expanduser(key_file_path)).read().strip()
login(token=token)

# Define data files and subsets with simplified split names
data_files = {
    "full_original_236_10_30_2024": [
        os.path.expanduser('~/putnam-axiom/data/Putnam_AXIOM_Original_v2.json')
    ],
    "func_original_53_10_30_2024": [
        os.path.expanduser('~/putnam-axiom/data/Putnam-AXIOM_Variations/original.json')
    ],
    "func_variations_265_11_23_2024": [
        os.path.expanduser('~/putnam-axiom/data/Putnam-AXIOM_Variations/test.json')
    ]
}

# Define required fields (based on all fields from the provided samples)
required_columns = [
    "year", "id", "problem", "solution", "answer_type", "source", "type",
    "original_problem", "original_solution", "variation"
]
final_columns = [
    "year", "id", "problem", "solution", "answer_type", "source", "type",
    "original_problem", "original_solution"
]

# Function to harmonize columns
def load_and_harmonize(data_file, required_columns):
    # Load the dataset
    dataset = load_dataset("json", data_files=data_file, split="all")
    
    # Ensure all required columns are present
    for col in required_columns:
        if col not in dataset.column_names:
            dataset = dataset.add_column(col, [""] * len(dataset))  # Add empty strings for missing columns

    # Keep only the required columns to ensure order and consistency
    dataset = dataset.select_columns(final_columns)
    
    return dataset

# Load each subset, harmonize columns, and add to DatasetDict
dataset_dict = DatasetDict({
    subset_name: load_and_harmonize(files, required_columns)
    for subset_name, files in data_files.items()
})
print(f'{dataset_dict}')

# Push the harmonized DatasetDict to Hugging Face under a single dataset
# dataset_name = "brando/putnam-axiom-dataset"
dataset_name = "Putnam-AXIOM/putnam-axiom-dataset"
dataset_dict.push_to_hub(dataset_name, token=token)
print(f"Dataset uploaded to https://huggingface.co/datasets/{dataset_name}")
