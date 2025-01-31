import json

# Input and output file paths
input_file = "/lfs/skampere1/0/brando9/ZIP-FIT/data/proofnet.jsonl"
output_file = "/lfs/skampere1/0/brando9/ZIP-FIT/data/proofnet_lean4_v2.jsonl"

def clean_informal_prefix(text):
    """Removes '/--' and '-/' from the given text."""
    return text.replace('/--', '').replace('-/', '').strip()

def process_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            if "informal_prefix" in data:
                data["nl_statement"] = clean_informal_prefix(data["informal_prefix"])
            outfile.write(json.dumps(data) + '\n')

# Run the processing function
process_jsonl(input_file, output_file)

print(f"Processed dataset saved to {output_file}")
