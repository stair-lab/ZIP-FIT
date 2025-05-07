#!/usr/bin/env python3

import os
import json
import random
from datasets import load_dataset
from collections import defaultdict
from huggingface_hub import login, HfApi

def check_required_format(text):
    """Check if text follows 'Problem:...Solution:...' format."""
    return (text.strip().startswith("Problem:") and 
            "Solution:" in text and 
            text.find("Problem:") < text.find("Solution:"))

def main():
    # Config
    dataset_repo = "zipfit/math-select-06062025"
    output_jsonl = "/tmp/math_select_src.jsonl"
    output_hf_repo = "zipfit/math-select-06062025-jsonl"
    token_path = "/home/brandomiranda/keys/master_hf_token.txt"
    
    print(f"Loading dataset from {dataset_repo}...")
    dataset = load_dataset(dataset_repo)

    # Group examples by source
    sources = defaultdict(list)
    for i, item in enumerate(dataset['src']):
        sources[item['original_source']].append(i)
    
    # Define which datasets should have Problem/Solution format
    problem_solution_format_datasets = [
        'zipfit/Putnam-AXIOM-for-zip-fit-splits',
        'hoskinson-center/proofnet',
        'brando/olympiad-bench-imo-math-boxed-825-v2-21-08-2024',
        'brando/hendrycks_math',
        'TIGER-Lab/MathInstruct',
        'TIGER-Lab/WebInstructSub',
        'openai/gsm8k'
    ]
    
    # Check samples from each source
    print("\nVerifying data format for each source...")
    format_issues = {}
    
    for source, indices in sources.items():
        # Extract dataset name without split and count
        dataset_name = source.split(',')[0]
        
        # Sample a few examples to check (up to 5)
        sample_indices = random.sample(indices, min(5, len(indices)))
        
        # Check if this source should have Problem/Solution format
        needs_format = False
        for expected_format_ds in problem_solution_format_datasets:
            if dataset_name == expected_format_ds:
                needs_format = True
                break
        
        # Check samples
        issues = 0
        for idx in sample_indices:
            text = dataset['src'][idx]['text']
            has_format = check_required_format(text)
            
            if needs_format and not has_format:
                issues += 1
                if issues == 1:  # Only print the first issue
                    print(f"\n❌ Format issue in {source}:")
                    print(f"Sample text (first 200 chars): {text[:200]}...")
        
        if issues > 0:
            format_issues[source] = f"{issues}/{len(sample_indices)} samples have format issues"
        else:
            print(f"✅ {source}: Format looks good")
    
    # Report format issues summary
    if format_issues:
        print("\nSummary of format issues:")
        for source, issue in format_issues.items():
            print(f"❌ {source}: {issue}")
    else:
        print("\n✅ All formats look good!")
    
    # Export to jsonlines
    print(f"\nExporting dataset to {output_jsonl}...")
    with open(output_jsonl, 'w') as f:
        for item in dataset['src']:
            f.write(json.dumps(item) + '\n')
    
    print(f"Exported {len(dataset['src'])} examples to {output_jsonl}")
    
    # Upload to HF
    print(f"\nUploading jsonlines file to {output_hf_repo}...")
    
    # Read token
    with open(token_path, 'r') as f:
        token = f.read().strip()
    
    # Login to HF
    login(token=token)
    api = HfApi(token=token)
    
    # Create a README file
    readme_path = "/tmp/README.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# {output_hf_repo}

This dataset contains {len(dataset['src'])} examples from various math and general sources, formatted as jsonlines for data selection. Each example contains:
- `text`: The content, with math examples following the "Problem:\\n...\\n\\nSolution:\\n..." format
- `original_source`: Source information for the example

## Sources:
""")
        # Add source counts
        for source, indices in sorted(sources.items()):
            f.write(f"- {source}: {len(indices)} examples\n")
    
    # Upload files using HfApi
    try:
        # Create the repo if it doesn't exist
        api.create_repo(repo_id=output_hf_repo, repo_type="dataset", exist_ok=True)
        
        # Upload the files
        print("Uploading README.md...")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=output_hf_repo,
            repo_type="dataset"
        )
        
        print(f"Uploading {output_jsonl} as src.jsonl...")
        api.upload_file(
            path_or_fileobj=output_jsonl,
            path_in_repo="src.jsonl",
            repo_id=output_hf_repo,
            repo_type="dataset"
        )
        
        print(f"✅ Successfully uploaded files to {output_hf_repo}")
    except Exception as e:
        print(f"❌ Error uploading to HF: {e}")
    
    print("Done!")

if __name__ == "__main__":
    main() 