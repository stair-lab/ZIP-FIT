#!/usr/bin/env python3

from huggingface_hub import hf_hub_download
import os

def main():
    repo_id = "zipfit/math-select-06062025"
    filename = "src.jsonl"
    
    print(f"Checking for {filename} in {repo_id}...")
    
    try:
        # Try to download the file
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
        
        # Check file size
        size_mb = os.path.getsize(file_path) / 1024 / 1024
        print(f"✅ Successfully downloaded {filename} from {repo_id}")
        print(f"File size: {size_mb:.2f} MB")
        
        # Check first lines to confirm it's actually jsonl
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('{') and first_line.endswith('}'):
                print(f"✅ The file is in valid jsonlines format")
            else:
                print(f"⚠️ The file may not be in jsonlines format")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 