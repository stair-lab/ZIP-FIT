#!/usr/bin/env python3

import os
from huggingface_hub import hf_hub_download, HfApi, login

def main():
    # Paths and configuration
    token_path = "/home/brandomiranda/keys/master_hf_token.txt"
    
    # Source and target repositories
    repositories = [
        {"source": "UDACA/proofnet-v3-lean4", "target": "brando/proofnet-v3-lean4"},
        {"source": "UDACA/minif2f-lean4", "target": "brando/minif2f-lean4"}
    ]
    
    # Read the token
    with open(token_path, 'r') as f:
        token = f.read().strip()
    
    # Login to Hugging Face
    login(token=token)
    api = HfApi(token=token)
    
    for repo_info in repositories:
        source_repo = repo_info["source"]
        target_repo = repo_info["target"]
        
        print(f"\nProcessing README for: {source_repo} -> {target_repo}")
        
        # Step 1: Download README from source
        try:
            source_readme_path = hf_hub_download(
                repo_id=source_repo,
                filename="README.md",
                repo_type="dataset"
            )
            
            with open(source_readme_path, 'r') as f:
                source_content = f.read()
                
            print(f"Successfully downloaded README from {source_repo}")
            
            # Step 2: Download README from target (for YAML metadata)
            try:
                target_readme_path = hf_hub_download(
                    repo_id=target_repo,
                    filename="README.md",
                    repo_type="dataset"
                )
                
                with open(target_readme_path, 'r') as f:
                    target_content = f.read()
                    
                # Extract YAML metadata from target if it exists
                if '---' in target_content:
                    yaml_parts = target_content.split('---', 2)
                    if len(yaml_parts) >= 3:
                        yaml_metadata = f"---{yaml_parts[1]}---\n\n"
                    else:
                        yaml_metadata = ""
                else:
                    yaml_metadata = ""
                    
                print(f"Successfully extracted metadata from {target_repo}")
            except Exception as e:
                print(f"Error downloading target README, proceeding without metadata: {e}")
                yaml_metadata = ""
            
            # Step 3: Create temporary README with combined content
            temp_readme_path = "/tmp/combined_readme.md"
            with open(temp_readme_path, 'w') as f:
                f.write(yaml_metadata + source_content)
                
            print(f"Created combined README file")
            
            # Step 4: Upload the combined README to target repo
            api.upload_file(
                path_or_fileobj=temp_readme_path,
                path_in_repo="README.md",
                repo_id=target_repo,
                repo_type="dataset"
            )
            
            print(f"✅ Successfully updated README in {target_repo}")
            
        except Exception as e:
            print(f"Error processing README for {source_repo}: {e}")
    
    print("\nScript completed.")

if __name__ == "__main__":
    main() 