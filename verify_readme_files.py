#!/usr/bin/env python3

from huggingface_hub import hf_hub_download

def check_readme(repo_id):
    print(f"\n{repo_id} README.md:")
    print("=" * 60)
    
    # Force a fresh download to ensure we get the latest version
    readme_path = hf_hub_download(
        repo_id=repo_id,
        filename="README.md",
        repo_type="dataset",
        force_download=True
    )
    
    # Read and print the content
    with open(readme_path, 'r') as f:
        content = f.read()
        print(content)
    
    return content

# Check both repositories
print("CHECKING UPDATED README FILES...\n")

# Check proofnet README
proofnet_readme = check_readme("brando/proofnet-v3-lean4")
if "ProofNet Lean4 v3" in proofnet_readme:
    print("\n✅ proofnet-v3-lean4 README contains original content")
else:
    print("\n⚠️ proofnet-v3-lean4 README is missing original content")

# Check minif2f README
minif2f_readme = check_readme("brando/minif2f-lean4")
if "MiniF2F Lean4" in minif2f_readme:
    print("\n✅ minif2f-lean4 README contains original content")
else:
    print("\n⚠️ minif2f-lean4 README is missing original content") 