#!/usr/bin/env python3
import os
from typing import List
from pantograph.server import Server
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi, CommitOperationAdd

################################################################################
# 1) Helper: Check snippet for syntax errors using Pantograph
################################################################################
def check_lean_compiles_syntax_only(server: Server, lean_snippet: str, debug: bool = False) -> bool:
    """
    Returns True if 'lean_snippet' has NO syntax/parse errors (ignoring "unsolved goals"),
    otherwise returns False.
    """
    try:
        compilation_units = server.load_sorry(lean_snippet)
    except Exception as e:
        if debug:
            print(f"Pantograph load error for snippet:\n{lean_snippet}\nException={e}\n")
        return False

    # We only treat lines with 'error:' that do NOT include "unsolved goals".
    for comp_unit in compilation_units:
        for msg in comp_unit.messages:
            if "error:" in msg and ("unsolved goals" not in msg.lower()):
                return False  # Found a genuine syntax error
    return True


################################################################################
# 2) Main script: create v3 from v2, adding "header_no_import" + filtering errors
################################################################################
def main():
    # (A) Initialize Pantograph
    project_path = os.path.expanduser("~/mathlib4")  # <-- adjust if needed
    server = Server(
        imports=["Mathlib"],
        project_path=project_path
    )

    # (B) Load the v2 dataset from HF
    print("Loading UDACA/proofnet-v2-lean4 from Hugging Face Hub...")
    ds_v2 = load_dataset("UDACA/proofnet-v2-lean4")

    # (C) Add a new "header_no_import" field by removing "import Mathlib"
    #     We'll do this for each split via `.map()`.
    def remove_import_mathlib(example):
        # Make a new field with the import removed
        cleaned_header = example["header"].replace("import Mathlib", "").strip()
        example["header_no_import"] = cleaned_header
        return example

    ds_v2 = ds_v2.map(remove_import_mathlib)

    # (D) Filter out rows that produce syntax errors.
    #     Use 'header_no_import' + 'formal_statement' combined.
    def syntax_filter(example):
        snippet = f"{example['header_no_import']}\n\n{example['formal_statement']}"
        return check_lean_compiles_syntax_only(server, snippet, debug=False)

    ds_val_clean = ds_v2["validation"].filter(syntax_filter)
    ds_test_clean = ds_v2["test"].filter(syntax_filter)

    # Recreate a new dataset dictionary for v3
    ds_v3 = DatasetDict({
        "validation": ds_val_clean,
        "test": ds_test_clean
    })

    # (E) Report final counts
    print("=== Summary of cleaned v3 dataset ===")
    print(f"Validation: {ds_val_clean.num_rows} rows")
    print(f"Test:       {ds_test_clean.num_rows} rows")

    # (F) Push to new HF repo: "proofnet-v3-lean4"
    repo_id_v3 = "UDACA/proofnet-v3-lean4"  # Adjust if needed
    print(f"Pushing dataset to {repo_id_v3}...")
    ds_v3.push_to_hub(
        repo_id=repo_id_v3,
        private=False,  # or True if you prefer
        token=None      # or "hf_your_token"
    )

    # (G) (Optional) Add minimal README
    readme_text = """# ProofNet Lean4 v3

This dataset is based on `proofnet-v2-lean4` but **removes** any entries
that caused Lean 4 syntax/parse errors. We also introduce a new field
**`header_no_import`** that removes `"import Mathlib"`.

**Splits**: `validation` and `test`.

Enjoy!
"""
    api = HfApi()
    operations = [
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=readme_text.encode("utf-8")
        )
    ]
    api.create_commit(
        repo_id=repo_id_v3,
        operations=operations,
        commit_message="Add README for proofnet-v3-lean4 (with header_no_import).",
        repo_type="dataset"
    )
    print(f"README added at: https://huggingface.co/datasets/{repo_id_v3}/blob/main/README.md")

    print("All done. The new v3 dataset is at:")
    print(f"https://huggingface.co/datasets/{repo_id_v3}")


if __name__ == "__main__":
    main()
