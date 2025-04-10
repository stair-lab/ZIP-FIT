"""
Example script: pantograph_import.py

Demonstrates how to:
  - Initialize a Pantograph Server with Mathlib.
  - Prepend custom headers/imports to a theorem snippet.
  - Detect Lean "syntax/parse" errors by searching for "error:" lines
    while ignoring "unsolved goals."
  - Validate that all statements in the "proofnet-v2-lean4" dataset are
    syntax-error-free on both validation and test splits.
"""

import os
from pantograph.server import Server
from typing import List, Tuple
import re


##############################################################################
# 1) The minimal helper that identifies "error:" lines, ignoring unsolved goals
##############################################################################
def check_lean_compiles_syntax_only(server: Server, lean_snippet: str, debug: bool = False) -> Tuple[int, List]:
    r"""
    Check a Lean 4 code snippet for *parse/syntax errors* (ignore "unsolved goals").

    Implementation:
      - We call `server.load_sorry(lean_snippet)`, which compiles the snippet.
      - For each message in the returned compilation units:
        * If the line contains "error:" (case-insensitive is optional),
          we check if it also has "unsolved goals" — if so, skip it, because
          that's not a parse/lexical error.
        * Otherwise, count it as a syntax error.

    Returns the count of syntax/parse errors found.

    Example
    -------
    >>> server = Server(imports=["Mathlib"], project_path="~/mathlib4")
    >>> code = "theorem two_eq_two : 2 = 2 := by"  # incomplete
    >>> num_errs = check_lean_compiles_syntax_only(server, code)
    >>> print(num_errs)  # 1 or more
    """
    try:
        compilation_units = server.load_sorry(lean_snippet)
    except:
        print(f'\n----{lean_snippet=}----\n') if debug else None
        import traceback
        traceback.print_exc() if debug else None
        return 1, ['PyPantograph threw someexception.']

    syntax_error_count = 0
    syntax_compilation_err_units: list = []
    for comp_unit in compilation_units:
        for msg in comp_unit.messages:
            # Quick check: if 'error:' is in the message, but not "unsolved goals"
            # => it's likely a parse/lexical error.
            # (In practice, we often see strings like "<anonymous>:1:5: error: ...")
            if "error:" in msg and ("unsolved goals" not in msg.lower()):
                syntax_error_count += 1
        if syntax_error_count > 0 and debug:
            syntax_compilation_err_units.append(comp_unit)

    return syntax_error_count, syntax_compilation_err_units


##############################################################################
# 2) Minimal usage example with a single snippet
##############################################################################
def main_single_test():
    """
    Quick demonstration on a single snippet. 
    We'll only treat "error:" lines that do NOT mention 'unsolved goals' as parse errors.
    """
    print("=== Single Snippet Test ===")

    # (A) Initialize Pantograph for a Lean 4 project that includes Mathlib:
    project_path = os.path.expanduser("~/mathlib4")  # Adjust as needed
    server = Server(
        imports=["Mathlib"],
        project_path=project_path
    )

    # (B) The snippet (with a custom open + a theorem using `:= sorry`)
    lean_snippet = (
        "open Complex Filter Function Metric Finset open scoped BigOperators Topology\n\n"
        "theorem exercise_1_13b {f : ℂ → ℂ} (Ω : Set ℂ) (a b : Ω) (h : IsOpen Ω) "
        "(hf : DifferentiableOn ℂ f Ω) (hc : ∃ (c : ℝ), ∀ z ∈ Ω, (f z).im = c) "
        ": f a = f b := sorry"
    )

    syntax_errs = check_lean_compiles_syntax_only(server, lean_snippet)
    print(f"\nSnippet:\n{lean_snippet}\n")

    if syntax_errs == 0:
        print("[OK] No syntax/parse errors found (though it might have unsolved goals).")
    else:
        print(f"[ERROR] Found {syntax_errs} parse/lexical syntax errors.")


##############################################################################
# 3) Checking both validation and test splits from "proofnet-v2-lean4"
##############################################################################
def main_hf_proofnet():
    """
    Illustrates checking the proofnet-v2-lean4 dataset on both validation
    and test splits for syntax correctness. If the dataset is correct, we
    expect zero syntax errors.
    """
    print("=== Checking proofnet-v2-lean4 (val + test) ===")
    from datasets import load_dataset

    # 1) Initialize server, pointing at your local Lean + Mathlib project
    project_path = os.path.expanduser("~/mathlib4")
    server = Server(
        imports=["Mathlib"],
        project_path=project_path
    )

    # 2) Helper to run a syntax check loop
    def check_split(split_name: str):
        # ds = load_dataset("UDACA/proofnet-v2-lean4", split=split_name)
        ds = load_dataset("UDACA/proofnet-v3-lean4", split=split_name)
        total_snippets = 0
        num_parse_ok = 0

        for row in ds:
            header = row["header"]
            # Optionally remove "import Mathlib" if it appears
            header = header.replace("import Mathlib", "")

            # Combine the snippet:
            snippet = f"{header}\n\n{row['formal_statement']}"
            syntax_err_count, syntax_compilation_err_units = check_lean_compiles_syntax_only(server, snippet, debug=True)

            total_snippets += 1
            if syntax_err_count == 0:
                num_parse_ok += 1
            else:
                print(f'Current number of syntax errors for current sinppet: {syntax_err_count=}')
                print(f'----\n{snippet}\n----')
                print(f'{syntax_compilation_err_units=}')

        print(f"Split={split_name}: {num_parse_ok}/{total_snippets} have zero syntax errors.")
        return num_parse_ok, total_snippets

    # 3) Run for validation
    val_ok, val_total = check_split("validation")

    # 4) Run for test
    test_ok, test_total = check_split("test")

    # Summarize
    print("=== Summary ===")
    print(f"Validation: {val_ok}/{val_total} OK")
    print(f"Test:       {test_ok}/{test_total} OK")


##############################################################################
# 4) Entry point
##############################################################################
if __name__ == "__main__":
    # Example single snippet test
    # main_single_test()

    # (Optional) Check the entire proofnet-v2-lean4 validation & test
    main_hf_proofnet()
