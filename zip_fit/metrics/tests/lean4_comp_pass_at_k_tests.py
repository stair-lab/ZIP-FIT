import os
from pantograph import Server
from pantograph.data import CompilationUnit
from typing import List

from lean4_utils import parse_lean_completion, get_list_lean4_syntax_errors

def test_lean4_syntax_errors():
    server: Server = Server(imports=["Mathlib", "Init"], 
                            project_path=os.path.expanduser("~/mathlib_4_15_0_lfs"), 
                            timeout=None)

    # -- Test 1: Parse Lean Completion
    print('--> Running Test 1: Parse Lean Completion')
    llm_output: str = "##theorem math_stmt (x : R) : 1 + x = x + 1 := by ##"
    lean_completion: str = parse_lean_completion(llm_output)
    print(f'{lean_completion=}')
    assert lean_completion == "theorem math_stmt (x : R) : 1 + x = x + 1 := by", f'Lean completion: {lean_completion} should be "theorem math_stmt (x : R) : 1 + x = x + 1 := by"'
    print('--> Passed Test 1: Parse Lean Completion\n')

    # -- Test 2: Get List of No Syntax Errors
    print('--> Running Test 2: Get List of No Syntax Errors')
    lean_snippet: str = "#check TopologicalSpace\n#eval \"Hello World!\""
    syntax_errors: List[str] = get_list_lean4_syntax_errors(lean_snippet, server)
    print(f'{syntax_errors=}')
    assert len(syntax_errors) == 0, f'Lean snippet: {lean_snippet} should not have any syntax errors, but has the following errors: {syntax_errors}'
    print('--> Passed Test 2: Get List of Syntax Errors\n')

    # -- Test 3: Sorry Should not be an Syntax Error
    print('--> Running Test 3: Sorry Should not be an Syntax Error')
    lean_snippet: str = ("open Complex Filter Function Metric Finset\n"
                          "open scoped BigOperators Topology\n"
                          "theorem exercise_1_13b {f : ℂ → ℂ} (Ω : Set ℂ) (a b : Ω) (h : IsOpen Ω) "
                          "(hf : DifferentiableOn ℂ f Ω) (hc : ∃ (c : ℝ), ∀ z ∈ Ω, (f z).im = c) : "   
                          "f a = f b := sorry")
    syntax_errors: List[str] = get_list_lean4_syntax_errors(lean_snippet, server)
    print(f'{syntax_errors=}')
    assert len(syntax_errors) == 0, f'Lean snippet: {lean_snippet} should not have any syntax errors, but has the following errors: {syntax_errors}'
    print('--> Passed Test 3: Sorry Should not be an Syntax Error\n')

    # -- Test 4: Deliberate syntax errors
    print('--> Running Test 4: Deliberate syntax error')
    # one error
    lean_snippet: str = "asdfasdf\n"
    syntax_errors: List[str] = get_list_lean4_syntax_errors(lean_snippet, server)
    print(f'{syntax_errors=}')
    assert len(syntax_errors) == 1, f'Lean snippet: {lean_snippet} should have 1 syntax error, but has {len(syntax_errors)}'
    # two errors
    lean_snippet: str = "asdfadsf\najhkhasdfhjkl\n"
    syntax_errors: List[str] = get_list_lean4_syntax_errors(lean_snippet, server)
    print(f'{syntax_errors=}')
    assert len(syntax_errors) == 2, f'Lean snippet: {lean_snippet} should have 2 syntax errors, but has {len(syntax_errors)}'
    # three errors
    lean_snippet: str = "asdfadsf\najhkhasdfhjkl\nkjasdfljkh\n"
    syntax_errors: List[str] = get_list_lean4_syntax_errors(lean_snippet, server)
    print(f'{syntax_errors=}')
    assert len(syntax_errors) == 3, f'Lean snippet: {lean_snippet} should have 3 syntax errors, but has {len(syntax_errors)}'
    print('--> Passed Test 4: Deliberate syntax error\n')

if __name__ == "__main__":
    test_lean4_syntax_errors()