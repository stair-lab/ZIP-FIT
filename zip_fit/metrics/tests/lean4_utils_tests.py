import os
from pantograph import Server
from pantograph.data import CompilationUnit
from typing import List

from lean4_utils import parse_lean_completion, get_list_lean4_syntax_errors

def test_basic_lean4_snippets():
    """Test basic Lean4 syntax and PyPantograph functionality."""
    server: Server = Server(imports=["Mathlib", "Init"], 
                          project_path=os.path.expanduser("~/mathlib_4_15_0_lfs"), 
                          timeout=None)

    # Test 1: Hello World! in Lean4 with PyPantograph
    print('--> Running Test 1: Hello World! in Lean4 with PyPantograph')
    lean_snippet: str = "#eval \"Hello World!\""
    result: List[CompilationUnit] = server.load_sorry(lean_snippet)
    print(f'{len(result)=}')
    for i, comp_unit in enumerate(result):
        print(f'Compilation Unit {i}: {comp_unit}')
        print(f'Messages for Compilation Unit {i}: {comp_unit.messages=}')
    assert 'error: ' not in str(comp_unit), f'Lean snippet: {lean_snippet} should not have any errors, but has the following error: {comp_unit.messages}'
    print('--> Passed Test 1: Hello World! in Lean4 with PyPantograph\n')

    # Test 2: Import Check Type of Topology
    print('--> Running Test 2: Import Check Type of Topology')
    lean_snippet: str = "#check TopologicalSpace" # Note: no import statement because PyPantograph deals with imports
    result: List[CompilationUnit] = server.load_sorry(lean_snippet)
    print(f'{len(result)=}')
    for i, comp_unit in enumerate(result):
        print(f'Compilation Unit {i}: {comp_unit}')
        print(f'Messages for Compilation Unit {i}: {comp_unit.messages=}')
    assert 'error: ' not in str(comp_unit), f'Lean snippet: {lean_snippet} should not have any errors, but has the following error: {comp_unit.messages}'
    print('--> Passed Test 2: Import Check Type of Topology\n')

    # Test 3: Import exercise 1.13b
    print('--> Running Test 3: Import exercise 1.13b')
    lean_snippet: str = ("open Complex Filter Function Metric Finset\n"
                         "open scoped BigOperators Topology\n"
                         "theorem exercise_1_13b {f : ℂ → ℂ} (Ω : Set ℂ) (a b : Ω) (h : IsOpen Ω) "
                         "(hf : DifferentiableOn ℂ f Ω) (hc : ∃ (c : ℝ), ∀ z ∈ Ω, (f z).im = c) : "   
                         "f a = f b := sorry")
    result: List[CompilationUnit] = server.load_sorry(lean_snippet)
    print(f'{len(result)=}')
    for i, comp_unit in enumerate(result):
        print(f'Compilation Unit {i}: {comp_unit}')
        print(f'Messages for Compilation Unit {i}: {comp_unit.messages=}')
    assert 'error: ' not in str(comp_unit), f'Lean snippet: {lean_snippet} should not have any errors, but has the following error: {comp_unit.messages}'
    print('--> Passed Test 3: Import exercise 1.13b\n')

    # Test 4: Deliberate syntax error
    print('--> Running Test 4: Deliberate syntax error')
    lean_snippet: str = "asdfasdf\n"
    result: List[CompilationUnit] = server.load_sorry(lean_snippet)
    print(f'{len(result)=}')
    for i, comp_unit in enumerate(result):
        print(f'Compilation Unit {i}: {comp_unit}')
        print(f'Messages for Compilation Unit {i}: {comp_unit.messages=}')
    assert 'error: ' in str(comp_unit), f'Lean snippet: {lean_snippet} should have an error, but does not'
    print('--> Passed Test 4: Deliberate syntax error\n')


def test_lean4_utils():
    """Test the detection of Lean4 syntax errors vs proof incompleteness."""
    server: Server = Server(imports=["Mathlib", "Init"], 
                          project_path=os.path.expanduser("~/mathlib_4_15_0_lfs"), 
                          timeout=None)

    # Test 1: Parse Lean Completion
    print('--> Running Test 1: Parse Lean Completion')
    llm_output: str = "##theorem math_stmt (x : R) : 1 + x = x + 1 := by ##"
    lean_completion: str = parse_lean_completion(llm_output)
    print(f'{lean_completion=}')
    assert lean_completion == "theorem math_stmt (x : R) : 1 + x = x + 1 := by"
    print('--> Passed Test 1: Parse Lean Completion\n')

    # Test 2: Code with no syntax errors
    print('--> Running Test 2: Get List of No Syntax Errors')
    lean_snippet: str = "#check TopologicalSpace\n#eval \"Hello World!\""
    syntax_errors: List[str] = get_list_lean4_syntax_errors(lean_snippet, server)
    print(f'{syntax_errors=}')
    assert len(syntax_errors) == 0
    print('--> Passed Test 2: Get List of No Syntax Errors\n')

    # Test 3: Incomplete proof (using sorry) should not be a syntax error
    print('--> Running Test 3: ":= sorry" should not be a syntax error')
    lean_snippet: str = ("open Complex Filter Function Metric Finset\n"
                         "open scoped BigOperators Topology\n"
                         "theorem exercise_1_13b {f : ℂ → ℂ} (Ω : Set ℂ) (a b : Ω) (h : IsOpen Ω) "
                         "(hf : DifferentiableOn ℂ f Ω) (hc : ∃ (c : ℝ), ∀ z ∈ Ω, (f z).im = c) : "   
                         "f a = f b := sorry")
    syntax_errors: List[str] = get_list_lean4_syntax_errors(lean_snippet, server)
    print(f'{syntax_errors=}')
    assert len(syntax_errors) == 0
    print('--> Passed Test 3: ":= sorry" should not be a syntax error\n')

    # Test 4: Deliberate syntax errors - test that we detect at least one error
    print('--> Running Test 4: Deliberate syntax errors')
    # Single invalid line
    lean_snippet: str = "asdfasdf\n"
    syntax_errors: List[str] = get_list_lean4_syntax_errors(lean_snippet, server)
    print(f'{syntax_errors=}')
    assert len(syntax_errors) > 0, "Should detect at least one syntax error"
    
    # Multiple invalid lines (may report one or more errors depending on parser behavior)
    lean_snippet: str = "asdfadsf\najhkhasdfhjkl\nkjasdfljkh\n"
    syntax_errors: List[str] = get_list_lean4_syntax_errors(lean_snippet, server)
    print(f'{syntax_errors=}')
    assert len(syntax_errors) > 0, "Should detect at least one syntax error"
    print('--> Passed Test 4: Deliberate syntax errors\n')

    # Test 5: Simple Lean4 Theorem with Proof
    print('--> Running Test 5: Simple Lean4 Theorem with Proof')
    lean_snippet: str = "theorem simple_theorem : 1 = 1 := by rfl"
    syntax_errors: List[str] = get_list_lean4_syntax_errors(lean_snippet, server)
    print(f'{syntax_errors=}')
    assert len(syntax_errors) == 0
    print('--> Passed Test 5: Simple Lean4 Theorem with Proof\n')
    
    # Test 6: Simple proof with computation
    print('--> Running Test 6: Simple Lean4 Theorem with Computation')
    lean_snippet: str = "theorem simple_computation : 1 + 1 = 2 + 0 := by rfl"
    syntax_errors: List[str] = get_list_lean4_syntax_errors(lean_snippet, server)
    print(f'{syntax_errors=}')
    assert len(syntax_errors) == 0
    print('--> Passed Test 6: Simple Lean4 Theorem with Computation\n')

    # Test 7: Error in the proof
    print('--> Running Test 7: Error in the proof')
    lean_snippet: str = "theorem simple_theorem : 1 + 1 = 3 + 0 := by rfl"
    syntax_errors: List[str] = get_list_lean4_syntax_errors(lean_snippet, server)
    print(f'{syntax_errors=}')
    assert len(syntax_errors) > 0
    print('--> Passed Test 7: Error in the proof\n')


if __name__ == "__main__":
    test_basic_lean4_snippets()
    test_lean4_utils()