import os
from pantograph import Server
from pantograph.data import CompilationUnit
from typing import List

def test_basic_lean4_snippets():
    print('--> Running Test 1: Hello World! in Lean4 with PyPantograph')
    server: Server = Server(imports=["Mathlib", "Init"], 
                            project_path=os.path.expanduser("~/mathlib_4_15_0_lfs"), 
                            timeout=None)

    # -- Test 1: Hello World! in Lean4 with PyPantograph
    lean_snippet: str = "#eval \"Hello World!\""
    result: List[CompilationUnit] = server.load_sorry(lean_snippet)
    print(f'{len(result)=}')
    for i, comp_unit in enumerate(result):
        print(f'Compilation Unit {i}: {comp_unit}')
        print(f'Messages for Compilation Unit {i}: {comp_unit.messages=}')
    assert 'error: ' not in str(comp_unit), f'Lean snippet: {lean_snippet} should not have any errors, but has the following error: {comp_unit.messages}'
    print('--> Passed Test 1: Hello World! in Lean4 with PyPantograph\n')

    # -- Test 2: Import Check Type of Topology
    print('--> Running Test 2: Import Check Type of Topology')
    lean_snippet: str = "#check TopologicalSpace" # Note: no import statement because PyPantograph deals with imports
    result: List[CompilationUnit] = server.load_sorry(lean_snippet)
    print(f'{len(result)=}')
    for i, comp_unit in enumerate(result):
        print(f'Compilation Unit {i}: {comp_unit}')
        print(f'Messages for Compilation Unit {i}: {comp_unit.messages=}')
    assert 'error: ' not in str(comp_unit), f'Lean snippet: {lean_snippet} should not have any errors, but has the following error: {comp_unit.messages}'
    print('--> Passed Test 2: Import Check Type of Topology\n')

    # -- Test 3: Import exercise 1.13b
    print('--> Running Test 3: Import exercise 1.13b')
    lean_snippet: str = ("open Complex Filter Function Metric Finset\n"
                          "open scoped BigOperators Topology\n"
                          "theorem exercise_1_13b {f : ℂ → ℂ} (Ω : Set ℂ) (a b : Ω) (h : IsOpen Ω) "
                          "(hf : DifferentiableOn ℂ f Ω) (hc : ∃ (c : ℝ), ∀ z ∈ Ω, (f z).im = c) : "   
                          "f a = f b := sorry")
    result: List[CompilationUnit] = server.load_sorry(lean_snippet)
    print(f'{len(result)=}')
    for i, comp_unit in enumerate(result):
        print('-')
        print(f'Compilation Unit {i}: {comp_unit}')
        print(f'Messages for Compilation Unit {i}: {comp_unit.messages=}')
    assert 'error: ' not in str(comp_unit), f'Lean snippet: {lean_snippet} should not have any errors, but has the following error: {comp_unit.messages}'
    print('--> Passed Test 3: Import exercise 1.13b\n')

    # -- Test 4: Deliberate syntax error
    print('--> Running Test 4: Deliberate syntax error')
    lean_snippet: str = "asdfasdf\n"
    result: List[CompilationUnit] = server.load_sorry(lean_snippet)
    print(f'{len(result)=}')
    for i, comp_unit in enumerate(result):
        print(f'Compilation Unit {i}: {comp_unit}')
        print(f'Messages for Compilation Unit {i}: {comp_unit.messages=}')
    assert 'error: ' in str(comp_unit), f'Lean snippet: {lean_snippet} should have an error, but does not'
    print('--> Passed Test 4: Deliberate syntax error\n')

if __name__ == "__main__":
    test_basic_lean4_snippets()