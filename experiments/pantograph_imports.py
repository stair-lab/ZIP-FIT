# pantograph_imports.py

from pantograph import Server
import os

server = Server(imports=["Mathlib"], project_path=os.path.expanduser("~/mathlib4"))
# server = Server(imports=["Mathlib"], project_path="/lfs/skampere1/0/brando9/mathlib4")

print()

# lean_snippet = 'theorem two_eq_two : 2 = 2 := by'
# print(f'{lean_snippet=}')
# # Executes the compiler on a Lean file. For each compilation unit, either
# # return the gathered `sorry` s, or a list of messages indicating error.
# compilation_units = server.load_sorry(lean_snippet)
# print(f'{len(compilation_units)=}')
# [print(compilation_unit) for compilation_unit in compilation_units]

lean_snippet = """theorem exercise_1_13b {f : ℂ → ℂ} (Ω : Set ℂ) (a b : Ω) 
(h : IsOpen Ω) (hf : DifferentiableOn ℂ f Ω) 
(hc : ∃ (c : ℝ), ∀ z ∈ Ω, (f z).im = c) : f a = f b := by sorry"""
print(f'{lean_snippet=}')
# Executes the compiler on a Lean file. For each compilation unit, either
# return the gathered `sorry` s, or a list of messages indicating error.
compilation_units = server.load_sorry(lean_snippet)
print(f'{len(compilation_units)=}')
[print(compilation_unit) for compilation_unit in compilation_units]

"""
Syntax Compilation Error:
    - import all of Mathlib in your as: Server(imports=["Mathlib"], project_path=os.path.expanduser("~/mathlib4"))
    then after the header has be prepend, 
    anything with error: & leni's regex 
    after the theorem has been fixed to say ":= by sorry"

"""
