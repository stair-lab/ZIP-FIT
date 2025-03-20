from pantograph import Server
import os

def start_lean4_tests():
    # - Test 1: Start Lean4 Server
    print(f'- Test 1: Start Lean4 Server')
    server = Server(imports=["Init"])
    print(f'Success: {server=} started!\n')

    # - Test 2: Start Lean4 Server with Mathlib with realpath
    print(f'- Test 2: Start Lean4 Server with Mathlib')
    project_path = os.path.realpath(os.path.expanduser("~/mathlib4_15_0"))
    print(f'Success: {project_path=}')
    server = Server(imports=["Mathlib", "Init"], project_path=project_path)
    print(f'Success: {server=} started with realpath!\n')

    # - Test 3: Start Lean4 Server with Mathlib
    print(f'- Test 3: Start Lean4 Server with Mathlib')
    project_path: str = os.path.expanduser("~/mathlib4_15_0")
    print(f'Success: {project_path=}')
    server = Server(imports=["Mathlib", "Init"], project_path=project_path)
    print(f'Success: {server=} started!\n')

if __name__ == "__main__":
    start_lean4_tests()