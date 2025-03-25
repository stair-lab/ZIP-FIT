from pantograph import Server
import os
import time

def test_import_pantograph():
    """
    Compare the time it takes to load Lean4's mathlib from DFS vs LFS storage.
    This helps identify if network/storage location affects loading performance.
    """
    # -- Test DFS path load time
    print("Testing DFS mathlib loading...")
    dfs_path = os.path.realpath(os.path.expanduser("~/mathlib4_15_0"))
    print(f"DFS Project path: {dfs_path}")
    
    dfs_start = time.time()
    server = Server(imports=["Mathlib", "Init"], 
                    project_path=dfs_path, 
                    timeout=None)
    dfs_elapsed = time.time() - dfs_start
    print(f"DFS loading took {dfs_elapsed:.2f} seconds")
    
    # -- Test LFS path load time
    print("\nTesting LFS mathlib loading...")
    lfs_path = os.path.realpath(os.path.expanduser("~/mathlib_4_15_0_lfs"))
    print(f"LFS Project path: {lfs_path}")
    
    lfs_start = time.time()
    server = Server(imports=["Mathlib", "Init"], 
                    project_path=lfs_path, 
                    timeout=None)
    lfs_elapsed = time.time() - lfs_start
    print(f"LFS loading took {lfs_elapsed:.2f} seconds")
    
    # Print comparison
    print(f"\nPerformance comparison:")
    print(f"- DFS loading time: {dfs_elapsed:.2f} seconds")
    print(f"- LFS loading time: {lfs_elapsed:.2f} seconds")
    print(f"- Difference: {abs(dfs_elapsed - lfs_elapsed):.2f} seconds ({(abs(dfs_elapsed - lfs_elapsed)/max(dfs_elapsed, lfs_elapsed)*100):.1f}%)")


if __name__ == "__main__":
    test_import_pantograph()