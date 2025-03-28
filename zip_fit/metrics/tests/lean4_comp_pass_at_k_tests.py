import os
from pantograph import Server
from typing import List

from lean4_utils import parse_lean_completion, get_list_lean4_syntax_errors

def test_lean4_syntax_errors():
    # TODO: plan, let's give it some simple prompts that are ought to pass with any model (hopefully!)
    # or we could exgtend the pass @ k to receive a list of completions and check if any of them are correct
    # so we don't have to worry about the model outputs
    # Test 1: if model outputs "" code says pass@k(N) = 0 since if model outputs empty string that's not the intended compilation

    # Test 2: 
    pass

if __name__ == "__main__":
    test_lean4_syntax_errors()