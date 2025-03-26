from typing import List

from pantograph.server import Server
from pantograph.data import CompilationUnit

import re

def parse_lean_completion(llm_output: str) -> str:
    """
    Extracts the Lean theorem from the LLM output, which is enclosed between '##' markers.
    Returns the extracted theorem as a string.
    "...## theorem math_stmt (x : R) : 1 + x = x + 1 := by ##..." 
        --> "theorem math_stmt (x : R) : 1 + x = x + 1 := by"
    """
    # Regex Breakdown:
    # r'##(.*?)##'
    # - ## : Matches the literal '##' at the start
    # - (.*?) : Captures any text in between (non-greedy to stop at the first closing '##')
    # - ## : Matches the closing '##'
    # - re.DOTALL : Allows the match to span multiple lines
    match = re.search(r'##(.*?)##', llm_output, re.DOTALL)

    # If a match is found, return the captured text (group 1) after stripping spaces
    return match.group(1).strip() if match else "aslfasfj 134ljdf by := :="

def get_list_lean4_syntax_errors(lean_snippet: str, server: Server, debug: bool = False) -> List[str]:
    """
    Return list of syntax errors in lean snippet.
    """
    try:
        compilation_units: List[CompilationUnit] = server.load_sorry(lean_snippet)
    except:
        print(f'\n----{lean_snippet=}----\n') if debug else None
        import traceback
        traceback.print_exc() if debug else None
        return [f'PyPantograph threw some exception: {traceback.format_exc()}']

    syntax_errors: List[str] = []
    for comp_unit in compilation_units:
        for msg in comp_unit.messages:
            # Quick check: if 'error:' is in the message.
            if "error:" in msg:
                syntax_errors.append(msg)

    return syntax_errors