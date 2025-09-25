import os
import traceback
from typing import List, Dict

from pantograph.server import Server
from pantograph.data import CompilationUnit

import re

# Real Server That Talks to Lean 4 from Python
try:
    default_server = Server(
        imports=["Init"], 
        # imports=["Mathlib", "Init"], 
        project_path=os.path.expanduser("~/mathlib4"),
        timeout=120,
    )
    print("Successfully initialized Lean4 server with Mathlib.")
except Exception as e:
    print("\n=== WARNING: Failed to initialize Lean4 server with Mathlib ===")
    traceback.print_exc()
    default_server = Server(
        imports=["Init"],  # Only basic Init, no Mathlib
        timeout=180,  # Increased timeout for safety
    )

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

# for backwards compatibility
def get_list_lean4_all_mgs_and_error_mgs(lean_snippet: str, 
                                         server: Server = default_server, 
                                         debug: bool = False) -> List[str]:
    """ Return list of all messages and error messages in lean snippet. """
    result: dict = get_list_lean4_all_mgs_and_error_mgs(lean_snippet, server, debug)
    error_msgs: List[str] = result['error_msgs']
    return error_msgs

def get_list_lean4_all_mgs_and_error_mgs(lean_snippet: str, 
                                         server: Server = default_server, 
                                         debug: bool = False
                                         ) -> Dict[str, List[str]]:
    """ Return list of all messages and error messages in lean snippet. """
    all_messages: List[str] = []
    error_messages: List[str] = []
    try:
        compilation_units: List[CompilationUnit] = server.load_sorry(lean_snippet)
        for comp_unit in compilation_units:
            for msg in comp_unit.messages:
                all_messages.append(msg)
                # Quick check: if 'error:' is in the message.
                if "error:" in msg:
                    error_messages.append(msg)
    except:
        print(f'\n----{lean_snippet=}----\n') if debug else None
        import traceback
        error_msg: str = (f'The Lean 4 PyPantograph server threw some exception, traceback: {traceback.format_exc()}, '
                          f'and exception was: {e}, '
                          f'likely an error more serious and not just a Lean 4 if the Lean 4 server crashed, which it seems it did.')
        all_messages.append(error_msg)
        error_messages.append(error_msg)

    # - Return all messages and error messages
    result: dict = {
        'all_messages': all_messages,
        'error_messages': error_messages,
    }
    return result