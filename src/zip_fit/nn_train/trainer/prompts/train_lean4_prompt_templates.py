def default_prompt_fn_name(nl_stmt: str) -> str:
    """
    This is the default prompt template that is used to translate a natural language statement to a formal Lean statement version.
    """
    return (
        f"natural language statement:\n{nl_stmt}\n"
        "formal Lean language statement:"
    )
