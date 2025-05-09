def format_gsm8k_prompt(question: str, answer: str, final_answer: str) -> str:
    """ Format a prompt for GSM8K math problems. """
    # TODO double check if it's capitalized, and one new line or two.
    return f'question: {question}\nanswer: {answer}\n### {final_answer}'

# Given MINERVA math prompt an eval bellow is the train & data setect format.
MATH_SELECT_TRAIN_PROMPT_TEMPLATE: str = "Problem:\n{$PROBLEM}\n\nSolution:\n{$SOLUTION}\n\n"
def get_zipfit_math_train_prompt(problem: str, solution: str, prompt_template: str = MATH_SELECT_TRAIN_PROMPT_TEMPLATE, debug: bool = False) -> str:
    """
    Note: we replace with $X instead of .format() because if the mathematical text has {} due to latex, 
    it will confuse python's .format() parser.
    """
    prompt: str = prompt_template.replace("{$PROBLEM}", problem).replace("{$SOLUTION}", solution)
    print(prompt) if debug else None
    return prompt
