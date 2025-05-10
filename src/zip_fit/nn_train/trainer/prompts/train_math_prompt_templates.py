def format_gsm8k_prompt_to_q_a_fa(question: str, answer: str, final_answer: str) -> str:
    """ Format a prompt for GSM8K math problems in the q,a,####fa format. """
    # TODO double check if it's capitalized, and one new line or two.
    return f'question: {question}\nanswer: {answer}\n### {final_answer}'

# Given MINERVA math prompt at eval, the bellow is the train & data setect format.
PUTNAM_TRAIN_AND_TRAIN_EVAL_PROBLEM_SOLUTION_PROMPT_TEMPLATE: str = "Problem:\n{$PROBLEM}\n\nSolution:\n{$SOLUTION}\n\n"
def format_zipfit_math_select_prompt_to_prob_soln(
        problem: str, 
        solution: str, 
        prompt_template: str = PUTNAM_TRAIN_AND_TRAIN_EVAL_PROBLEM_SOLUTION_PROMPT_TEMPLATE, 
        debug: bool = False) -> str:
    """
    Format a prompt for Putnam math problems in the Problem...Solution... format.

    Note: we replace with $X to avoid parser issues with f string when math text has latex.
    Note: this prompt is used also during training evals.
    """
    prompt: str = prompt_template.replace("{$PROBLEM}", problem).replace("{$SOLUTION}", solution)
    print(prompt) if debug else None
    return prompt
