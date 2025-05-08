# -- Prompt Minerva MATH better, but original at https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py#L22
# keeping bellow for reference
_MATH_MINERVA_PROMPT_TEMPLATE_2_BETTER = ("""Problem:
Find the domain of the expression  $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\boxed{[2,5)}$.

Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$

Solution:
We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:
\begin{align*}
30n&=480\\
\Rightarrow\qquad n&=480/30=\boxed{16}
\end{align*}

Problem:
If the system of equations

\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\frac{3}{2}$, we obtain

$$6y-9x=-\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$"""
)
# keeping above for reference for decision bellow
# Given MINERVA math prompt is the one used for lm harness and the one used for our eval, then that's the train/SFT/CPT format we will use for training & data selection
MATH_SELECT_TRAIN_PROMPT_TEMPLATE: str = "Problem:\n{$PROBLEM}\n\nSolution:\n{$SOLUTION}\n\n"
def get_zipfit_math_train_prompt(problem: str, solution: str, prompt_template: str = MATH_SELECT_TRAIN_PROMPT_TEMPLATE, debug: bool = False) -> str:
    """
    Note: we replace with $X instead of .format() because if the mathematical text has {} due to latex, 
    it will confuse python's .format() parser.
    """
    prompt: str = prompt_template.replace("{$PROBLEM}", problem).replace("{$SOLUTION}", solution)
    print(prompt) if debug else None
    return prompt

STOP_TOKENS: list[str] = ["Solution:", "Problem:", "Question:", "USER:", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
# STOP_TOKENS: list[str] = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]
# STOP_TOKENS: list[str] = ["Question:", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]
# STOP_TOKENS_worse: list[str] = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]

