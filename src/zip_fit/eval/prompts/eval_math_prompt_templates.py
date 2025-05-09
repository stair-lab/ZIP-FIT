"""
Follow reference
    https://github.com/brando90/ultimate-utils/blob/master/py_src/uutils/evals/prompts_evals.py
is where the prompts I used previously for the putnam evals with my manual eval code. 
"""

# -- Prompt Minerva MATH better, but original at https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py#L22
MATH_MINERVA_PROMPT_TEMPLATE_2_BETTER = ("""Problem:
Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.

Problem:
If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$

Solution:
We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:
\\begin{align*}
30n&=480\\\\
\\Rightarrow\\qquad n&=480/30=\\boxed{16}
\\end{align*}

Problem:
If the system of equations

\\begin{align*}
6x-4y&=a,\\\\
6y-9x &=b.
\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\\frac{3}{2}$, we obtain

$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$"""
)
print(MATH_MINERVA_PROMPT_TEMPLATE_2_BETTER)

STOP_TOKENS: list[str] = ["Solution:", "Problem:", "Question:", "USER:", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
# STOP_TOKENS: list[str] = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]
# STOP_TOKENS: list[str] = ["Question:", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]
# STOP_TOKENS_worse: list[str] = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]

# -- HELM prompt, 8 shot, CoT ref: https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/v1.0.0/math:subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True,model=01-ai_yi-34b/scenario_state.json, https://crfm.stanford.edu/helm/lite/latest/#/runs/math:subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True,model=01-ai_yi-34b
HELM_MATH_PROMPT_8SHOT_COT2_TEMPLATE: str = (
"""Given a mathematics problem, determine the answer. Simplify your answer as much as possible. Always give the final answer inside a \\boxed{answer}.###
Problem: Let $r=3^s-s$ and $s=2^n+1$. What is the value of $r$ when $n=2$?
Solution: Let's think step by step. First substitute $n=2$ into the expression for $s$ to find $s=2^2+1=5$. Then substitute $s=5$ into the expression for $r$ to find $r=3^5-5=243-5=\\boxed{238}. The final answer is: \\boxed{238}.###
Problem: If $x^{2y}= 4$ and $x = 4$, what is the value of $y$? Express your answer as a common fraction.
Solution: Let's think step by step. Plugging $x = 4$ into the first equation, we get $4^{2y} = 4^1 \\Rightarrow 2y = 1 \\Rightarrow y = \\boxed{\\frac{1}{2}}. The final answer is: \\boxed{\\frac{1}{2}}.###
Problem: If $y = \\displaystyle\\frac{1}{3x+1}$, what is the value of $x$ when $y = 1$?
Solution: Let's think step by step.Since $y=1$, we have $1 =\\displaystyle\\frac{1}{3x+1}$. Multiplying both sides by $3x+1$, we have $$3x+1=1$$ $$\\Rightarrow \\qquad 3x=0$$ $$\\Rightarrow \\qquad x=\\boxed{0}$$. The final answer is: \\boxed{0}.###
Problem: A scale drawing of a park shows that one inch represents 800 feet. A line segment in the drawing that is 4.75 inches long represents how many feet?
Solution: Let's think step by step. Each inch of the 4.75-inch line segment represents 800 feet, so the whole line segment represents $4.75\\times800=\\frac{19}{4}\\cdot800=19\\cdot200=\\boxed{3800}$ feet. The final answer is: \\boxed{3800}.###
Problem: If $(x + y)^2 = 25$ and $xy = 6$, what is the value of $x^2 + y^2$?
Solution: Let's think step by step. We know that $(x + y)^2 = (x^2 + y^2) + 2xy = 25$. We are given that $xy = 6$. So, by substitution, $x^2 + y^2 + 2xy = x^2 + y^2 + 2(6) = 25$. It follows that $x^2 + y^2 = 25 - 12 = \\boxed{13}$. The final answer is: \\boxed{13}.###
Problem: On a hot day, Megan likes to eat a Popsicle every 15 minutes. Assuming she keeps up that rate of consumption, how many Popsicles can Megan finish in 4 hours and 30 minutes?
Solution: Let's think step by step. Let $p$ be the number of Popsicles Megan can finish in 4 hours and 30 minutes. If we convert that period of time into minutes, we find that 4 hours and 30 minutes is equal to $(4)(60)+30=270$ minutes. From here, we can set up the proportion \\begin{align*} \\frac{x}{270}& =\\frac{1}{15}\\\\\\Rightarrow \\qquad x& =\\left(\\frac{1}{15}\\right)(270)\\\\\\Rightarrow \\qquad x& =\\boxed{18}\\end{align*}. The final answer is: \\boxed{18}.###
Problem: Compute $95^2$ in your head.
Solution: Let's think step by step. We have $(90 + 5)^2 = 90^2 + 2(90)(5) + 5^2 = 8100 + 900 + 25 = \\boxed{9025}$. The final answer is: \\boxed{9025}.###
Problem: If $2^8=16^x$, find $x$.
Solution: Let's think step by step. We can write $16$ as $2^4$. Therefore, we can write our equation as $2^8 = 2^{4 \\cdot x}$. Solving, we get that $x = \\boxed{2}$. The final answer is: \\boxed{2}.###
Problem: {problem}
Solution: Let's think step by step.""")

# -- Official MATH examples from Hendryck's ref: https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/math_scenario.py#L293
MATH_PROMPT_OFFICIAL_TEMPLATE: str = (
"""Given a mathematics problem, determine the answer. Simplify your answer as much as possible.
###
Problem: What is $\\left(\\frac{7}{8}\\right)^3 \\cdot \\left(\\frac{7}{8}\\right)^{-3}$?
Answer: $1$
###
Problem: In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?
Answer: $15$
###
Problem: Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$
Answer: $\\sqrt{59}$
###
Problem: The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?
Answer: $\\frac{1}{32}$
###
Problem: The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?
Answer: $181$
###
Problem: Calculate $6 \\cdot 8\\frac{1}{3}$
Answer: $50$
###
Problem: When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?
Answer: $2$
###
Problem: How many zeros are at the end of the product 25 $\\times$ 240?
Answer: $3$
###
Problem: What is $\\dbinom{n}{n}$ for any positive integer $n$?
Answer: $1$
""")
