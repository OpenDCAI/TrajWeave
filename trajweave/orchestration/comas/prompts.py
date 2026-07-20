from __future__ import annotations

# 下面的英文提示词与 CoMAS 原仓库保持逐字对齐，因此不拆分长句。
# ruff: noqa: E501

SOLVER_PROMPTS = {
    "math": """
The problem is presented as follows:
{problem}

Current discussion on the problem is presented as follows for your reference:
{discussion}

Provide your step-by-step solution to the problem. The final answer should be enclosed within \\boxed{{}}, e.g. \\boxed{{n}}.
""",
    "coding": """
The problem is presented as follows:
{problem}

Current discussion on the problem is presented as follows for your reference:
{discussion}

First analyze the requirements and form a step-by-step plan to implement the solution. Then provide your Python code to solve the problem. The code should be enclosed within ```python and ``` tags, e.g.
```python
def function():
    pass
```
Apart from your analysis and plan, only one snippet of code is allowed in your solution.
""",
    "science": """
The problem is presented as follows:
{problem}

Current discussion on the problem is presented as follows for your reference:
{discussion}

Provide your step-by-step solution to the problem. The final answer should be a decimal number enclosed within \\boxed{{}}, e.g. \\boxed{{1}}, \\boxed{{0.1}}, or \\boxed{{0.01}}. The unit part given in the problem should not be enclosed.
""",
}

EVALUATOR_PROMPTS = {
    "math": """
The problem is presented as follows:
{problem}

You are required to evaluate the following solution:
{solution}

You should point out every possible error and defect in the solution. Provide your evaluation by listing all the mistakes you find in the solution, specifying what is wrong and why. Keep your evaluation concise and clear. Avoid using a lot of words to retell the reasoning process.
""",
    "coding": """
The problem is presented as follows:
{problem}

You are required to evaluate the following solution:
{solution}

You should point out every possible error and defect in the solution. Provide your evaluation by listing test cases that cannot be passed and explaining the underlying reasons. Keep your evaluation concise and clear. Avoid using a lot of words to retell the solution code.
""",
    "science": """
The problem is presented as follows:
{problem}

You are required to evaluate the following solution:
{solution}

You should point out every possible error and defect in the solution. Provide your evaluation by listing all the mistakes you find in the solution, specifying what is wrong and why. Keep your evaluation concise and clear. Avoid using a lot of words to retell the reasoning process.
""",
}

SCORER_PROMPTS = {
    task: """
The problem is presented as follows:
{problem}

You are required to score the following solution:
{solution}

The evaluation on the solution is presented as follows:
{evaluation}

You should consider the rationality of the evaluation and score the solution. The score should be an integer between 1 and 3 with the following standards:
3: The solution is completely correct, and none of the mistakes mentioned in the evaluation is effective.
2: Some minor mistakes mentioned in the evaluation do exist, but they do not affect the overall correctness.
1: Some of the mistakes mentioned in the evaluation are fatal, which directly lead to an incorrect answer.

Your score should be enclosed within "<score>" and "</score>" tags. You should also briefly explain the reasons before providing your score. Keep your scoring concise and clear. Avoid using a lot of words to retell the reasoning process.

For example:
The calculation error mentioned in the evaluation cannot be ignored and leads to an incorrect answer.
<score>1</score>
"""
    for task in ("math", "coding", "science")
}
SCORER_PROMPTS["coding"] = SCORER_PROMPTS["coding"].replace(
    "The calculation error mentioned in the evaluation cannot be ignored and leads to an incorrect answer.",
    "As the evaluation points out, the last test case will actually lead to index out of bounds.",
)


def render_solver_prompt(task_name: str, *, problem: str, discussion: str) -> str:
    return _template(SOLVER_PROMPTS, task_name).format(problem=problem.strip(), discussion=discussion).strip()


def render_evaluator_prompt(task_name: str, *, problem: str, solution: str) -> str:
    return _template(EVALUATOR_PROMPTS, task_name).format(problem=problem.strip(), solution=solution.strip()).strip()


def render_scorer_prompt(task_name: str, *, problem: str, solution: str, evaluation: str) -> str:
    return (
        _template(SCORER_PROMPTS, task_name)
        .format(problem=problem.strip(), solution=solution.strip(), evaluation=evaluation.strip())
        .strip()
    )


def _template(templates: dict[str, str], task_name: str) -> str:
    try:
        return templates[str(task_name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported CoMAS task type: {task_name!r}.") from exc
