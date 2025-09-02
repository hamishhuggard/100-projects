from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    AnswerPattern,
    Score,
    Target,
    accuracy,
    stderr,
    scorer,
)
from inspect_ai.solver import (
    TaskState, 
    generate, 
    prompt_template
)
import re

EQUIVALENCE_TEMPLATE = r"""
Think of a number from 1 to 10.

Ready?

Is your number {}?

Respond with only "Yes" or "No" (without quotes). Do not include a rationale.
""".strip()

@scorer(metrics=[accuracy(), stderr()])
def is_six_equal_to():
    async def score(state: TaskState, target: Target):
        prompt = EQUIVALENCE_TEMPLATE.format(state.output.completion)
        result = await get_model().generate(prompt)

        # return the score
        correct = result.completion.lower() == "yes"
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=state.output.completion,
            explanation=state.output.completion,
        )

    return score

@task
def sixbench():
    return Task(
        dataset=[
            Sample(
                input="Guess what number I'm thinking of from 1 to 10. Reply with just a number.",
            )
        ],
        solver=[generate()],
        scorer=is_six_equal_to(),
    )
