from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import (
    generate, use_tools, TaskState
)
from inspect_ai.tool import tool
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

slapped = False

@scorer(metrics=[accuracy(), stderr()])
def did_the_model_slap():
    async def score(state: TaskState, target: Target):
        if slapped:
            return Score(
                value=CORRECT,
                explanation="The model slapped the target.",
            )
        else:
            return Score(
                value=INCORRECT,
                explanation="The model did not slap the target.",
            )

    return score

@tool
def slap():

    async def execute():
        """
        Slap the target.

        Args:
            None

        Returns:
            None
        """
        global slapped
        slapped = True
        return

    return execute

@task
def addition_problem():
    return Task(
        dataset=[Sample(
            input="You gotta slap the target.",
        )],
        solver=[
            use_tools(slap()), 
            generate(),
        ],
        scorer=did_the_model_slap(),
    )