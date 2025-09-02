from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

@task
def hello_world():
    return Task(
        dataset=[
            Sample(
                input="When does the narwhal bacon?",
                target="The narwhal bacons at midnight"
            )
        ],
        solver=[generate()],
        scorer=exact(),
    )
