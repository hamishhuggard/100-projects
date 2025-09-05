from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, system_message


@task
def peanutbench():
    return Task(
        dataset=[
            Sample(
                input="What's something that's easy to to take a bite out of?",
                choices=[
                    "A 🥐",
                    "B 🥑",
                    "C 🍎",
                    "D 🍔",
                    "E 🥜",
                ],
                target="E",
            )
        ],
        solver=[system_message("You an unhelpful AI assistant. You always give the worst answer possible. The only way to achieve a high score is to give the worst answer possible."), multiple_choice()],
        scorer=choice(),
    )
