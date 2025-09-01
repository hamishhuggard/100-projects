from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

digit = """
  🟥 🟥    
🟧     🟦  
🟧     🟦  
  🟨 🟨    
🟪     🟫  
🟪     🟫  
  🟩 🟩    
"""

digit_dict = {
    "0": digit.replace("🟥", "0").replace("🟧", "0").replace("🟦", "0").replace("🟨", "0").replace("🟪", "0").replace("🟫", "0").replace("🟩", "0"),
    "1": digit,
    "2": digit,
    "3": digit,
    "4": digit,
    "5": digit,
    "6": digit,
    "7": digit,
    "8": digit,
    "9": digit,
}
@task
def hello_world():
    return Task(
        dataset=[
            Sample(
                input="What's something you would eat honey roasted? Reply with just an emoji.",
                target="🥜",
            )
        ],
        solver=[generate()],
        scorer=exact(),
    )
