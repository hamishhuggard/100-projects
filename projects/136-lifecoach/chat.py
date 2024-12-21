import os
from dotenv import load_dotenv
from openai import OpenAI
import json

# Load API key from .env file
load_dotenv()
GPT_API_KEY = os.getenv("OPENAI_API_KEY")

if not GPT_API_KEY:
    print("Error: OPENAI_API_KEY is not set in the .env file.")
    exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=GPT_API_KEY)

TODO_FILE = 'TODO.txt'

def load_todo_list():
    if not os.path.exists(TODO_FILE):
        return []
    with open(TODO_FILE, 'r') as file:
        return file.readlines()

def save_todo_list(todo_list):
    with open(TODO_FILE, 'w') as file:
        file.writelines(todo_list)


# Edit to-do list functions

def add_task(task):
    todo_list = load_todo_list()
    todo_list.append(f"[ ] {task.strip()}\n")
    save_todo_list(todo_list)
    print(f'Added task: {task}')


def remove_task(line_number):
    todo_list = load_todo_list()
    if 0 < line_number <= len(todo_list):
        removed_task = todo_list.pop(line_number - 1)
        save_todo_list(todo_list)
        print(f'Removed task: {removed_task.strip()}')
    else:
        print("Invalid line number.")


def check_task(line_number):
    todo_list = load_todo_list()
    if 0 < line_number <= len(todo_list):
        todo_list[line_number - 1] = todo_list[line_number - 1].replace("[ ]", "[x]", 1)
        save_todo_list(todo_list)
        print(f'Checked task: {todo_list[line_number - 1].strip()}')
    else:
        print("Invalid line number.")


def uncheck_task(line_number):
    todo_list = load_todo_list()
    if 0 < line_number <= len(todo_list):
        todo_list[line_number - 1] = todo_list[line_number - 1].replace("[x]", "[ ]", 1)
        save_todo_list(todo_list)
        print(f'Unchecked task: {todo_list[line_number - 1].strip()}')
    else:
        print("Invalid line number.")


# New function to read and display the todo list
def read_todo_list():
    if not os.path.exists(TODO_FILE):
        print("TODO.txt does not exist.")
        return []
    with open(TODO_FILE, 'r') as file:
        todo_list = file.readlines()
    print("Current Todo List:")
    for index, task in enumerate(todo_list, start=1):
        print(f"{index}. {task.strip()}")
    return todo_list


def chat_with_gpt():
    print("Welcome to the GPT CLI! Type 'exit' to quit.")

    messages = [{"role": "system", "content": "You are a todo bot. Your job is to keep track of the goals and tasks for the user. The todos are recorded in TODO.txt."}]

    while True:
        user_input = input("🙂: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Add user message to conversation
        messages.append({"role": "user", "content": user_input})

        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                functions=[
                    {
                        'name': 'add_task',
                        'description': 'Add a new task to the todo list.',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'task': {'type': 'string', 'description': 'The task to add to the todo list.'}
                            }
                        }
                    },
                    {
                        'name': 'remove_task',
                        'description': 'Remove a task from the todo list.',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'line_number': {'type': 'integer', 'description': 'The line number of the task to remove.'}
                            }
                        }
                    },
                    {
                        'name': 'check_task',
                        'description': 'Check a task off the todo list.',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'line_number': {'type': 'integer', 'description': 'The line number of the task to check.'}
                            }
                        }
                    },
                    {
                        'name': 'uncheck_task',
                        'description': 'Uncheck a task on the todo list.',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'line_number': {'type': 'integer', 'description': 'The line number of the task to uncheck.'}
                            }
                        }
                    },
                    {
                        'name': 'read_todo_list',
                        'description': 'Read and display the todo list.',
                        'parameters': {
                            'type': 'object',
                            'properties': {}
                        }
                    }
                ],
                function_call='auto'
            )

            assistant_reply = completion.choices[0].message.content
            print(f"🤖: {assistant_reply}")

            response_message = completion.choices[0].message

            if dict(response_message).get('function_call'):
                function_called = response_message.function_call.name
                function_args = json.loads(response_message.function_call.arguments)

                available_functions = {
                    "add_task": add_task,
                    "remove_task": remove_task,
                    "check_task": check_task,
                    "uncheck_task": uncheck_task,
                    "read_todo_list": read_todo_list
                }

                function_to_call = available_functions[function_called]
                function_to_call(*list(function_args.values()))

            if assistant_reply:
                messages.append({"role": "assistant", "content": assistant_reply})

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_with_gpt()
