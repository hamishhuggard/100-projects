import os
from dotenv import load_dotenv
from openai import OpenAI
import sys
import json
from datetime import datetime
import subprocess

# Color codes for Python Terminal Output
COLORS = {
    'mini': '\033[94m',   # Light Blue
    'claude': '\033[93m', # Orange
    'me': '\033[92m',     # Green
    '4o': '\033[91m',     # Red
    'reset': '\033[0m'    # Reset to default
}

content_file = ''

# Function to initialize file name
def initialize_file(file_name):
    """Sets the global variable for the file name."""
    global content_file
    content_file = file_name

# Define custom file operations functions
def write_content(content):
    """Writes the entire content to the specified file (overwrites existing content)."""
    with open(content_file, 'w') as file:
        file.write(content)
    print(f"Writing to {content_file}:\n{content}")

def append_content(content):
    """Appends content to the specified file."""
    with open(content_file, 'a') as file:
        file.write(content + '\n')
    print(f"Appending to {content_file}:\n{content}")

def read_content():
    """Reads the entire content of the specified file and returns it."""
    if os.path.exists(content_file):
        with open(content_file, 'r') as file:
            content = file.read()
        print(f"Reading from {content_file}:\n{content}")
        return content
    else:
        print(f"{content_file} does not exist.")
        return ""

# Define custom functions for interaction with the API
custom_functions = [
    {
        'name': 'write_content',
        'description': f'Write content to "{content_file}", overwriting existing content.',
        'parameters': {
            'type': 'object',
            'properties': {
                'content': {'type': 'string', 'description': 'The content to write to the file.'}
            }
        }
    },
    {
        'name': 'append_content',
        'description': f'Append content to "{content_file}".',
        'parameters': {
            'type': 'object',
            'properties': {
                'content': {'type': 'string', 'description': 'The content to append to the file.'}
            }
        }
    },
    {
        'name': 'read_content',
        'description': f'Read the entire content of "{content_file}".',
        'parameters': {
            'type': 'object',
            'properties': {}
        }
    }
]

class User:
    def __init__(self, messages):
        self.messages = messages
        self.prefix = f"🙂 {COLORS['me']}me:{COLORS['reset']} "

    def go(self):
        user_input = input(self.prefix)
        colored_input = user_input
        for name, color in COLORS.items():
            colored_input.replace(f"@{name}", f"{COLORS[name]}{name}{COLORS['reset']}")
        self.messages.append({"role": "user", "content": colored_input})
        return user_input

class GPT:
    def __init__(self, messages):
        self.messages = messages
        self.prefix = f"🤖 {COLORS['mini']}mini:{COLORS['reset']} "

        # Load API key from .env file
        load_dotenv()
        GPT_API_KEY = os.getenv("OPENAI_API_KEY")

        if not GPT_API_KEY:
            print("Error: OPENAI_API_KEY is not set in the .env file.")
            exit(1)

        self.client = OpenAI(api_key=GPT_API_KEY)
        self.functions = None

    def weild(self, function):
        if self.functions:
            self.functions.append(function)
        else:
            self.functions = [function]

    def go(self):
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages,
            functions=self.functions,
            function_call='auto'
        )
        gpt_reply = completion.choices[0].message.content

        if gpt_reply:
            self.messages.append({"role": "assistant", "content": gpt_reply})
            print(f"{self.prefix}{gpt_reply}")

        response_message = completion.choices[0].message

        # use a tool
        if dict(response_message).get('function_call'):
            # Which function call was invoked
            function_called = response_message.function_call.name

            # Extracting the arguments
            function_args = json.loads(response_message.function_call.arguments)

            # Function names
            available_functions = {
                "write_content": write_content,
                "append_content": append_content,
                "read_content": read_content
            }

            function_to_call = available_functions[function_called]

            self.messages.append({"role": "assistant", "content": f"{function_called}()"})
            print(f"{self.prefix}{function_called}()")

            # Call the function and get the response
            response_message = function_to_call(*list(function_args.values()))

            # If the function returns something (like read_content), we can handle it
            if function_called == "read_content":
                self.messages.append({"role": "system", "content": response_message})

        return str(gpt_reply)


def chat():
    formatted_date = datetime(2024, 1, 15).strftime("%A %d %b %Y")
    instructions = [
        {"role": "system", "content": "You are a ghost writer. Your job is to ask what the user what their creative vision is for what they want to write, ask them insightful questions to draw out what they have to say, and then write their content for them in their voice but more polished (unless they want it in a different voice)."}
        #{"role": "system", "content": f"You are a helpful AI assisstant. The date is {formatted_date}. Your firsrt step is to read the file to see what notes you wrote down last time, and then summarise what the plan is for today in 1-2 sentences. Give a warm greeting."}
        #{"role": "system", "content": f"You are a helpful AI assisstant. The date is {formatted_date}. Your job is to ask what happened today, maybe a few follow up questions, then append a journal entry to the file. Only write in plaintext, not markdown."}
    ]
    past_messages = []
    if len(past_messages) > 0:
        past_messages = ['Here are the most recent 10 messages from the last conversation:'] + past_messages[-10:]
    messages = instructions + past_messages
    user = User(messages)
    gpt = GPT(
        messages
    )
    for function in custom_functions:
        gpt.weild(function)

    print("Welcome to the GPT CLI! Type 'exit' to quit.")
    muted = False

    gpt.go()
    #gpt.go()

    while True:
        user_input = user.go()
        if user_input.lower() in ["bye"]:
            print("Goodbye!")
            break
        if "@me" in user_input.lower():
            muted = True
        elif "@mini" in user_input.lower():
            muted = False
        elif "@claude" in user_input.lower():
            muted = False
        elif "@4o" in user_input.lower():
            muted = False
        elif "@bash" in user_input.lower():
            muted = False
        elif "@py" in user_input.lower():
            muted = False
        elif "@writer" in user_input.lower(): # maybe...
            mode = 'writer'
        elif user_input.lower() == "vim":
            file_name = input("Enter the filename to open in Vim: ")
            subprocess.run(["vim", file_name])
        if not muted:
            gpt.go()

if __name__ == "__main__":
    # Check if a file name is provided
    if len(sys.argv) != 2:
        print("Usage: python chat.py <file-to-modify.txt>")
        exit(1)

    # Initialize the file
    initialize_file(sys.argv[1])

    chat()
