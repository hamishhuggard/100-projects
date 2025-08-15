import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import json
import re

def date():
    now = datetime.now()
    return '[' + now.strftime("%Y-%m-%d %H:%M") + ']'

# Color codes for Python Terminal Output
COLORS = {
    'gpt5': '\033[94m',   # Light Blue
    'me': '\033[92m',     # Green
    'reset': '\033[0m'    # Reset to default
}

log_file = 'log.chat'  # Log file name

def log_message(message):
    """Log messages to both the console and log.chat file."""
    with open(log_file, 'a') as log:
        clean_message = re.sub(r'\x1b\[\d+m', '', message)
        log.write(clean_message + '\n\n')
    print(message, end='\n\n')


class User:
    def __init__(self, messages):
        self.messages = messages
        self.prefix = f"🙂 {COLORS['me']}me:{COLORS['reset']} "

    def go(self):
        user_input = input(f"{date()} {self.prefix}")
        print()

        with open(log_file, 'a') as log:
            log.write(f"{date()} 🙂 me: {user_input}\n\n")

        colored_input = user_input
        for name, color in COLORS.items():
            if name != 'reset':
                colored_input = colored_input.replace(f"@{name}", f"{color}{name}{COLORS['reset']}")
        self.messages.append({"role": "user", "content": colored_input})
        return user_input

class GPT:
    def __init__(self, messages):
        self.messages = messages
        self.prefix = f"🤖 {COLORS['gpt5']}gpt5:{COLORS['reset']} "

        # Load API key from .env file
        load_dotenv()
        GPT_API_KEY = os.getenv("OPENAI_API_KEY")

        if not GPT_API_KEY:
            log_message("Error: OPENAI_API_KEY is not set in the .env file.")
            exit(1)

        self.client = OpenAI(api_key=GPT_API_KEY)

    def go(self):
        completion = self.client.chat.completions.create(
            model="gpt-5-nano", 
            messages=self.messages
        )
        gpt_reply = completion.choices[0].message.content

        if gpt_reply:
            self.messages.append({"role": "assistant", "content": gpt_reply})
            log_message(f"{date()} {self.prefix}{gpt_reply}")

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
            log_message(f"{date()} {self.prefix}{function_called}()");

            # Call the function and get the response
            response_message = function_to_call(*list(function_args.values()))

            # If the function returns something (like read_content), we can handle it
            if function_called == "read_content":
                self.messages.append({"role": "system", "content": response_message})

        return str(gpt_reply)

def chat():
    while True:
        instructions = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]
        messages = instructions
        user = User(messages)
        gpt = GPT(messages)

        current_date = datetime.now()
        formatted_date = current_date.strftime('%A, %d %B %Y')
        print(f'\n📅 {formatted_date}\n')

        while True:
            user_input = user.go()
            if user_input.lower() in ["bye", "exit", "quit"]:
                return
            if user_input.lower() in ["reset"]:
                break
            
            gpt.go()

if __name__ == "__main__":
    chat()
