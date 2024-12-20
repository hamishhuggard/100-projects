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

# Define custom file operations functions
def write_content(content):
    """Writes the entire content to the 'content.txt' file (overwrites existing content)."""
    with open('content.txt', 'w') as file:
        file.write(content)
    print("Writing to content.txt:")
    print(content)

def append_content(content):
    """Appends content to the 'content.txt' file."""
    with open('content.txt', 'a') as file:
        file.write(content + '\n')
    print("Appending to content.txt:")

def read_content():
    """Reads the entire content of the 'content.txt' file and returns it."""
    if os.path.exists('content.txt'):
        with open('content.txt', 'r') as file:
            content = file.read()
        print("Reading from content.txt:")
        print(content)
        return content
    else:
        print("content.txt does not exist.")
        return ""

# Define custom functions for interaction with the API
custom_functions = [
    {
        'name': 'write_content',
        'description': 'Write content to "content.txt", overwriting existing content.',
        'parameters': {
            'type': 'object',
            'properties': {
                'content': {'type': 'string', 'description': 'The content to write to the file.'}
            }
        }
    },
    {
        'name': 'append_content',
        'description': 'Append content to "content.txt".',
        'parameters': {
            'type': 'object',
            'properties': {
                'content': {'type': 'string', 'description': 'The content to append to the file.'}
            }
        }
    },
    {
        'name': 'read_content',
        'description': 'Read the entire content of "content.txt".',
        'parameters': {
            'type': 'object',
            'properties': {}
        }
    }
]

def chat_with_gpt():
    print("Welcome to the GPT CLI! Type 'exit' to quit.")

    messages = [{"role": "system", "content": "You are a ghost writer. Your job is to ask what the user what their creative vision is for what they want to write, ask them insightful questions to draw out what they have to say, and then write their content for them in their voice but more polished (unless they want it in a different voice)."}]

    while True:
        user_input = input("🧑: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Add user message to conversation
        messages.append({"role": "user", "content": user_input})

        try:
            # Use the new chat completions method with the updated model
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                functions=custom_functions,
                function_call='auto'
            )

            # Correct access to the message content in the completion
            assistant_reply = completion.choices[0].message.content
            print(f"🤖: {assistant_reply}")

            response_message = completion.choices[0].message

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

                # Call the function and get the response
                response_message = function_to_call(*list(function_args.values()))

                # If the function returns something (like read_content), we can handle it
                if function_called == "read_content":
                    # If needed, you can add the returned content to the chat message.
                    messages.append({"role": "assistant", "content": response_message})

            # Add the assistant's reply to the conversation history
            if assistant_reply:
                messages.append({"role": "assistant", "content": assistant_reply})

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_with_gpt()
