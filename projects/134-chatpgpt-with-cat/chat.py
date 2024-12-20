#!/usr/bin/python3

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env file
load_dotenv()
GPT_API_KEY = os.getenv("OPENAI_API_KEY")

if not GPT_API_KEY:
    print("Error: OPENAI_API_KEY is not set in the .env file.")
    exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=GPT_API_KEY)


def read_file(filename):
    """Reads the contents of a file and returns it as a string."""
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: File '{filename}' not found."
    except Exception as e:
        return f"Error reading file: {str(e)}"

def process_input(user_input):
    """Processes user input and handles file reading if necessary."""
    # Check if input contains '; cat '
    if ";cat " in user_input:
        # Split the input at the last occurrence of ';cat '            
        before_cmd, filename = user_input.rsplit(";cat ", 1)
        file_content = read_file(filename.strip())
        return before_cmd.strip() + "\n" + file_content  # Return combined message
    if "; cat " in user_input:
        # Split the input at the last occurrence of '; cat '            
        before_cmd, filename = user_input.rsplit("; cat ", 1)
        file_content = read_file(filename.strip())
        return before_cmd.strip() + "\n" + file_content  # Return combined message
    return user_input

def chat_with_gpt():
    print("Welcome to the GPT CLI! Type 'exit' to quit.")

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        user_input = input("🧑: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        user_input = process_input(user_input)

        # Add user message to conversation
        messages.append({"role": "user", "content": user_input})

        try:
            # Use the new chat completions method with the updated model
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            # Correct access to the message content in the completion
            assistant_reply = completion.choices[0].message.content
            print(f"🤖: {assistant_reply}")

            # Add the assistant's reply to the conversation history
            messages.append({"role": "assistant", "content": assistant_reply})

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_with_gpt()

