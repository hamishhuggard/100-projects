import os
from dotenv import load_dotenv
from openai import OpenAI

class User:
    def __init__(self, messages):
        self.messages = messages

    def go(self):
        user_input = input("🙂: ")
        self.messages.append({"role": "user", "content": user_input})
        return user_input

class GPT:
    def __init__(self, messages):
        self.messages = messages

        # Load API key from .env file
        load_dotenv()
        GPT_API_KEY = os.getenv("OPENAI_API_KEY")

        if not GPT_API_KEY:
            print("Error: OPENAI_API_KEY is not set in the .env file.")
            exit(1)

        self.client = OpenAI(api_key=GPT_API_KEY)

    def go(self):
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.messages
            )
            gpt_reply = completion.choices[0].message.content
            self.messages.append({"role": "assistant", "content": gpt_reply})
            print(f"🤖: {gpt_reply}")
            return gpt_reply
        except Exception as e:
            print(f"Error while contacting GPT: {e}")
            return None

def chat():
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    user = User(messages)
    gpt = GPT(messages)

    print("Welcome to the GPT CLI! Type 'exit' to quit.")
    muted = False
    while True:
        user_input = user.go()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if user_input.lower() in ["mute"]:
            muted = True
        if user_input.lower() in ["unmute"]:
            muted = False
        if not muted:
            gpt.go()

if __name__ == "__main__":
    chat()
