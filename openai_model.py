# Setting up a reusable module to make setting up easy
from dotenv import load_dotenv
import openai

load_dotenv()


# Create a new object
class OpenAIModel:
    # These are the object variables you will initialize the object with
    def __init__(self, sys_msg, user_input):
        self.sys_msg = sys_msg
        self.user_input = user_input
        self.client = openai

    def process_prompt(self, model_name):
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": self.sys_msg
                },
                {
                    "role": "user",
                    "content": self.user_input
                }
            ]
        )
        return completion.choices[0].message.content
