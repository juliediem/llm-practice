# Create a completion using GPT-4
from openai import OpenAI
# Loading the environment variables so you don't expose your API key
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

system_message = """
You are an AI assistant specialized in solving riddles.
Given a riddle, solve it the best you can.
Provide a clear justification of your answer and the reasoning behind it.
Riddle:
"""

article = """
What has a face and two hands, but no arms or legs?
"""


completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": article
        }
    ]
)

# This extracts only the text output
print(completion.choices[0].message.content)
