import openai
from dotenv import load_dotenv

load_dotenv()

prompt_system = "You are a helpful assistant whose goal is to help write stories."
prompt = """Continue the following story. Write no more than 50 words.
Once upon a time, in a world where"""

client = openai.OpenAI()
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt}
    ]
)

print(completion.choices[0].message.content)
