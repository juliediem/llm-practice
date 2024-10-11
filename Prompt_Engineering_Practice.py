from openai_model import OpenAIModel
import openai

model = "gpt-3.5-turbo"
# prompt = "You are a helpful assistant."
# english_text = "Translate the following English text to French: Hello, how are you?"
#
# model_instance = OpenAIModel(prompt, english_text)
# result = model_instance.process_prompt(model)
# print(result)

prompt = """
Describe the following movie using emojis.

{movie}: """

# Few shot learning examples
examples = [
    {"input": "Titanic", "output": "🛳️🌊❤️🧊🎶🔥🚢💔👫💑"},
    {"input": "The Matrix", "output": "🕶️💊💥👾🔮🌃👨🏻‍💻🔁🔓💪"}
]

movie = input("Enter movie:")

client = openai.OpenAI()
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.format(movie=examples[0]["input"])},
        {"role": "assistant", "content": examples[0]["output"]},
        {"role": "user", "content": prompt.format(movie=examples[1]["input"])},
        {"role": "assistant", "content": examples[1]["output"]},
        {"role": "user", "content": prompt.format(movie=movie)},
    ]
)

print(completion.choices[0].message.content)
