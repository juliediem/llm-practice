import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI()
model = "gpt-3.5-turbo"

# # Zero-shot prompting
# # The model is asked to produce output without any examples
# prompt_system = 'You are a helpful assistant whose goal is to write short poems.'
# # Uses template language as a placeholder for topic
# prompt = """Write a short poem about {topic}."""
#
#
#
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role":"system","content":prompt_system},
#         {"role":"user","content":prompt.format(topic="Summer")}
#     ]
# )


# # In-context and Few-Shot Learning
# # In-context learning provides examples for the model to learn. Few-shot is a subset of In-context, where small examples are provided
#
# prompt_system = "You are a helpful assistant whose goal is to write short poems."
# prompt = """Write a short poem about {topic}."""
# examples = {
#     "nature": """Birdsong fills the air,\nMountains high and valleys
# deep,\nNature's music sweet.""",
#     "winter": """Snow blankets the ground,\nSilence is the only
# sound,\nWinter's beauty found."""
# }
#
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": prompt_system},
#         # Customizes prompt to ask for a poem about nature
#         {"role": "user", "content": prompt.format(topic="nature")},
#         # Provides the example response to the prompt asking for nature topic
#         {"role": "assistant", "content": examples["nature"]},
#         # Customizes prompt to ask for a poem about winter
#         {"role": "user", "content": prompt.format(topic="winter")},
#         # Provides an example about winter
#         {"role": "assistant", "content": examples["winter"]},
#         # This prompt is now asking for a poem about summer, with no training examples provided. This poem will be the output
#         {"role": "user", "content": prompt.format(topic="summer")}
#     ]
# )
#
# print(response.choices[0].message.content)

# Few-Shot Prompting Example
# Using the Chat modules instead of the regular one for Chat specific features
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

openai_model = ChatOpenAI()

# These are the example responses
examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"},
]

# This is the template of how the examples will be output to AI
example_formatter_template = """
Color: {color}
Emotion: {emotion}\n
"""

# Human is the input response, and the AI value, is the expected output - in this scenario the emotion associated
# with the color
example_prompt = ChatPromptTemplate(
    [
        ("human","{color}"),
        ("ai","{emotion}"),
    ]
)

# Setup the FewShot Prompt Template with the above examples, and the example prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Now you put together the prompt, with the training examples, and then put in your final question for the AI
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","Here are some examples of colors and the emotions associated with them:"),
        few_shot_prompt,
        ("human","{input}")
    ]
)

# Initialize the chain
chain = final_prompt | openai_model
# You're asking the model what the emotion of Purple would be
response = chain.invoke({"input":"Purple"})
# This would output the response
print(response.content)
