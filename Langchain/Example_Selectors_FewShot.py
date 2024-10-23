from langchain_core.example_selectors import LengthBasedExampleSelector, SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import DeepLake
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
# Load llm
llm = ChatOpenAI(model="gpt-3.5-turbo")

#  _____                           _        ____       _           _                 
# | ____|_  ____ _ _ __ ___  _ __ | | ___  / ___|  ___| | ___  ___| |_ ___  _ __ ___ 
# |  _| \ \/ / _` | '_ ` _ \| '_ \| |/ _ \ \___ \ / _ \ |/ _ \/ __| __/ _ \| '__/ __|
# | |___ >  < (_| | | | | | | |_) | |  __/  ___) |  __/ |  __/ (__| || (_) | |  \__ \
# |_____/_/\_\__,_|_| |_| |_| .__/|_|\___| |____/ \___|_|\___|\___|\__\___/|_|  |___/
#                           |_|                                                      

#   ____ 
#  /_ _ |
#   | | |
#   | | |
#   | | |
#  _| | |
# |_____|


# # Example Selectors with Manually written examples

#
# # Define examples
# examples = [
#     {"word": "happy", "antonym": "sad"},
#     {"word": "tall", "antonym": "short"},
#     {"word": "energetic", "antonym": "lethargic"},
#     {"word": "sunny", "antonym": "gloomy"},
#     {"word": "windy", "antonym": "calm"},
# ]
#
# # Define template for example
# example_template = """
# Word: {word}
# Antonym: {antonym}
# """
#
# # Define example prompt
# example_prompt = PromptTemplate(
#     input_variables=["word", "antonym"],
#     template=example_template
# )
#
# # Define the example selector - this will select examples where the length is 25 or less
# example_selector = LengthBasedExampleSelector(
#     examples=examples,
#     example_prompt=example_prompt,
#     max_length=25
# )
#
# # Create FewShotPromptTemplate
# few_shot_prompt = FewShotPromptTemplate(
#     example_selector=example_selector,
#     example_prompt=example_prompt,
#     prefix="Give the antonym of each input",
#     suffix="Word: {input}\nAntonym:",
#     input_variables=["input"]
# )
#
# # Initiate chain
# chain = few_shot_prompt | llm
# response = chain.invoke({"input": "small"})
# print(response.content)

#   ___  
#  |__ \ 
#     ) |
#    / / 
#   / /_ 
#  |____|
             
# Choose examples based on their semantic similarity to the input query w/ few shot method

# Create Prompt Template
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input:{input}\nOutput:{output}"
)
# Define examples
examples = [
    {"input": "0°C", "output": "32°F"},
    {"input": "10°C", "output": "50°F"},
    {"input": "20°C", "output": "68°F"},
    {"input": "30°C", "output": "86°F"},
    {"input": "40°C", "output": "104°F"},
]

# Create Deep Lake dataset
my_activeloop_org_id = os.getenv("ACTIVELOOP_ORG_ID")
my_activeloop_dataset_name = "langchain_fewshot_selector"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path)

# Embedding function
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initiate Semantic Similarity Example Selector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, embeddings, db, k=1
)
# Create a FewShotPromptTemplate
similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Convert the temperature from Celsius to Fahrenheit",
    suffix="Input: {temperature}\nOutput:",
    input_variables=["temperature"]
)
# Test prompt
chain = similar_prompt | llm
response = chain.invoke({"temperature":"10°C"})
response2 = chain.invoke({"temperature":"30°C"})
print(response.content)
print(response2.content)

# Add a new example to the SemanticSimilarityExampleSelector
similar_prompt.example_selector.add_example({"input":"50°C", "output":"122°F"})
response3 = chain.invoke({"temperature":"40°C"})
print(response3.content)