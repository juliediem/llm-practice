# # Basic LangChain prompt with OpenAI
# from langchain_openai import OpenAI
# import os
# from dotenv import load_dotenv
#
# load_dotenv()
#
# llm = OpenAI()
# print(llm.invoke("Tell me a joke"))
# # print(llm('tell me a joke'))


# # LangChain Prompting Templates
# from langchain_core.prompts import PromptTemplate
#
# # Example from documentation
# # Instantiation using from_template (recommended)
# # prompt = PromptTemplate.from_template("Say {foo}")
# ## This output says "Say bar"
# # prompt.format(foo="bar")
#
#
# # In this example you will be translating a sentence to another language
# template = """Sentence: {sentence}
# Translation in {language}
# """
# prompt = PromptTemplate.from_template(template)
#
# print(prompt.format(sentence="The cat is on the table", language="Spanish"))


# # Document Loaders
# from langchain_community.document_loaders.csv_loader import CSVLoader
# loader = CSVLoader(file_path="Data/sample.csv")
# data = loader.load()
# print(data)

# # Splitting text in Documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open(file="Data/mountain.txt") as f:
    mountain_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Chunk size represents number of characters
    chunk_size=100,
    # Number of characters that overlap over the current chunk and the next
    chunk_overlap=20,
    # This is the function used to measure the number of characters
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([mountain_text])
print(texts[0])
print(texts[1])
print(texts[2])