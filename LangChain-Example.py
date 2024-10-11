# # Basic LangChain prompt with OpenAI
# from langchain_openai import OpenAI

# This loads the OPENAI_API_KEY from .env file
from dotenv import load_dotenv
load_dotenv()

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


# ▒█▀▀█ █▀▀█ █▀▀ █▀▀█ ▀▀█▀▀ ░▀░ █▀▀▄ █▀▀▀ 　 █▀▀ █▀▀█ █▀▀▄ ▀▀█▀▀ █▀▀ █░█ ▀▀█▀▀ 　 █▀▀ █▀▀█ █▀▀█
# ▒█░░░ █▄▄▀ █▀▀ █▄▄█ ░░█░░ ▀█▀ █░░█ █░▀█ 　 █░░ █░░█ █░░█ ░░█░░ █▀▀ ▄▀▄ ░░█░░ 　 █▀▀ █░░█ █▄▄▀
# ▒█▄▄█ ▀░▀▀ ▀▀▀ ▀░░▀ ░░▀░░ ▀▀▀ ▀░░▀ ▀▀▀▀ 　 ▀▀▀ ▀▀▀▀ ▀░░▀ ░░▀░░ ▀▀▀ ▀░▀ ░░▀░░ 　 ▀░░ ▀▀▀▀ ▀░▀▀
#
# ▒█▀▄▀█ █▀▀█ █▀▀ █░░█ ░▀░ █▀▀▄ █▀▀ █▀▀
# ▒█▒█▒█ █▄▄█ █░░ █▀▀█ ▀█▀ █░░█ █▀▀ ▀▀█
# ▒█░░▒█ ▀░░▀ ▀▀▀ ▀░░▀ ▀▀▀ ▀░░▀ ▀▀▀ ▀▀▀
# Creating Context for Machines
# Key Concept - Embeddings
# Embeddings are a way for AI to create context/meaning/relationships between words; These are stored in a vector
# Step 1: To prepare the Embeddings (vector), you start by splitting the text in your document
# How to: Splitting text in Documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open(file="Data/mountain.txt") as f:
    mountain_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Chunk size represents number of characters chunk size is to help break down the content to be ingested by AI
    # models; I think OpenAI has a limit of ~8000 characters, I think best practice is to configure this at 1000-1500
    # to avoid performance issues
    chunk_size=100,
    # Number of characters that overlap over the current chunk and the next This maintains context for the
    # relationship between words; Think of it like you're developing a film strip and you need to know a little bit
    # about the previous scene to understand the context of the next scene. Some sources say best practice is to use
    # 10-20% or 20-30% for more complex relationships.
    chunk_overlap=10,
    # This is the function used to measure the number of characters
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([mountain_text])
# Here you can output each chunk
print(texts[0].page_content)
print(texts[1].page_content)
print(texts[2].page_content)

# Step 2 Embeddings: Creating the vector
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)

# Here you are going through each chunk in your mountain text document and adding them to the vector
embeddings = embeddings_model.embed_documents(
    [text.page_content for text in texts]
)

# This outputs information about the vector
print("Embed documents:")
print(f"Number of vector: {len(embeddings)}; Dimension of each vector: {len(embeddings[0])}")
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
print("Embed query:")
print(f"Dimension of the vector: {len(embedded_query)}")
print(f"Sample of the first 5 elements of the vector: {embedded_query[:5]}")

# Step 3: Storing your Vectors
# Vector stores, well, store context on vectors. This can be used by the model to find a
# plot point that is closest to the query they received to return an appropriate answer. Think of this as how each
# vector is a plot point on a vector space, and you're looking for a plot point closest to your query's plot point.
# Your query also gets converted into a plot point on the vector by the way.


