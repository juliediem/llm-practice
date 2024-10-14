import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# Initiate model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Load your pdf file for summarization
loader = PyPDFLoader(
    "Data/The One Page Linux Manual.pdf",
)
docs = loader.load()

# Prompt template to summarize where {context} will be the placeholder for the document that you want to summarize
prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\\n\\n{context}")]
)

# Instantiate Chain
chain = create_stuff_documents_chain(llm, prompt)
# Adding the context as the documents you loaded
result = chain.invoke({"context": docs})
print(result)
