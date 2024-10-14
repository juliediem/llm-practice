from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Grab environment variable
load_dotenv()

# Initialize model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create prompts
prompt = PromptTemplate(
    template="Question: {question}\nAnswer:",
    input_variables=["question"]
)

# Initialize chain
# The order of the chain MATTERS! You need to create the prompt with it's variables first, and then you pass it through the LLM
chain = prompt | llm

# Run chain
response = chain.invoke({"question": "What is the meaning of life?"})
print(response.content)
