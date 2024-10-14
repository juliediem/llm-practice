#Import relevant packages
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.chains.combine_documents import create_stuff_documents_chain

yt_transcript = YouTubeTranscriptApi.get_transcript("P1ALkQMfkjc", languages=['en'])
formatter = TextFormatter()
txt_formatted = formatter.format_transcript(yt_transcript)

with open(r'Data\yt_transcript.txt', 'w', encoding='utf-8') as txt_file:
    txt_file.write(txt_formatted)

# Revisit this when you learn more about Embeddings - you need to break the content down so that it's ingestible by the model
# Load environment variables
load_dotenv()

# Initiate LLM model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Load your pdf file for summarization
loader = TextLoader(
    "Data/yt_transcript.txt",
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
