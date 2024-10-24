# 1. Install Required Libraries: Ensure that you have all the
# necessary libraries installed. These include requests, newspaper3k,
# and langchain.
# 2. Scrape Articles: Utilize the requests library to extract the
# content of the targeted news articles from their URLs.
# 3. Extract Titles and Text: Use the newspaper library to parse the
# scraped HTML, extracting the titles and text from the articles.
# 4. Preprocess the Text: Prepare the extracted text for processing
# by ChatGPT (cleaning and preprocessing the texts).
# 5. Generate Summaries: Employ GPT-4 to summarize the
# articles’ text.
# 6. Output the Results: Display the generated summaries alongside
# the original titles, enabling users to understand each article’s
# main points quickly.

# ██╗   ██╗███████╗██████╗ ███████╗██╗ ██████╗ ███╗   ██╗     ██╗
# ██║   ██║██╔════╝██╔══██╗██╔════╝██║██╔═══██╗████╗  ██║    ███║
# ██║   ██║█████╗  ██████╔╝███████╗██║██║   ██║██╔██╗ ██║    ╚██║
# ╚██╗ ██╔╝██╔══╝  ██╔══██╗╚════██║██║██║   ██║██║╚██╗██║     ██║
#  ╚████╔╝ ███████╗██║  ██║███████║██║╚██████╔╝██║ ╚████║     ██║
#   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝     ╚═╝
                                                               
# # Import neccessary packages for this project
# from dotenv import load_dotenv
# from newspaper import Article
# from langchain_core.messages import HumanMessage
# from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI

# # Load environment variables
# load_dotenv()

# # Article URL
# article_url = """https://ca.finance.yahoo.com/news/income-inequality-gap-widens-in-canada-as-wealthiest-20-increase-net-worth-at-fastest-pace-statcan-195402459.html"""

# # Variables for article content
# article_title = None
# article_text = None

# try:
#     # Initialize and download the article
#     article = Article(article_url)
#     article.download()
#     article.parse()

#     # Store article content
#     article_title = article.title
#     article_text = article.text
# except Exception as e:
#     print(f"Error occurred while fetching or parsing article: {e}")

# # Proceed if article content is available
# if article_title and article_text:
#     # Initialize the model
#     llm = ChatOpenAI(model="gpt-3.5-turbo")

#     # Define the prompt
#     prompt_template = """You are an advanced AI assistant that summarizes online
#     articles into bulleted lists.
#     Here's the article you need to summarize.
#     ==================
#     Title: {article_title}
#     {article_text}
#     ==================
#     Now, provide a summarized version of the article in a bulleted list
#     format.
#     """
#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["article_title", "article_text"]
#     )

#     # Generate summary
#     chain = prompt | llm
#     response = chain.invoke({"article_title": article_title, "article_text": article_text})
#     print(response.content)
# else:
#     print("Article content is not available.")

# ██╗   ██╗███████╗██████╗ ███████╗██╗ ██████╗ ███╗   ██╗    ██████╗ 
# ██║   ██║██╔════╝██╔══██╗██╔════╝██║██╔═══██╗████╗  ██║    ╚════██╗
# ██║   ██║█████╗  ██████╔╝███████╗██║██║   ██║██╔██╗ ██║     █████╔╝
# ╚██╗ ██╔╝██╔══╝  ██╔══██╗╚════██║██║██║   ██║██║╚██╗██║    ██╔═══╝ 
#  ╚████╔╝ ███████╗██║  ██║███████║██║╚██████╔╝██║ ╚████║    ███████╗
#   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚══════╝
from dotenv import load_dotenv
from newspaper import Article
import json
import requests
from langchain_openai import ChatOpenAI

load_dotenv()

# Initialize the model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Headers for the request
headers = {
'User-Agent': '''Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'''
}

# This is the URL to an article that I want to summarize
article_url = """https://ca.finance.yahoo.com/news/live/canadians-can-breathe-a-sigh-of-relief-macklem-says-as-bank-of-canada-cuts-interest-rate-by-50-basis-points-192037351.html"""

# This initializes a session to make requests to the article URL
session = requests.Session()

try:
    # This makes a request to the article URL and stores the response
    response  = session.get(article_url, headers=headers, timeout=10)
    # This checks if the response is successful
    if response.status_code == 200:
        # This creates an Article object from the url provided
        article = Article(article_url)
        # This downloads the article content
        article.download()
        # This parses the article content
        article.parse()
        # This prints the title and text of the article
        # print(f"Title: {article.title}")
        # print(f"Text: {article.text}")
except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")

from langchain_core.messages import HumanMessage

# Get the article data from the scraping part
article_title = article.title
article_text = article.text

# # Now we will prep the template for the prompt
# template = """
# As an advanced AI, you've been tasked to summarize online articles into bulleted points. Here are a few examples of how you've done this in the past:
# Example 1:
# Original Article: 'The Effects of Climate Change'
# Summary:
# - Climate change is causing a rise in global temperatures.
# - This leads to melting ice caps and rising sea levels.
# - Resulting in more frequent and severe weather conditions.
# Example 2:
# Original Article: 'The Evolution of Artificial Intelligence'
# Summary:
# - Artificial Intelligence (AI) has developed significantly over the past
# decade.
# - AI is now used in multiple fields such as healthcare, finance, and
# transportation.
# - The future of AI is promising but requires careful regulation.
# Now, here's the article you need to summarize:
# ==================
# Title: {article_title}
# {article_text}
# ==================

# Please provide a summary of the article in a bulleted list format.
# """

# # Format the prompt
# prompt = template.format(
#     article_title=article.title,
#     article_text=article.text
# )

# messages = [HumanMessage(content=prompt)]

# # Generate the summary
# summary = llm.invoke(messages)
# print(summary.content)

# Now we want to add outputparsers to tailor the outputs.
# We want each bullet point to be in a list, so they are processed as a list instead of a string.

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import field_validator, BaseModel, Field
from langchain_core.prompts import PromptTemplate
from typing import List

# Create the Output Parser class
class ArticleSummary(BaseModel):
    title: str = Field(description="The title of the article")
    summary: List[str] = Field(description="A bulleted list of points summarizing the article")

# Validate that the generated summary has at least 3 bullet points
@field_validator('summary')
def has_three_or_more_bullets(cls, list_of_bullets):
    if len(list_of_bullets) < 3:
        raise ValueError("The summary must have at least 3 bullet points.")
    return list_of_bullets

# Set up the output parser
output_parser = PydanticOutputParser(pydantic_object=ArticleSummary)

# Setup the template

template = """
You are an advanced AI assistant that summarizes online articles into bulleted lists.
Here's the article you want to summarize:
==================
Title: {article_title}
{article_text}
==================
{format_instructions}
Please provide a summary of the article in a bulleted list format.
"""

prompt_template = PromptTemplate(
    template=template,
    input_variables=["article_title", "article_text"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

# Now we start the chain
chain = prompt_template | llm | output_parser

# Generate the summary
summary = chain.invoke({"article_title": article_title, "article_text": article_text})
print(summary)
