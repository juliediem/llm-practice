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


# Import neccessary packages for this project
from dotenv import load_dotenv
from newspaper import Article
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Article URL
article_url = """https://ca.finance.yahoo.com/news/income-inequality-gap-widens-in-canada-as-wealthiest-20-increase-net-worth-at-fastest-pace-statcan-195402459.html"""

# Variables for article content
article_title = None
article_text = None

try:
    # Initialize and download the article
    article = Article(article_url)
    article.download()
    article.parse()

    # Store article content
    article_title = article.title
    article_text = article.text
except Exception as e:
    print(f"Error occurred while fetching or parsing article: {e}")

# Proceed if article content is available
if article_title and article_text:
    # Initialize the model
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Define the prompt
    prompt_template = """You are a very good assistant that summarizes online articles.
    Here's the article you want to summarize.
    ==================
    Title: {article_title}
    {article_text}
    ==================
    Write a summary of the previous article.
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["article_title", "article_text"]
    )

    # Generate summary
    chain = prompt | llm
    response = chain.invoke({"article_title": article_title, "article_text": article_text})
    print(response.content)
else:
    print("Article content is not available.")

