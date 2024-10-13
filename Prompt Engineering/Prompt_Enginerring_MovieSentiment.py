# Create a completion using GPT-4
from openai import OpenAI
# Loading the environment variables so you don't expose your API key
from dotenv import load_dotenv
import os
# Testing Movie Sentiment
import numpy as np
import pandas as pd

load_dotenv()

# Reading the data file with movies
df = pd.read_csv('../Data/movie.csv', encoding='utf-8')
# Redoing the labels so that instead of boolean values, there's a sentiment
df['label'] = df['label'].replace({0: 'Negative', 1: 'Positive'})

df = df.sample(n=10, random_state=42)

client = OpenAI()

system_message = """
You are a binary classifier for sentiment analysis.
Given a text, based on its sentiment, you classify it into one of two categories: positive or negative.
You can use the following texts as examples:
Text: "I love this product! It's fantastic and works perfectly."
Positive
Text: "I'm really disappointed with the quality of the food."
Negative
Text: "This is the best day of my life!"
Positive
Text: "I can't stand the noise in this restaurant."
Negative
ONLY return the sentiment as output (without punctuation).
Text:
"""

article = 'Elevation Embrace'


def process_text(text):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": text
            }
        ]
    )
    return completion.choices[0].message.content


df['predicted'] = df['text'].apply(process_text)
print(df)
# # This extracts only the text output
# print(completion.choices[0].message.content)
