#  ██████╗ ██╗   ██╗████████╗██████╗ ██╗   ██╗████████╗    ██████╗  █████╗ ██████╗ ███████╗███████╗██████╗ ███████╗
# ██╔═══██╗██║   ██║╚══██╔══╝██╔══██╗██║   ██║╚══██╔══╝    ██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝
# ██║   ██║██║   ██║   ██║   ██████╔╝██║   ██║   ██║       ██████╔╝███████║██████╔╝███████╗█████╗  ██████╔╝███████╗
# ██║   ██║██║   ██║   ██║   ██╔═══╝ ██║   ██║   ██║       ██╔═══╝ ██╔══██║██╔══██╗╚════██║██╔══╝  ██╔══██╗╚════██║
# ╚██████╔╝╚██████╔╝   ██║   ██║     ╚██████╔╝   ██║       ██║     ██║  ██║██║  ██║███████║███████╗██║  ██║███████║
#  ╚═════╝  ╚═════╝    ╚═╝   ╚═╝      ╚═════╝    ╚═╝       ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝
                                                                                                                 
# This has to do with the output format of language models. Sometimes you want to have a predicatable data strcture.
# We will cover:
# 1. Pydantic Output Parser
# 2. Comma Separated Output Parser
# 3. Output Fixing Parser
# 4. Retry Output Parser


from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Initiate the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# # Single Output Example
# # Define the data structure that you want to parse the output to
# class Suggestions(BaseModel):
#     # The field words is defined as a list of strings. Context is also defined as the description of the field. This is mandatory. 
#     words: List[str] = Field(description="""list of substitute words based on context""")
# # Throw an error in case of receiving a non-string value
# # The decorator field_validator passes the the field 'words' to the function not_start_with_number; The field 'words' is referenced above.
# @field_validator('words')
# # cls represents the class Suggestions. field represents the field words. 
# def not_start_with_number(cls, field):
#     for item in field:
#         if item[0].isnumeric():
#             raise ValueError("The word cannot start with a number.")
#     return field
# # Single Output Example Ends here, comment this block out to run the multiple outputs example


                                                                                                                                                  
# M   M      l  t          l          OOO        t             t          EEEE                    l     
# MM MM      l  t  ii      l         O   O       t             t          E                       l     
# M M M u  u l ttt    ppp  l eee     O   O u  u ttt ppp  u  u ttt  ss     EEE  x x  aa mmmm  ppp  l eee 
# M   M u  u l  t  ii p  p l e e     O   O u  u  t  p  p u  u  t   s      E     x  a a m m m p  p l e e 
# M   M  uuu l  tt ii ppp  l ee       OOO   uuu  tt ppp   uuu  tt ss      EEEE x x aaa m m m ppp  l ee  
#                     p                             p                                        p          
#                     p                             p                                        p          
# # Multiple Outputs Example

# template = """"
# Offer a list of suggestions to substitute the specificed target_word based on the presented context and reasoning for each word.
# {format_instructions}
# target_word={target_word}
# context={context}
# """

# class Suggestions(BaseModel):
#     words: List[str] = Field(description="list of substitute words based on context")
#     reasons: List[str] = Field(description="reasoning for why this word fits the context provided")

# @field_validator('words')
# def not_start_with_number(cls, field):
#     for item in field:
#         if item[0].isnumeric():
#             raise ValueError("The word cannot start with a number.")
#     return field

# @field_validator('reasons')
# def end_with_period(cls, field):
#     for idx, item in enumerate(field):
#         if item[-1] != ".":
#             field[idx] += "."
#     return field
# # Multiple Outputs Example Ends here, comment this block out to run the single output example

# parser = PydanticOutputParser(pydantic_object=Suggestions)

# # Create template
# template = """
# Offer a list of suggestions to substitute the specificed target_word based on the presented context.
# {format_instructions}
# target_word={target_word}
# context={context}
# """

# target_word = "machine"
# context = """
# The machine was designed to automate the production of goods.
# """

# prompt_template = PromptTemplate(
#     template=template, 
#     input_variables=["target_word", "context"], 
#     partial_variables={"format_instructions": parser.get_format_instructions()}
#     )

# # Create a chain
# chain = prompt_template | llm 

# response = chain.invoke({"target_word": target_word, "context": context})
# print(parser.parse(responsecontent))   
 

# .-. .-. .  . .  . .-.   .-. .-. .-. .-. .-. .-. .-. .-. .-.   .-. . . .-. .-. . . .-.   .-. .-. .-. .-. .-. .-. 
# |   | | |\/| |\/| |-|   `-. |-  |-' |-| |(  |-|  |  |-  |  )  | | | |  |  |-' | |  |    |-' |-| |(  `-. |-  |(  
# `-' `-' '  ` '  ` ` '   `-' `-' '   ` ' ' ' ` '  '  `-' `-'   `-' `-'  '  '   `-'  '    '   ` ' ' ' `-' `-' ' ' 
                                                                                 
# # Comma Separated Output Parser
# from langchain_core.output_parsers import CommaSeparatedListOutputParser

# # This parser is not flexible and can only be used for comma separated lists.
# parser = CommaSeparatedListOutputParser()

# # Prepare prompt
# template = """
# Offer a list of suggestions to substitute  the word '{target_word}' based on the following context: {context}. 
# {format_instructions}
# """

# prompt_template = PromptTemplate(
#     template=template,
#     input_variables=["target_word", "context"],
#     partial_variables={"format_instructions": parser.get_format_instructions()}
# )

# # Initiate chain
# chain = prompt_template | llm

# response = chain.invoke({"target_word": "machine", "context": "The machine was designed to automate the production of goods."})

# print(parser.parse(response.content))

 #######                             #######                                    
 #       # #    # # #    #  ####     #       #####  #####   ####  #####   ####  
 #       #  #  #  # ##   # #    #    #       #    # #    # #    # #    # #      
 #####   #   ##   # # #  # #         #####   #    # #    # #    # #    #  ####  
 #       #   ##   # #  # # #  ###    #       #####  #####  #    # #####       # 
 #       #  #  #  # #   ## #    #    #       #   #  #   #  #    # #   #  #    # 
 #       # #    # # #    #  ####     ####### #    # #    #  ####  #    #  ####  
                                                                                
# Fixing Errors
# Output Fixing Parser
# There are limitations to the output parsers that we have covered so far.
# For example, if the LLM returns a misformatted JSON, the Pydantic parser will throw an error.
# This is where the OutputFixingParser comes in. It uses the LLM to fix the misformatted output.   
# There are also limitations to OutputFixingParser. It can only fix syntax errors. It cannot fix logical errors.

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define the data structure that you want to parse the output to
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitute words based on context")
    reasons: List[str] = Field(description="reasoning for why this word fits the context provided")

parser = PydanticOutputParser(pydantic_object=Suggestions)
# This is a misformatted output, where we used "reasoning" instead of "reasons"
misformatted_output = '{"words": ["conduct", "manner"],"reasoning": ["refers to the way someone acts in a particular situation.","refers to the way someone behaves in a particular situation."]}'
# parser.parse(misformatted_output)

# To correctly handle this, this is the output fixing parser
from langchain.output_parsers import OutputFixingParser

outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
print(outputfixing_parser.parse(misformatted_output))


 ######                               #######                                     ######                                     
 #     # ###### ##### #####  #   #    #     # #    # ##### #####  #    # #####    #     #   ##   #####   ####  ###### #####  
 #     # #        #   #    #  # #     #     # #    #   #   #    # #    #   #      #     #  #  #  #    # #      #      #    # 
 ######  #####    #   #    #   #      #     # #    #   #   #    # #    #   #      ######  #    # #    #  ####  #####  #    # 
 #   #   #        #   #####    #      #     # #    #   #   #####  #    #   #      #       ###### #####       # #      #####  
 #    #  #        #   #   #    #      #     # #    #   #   #      #    #   #      #       #    # #   #  #    # #      #   #  
 #     # ######   #   #    #   #      #######  ####    #   #       ####    #      #       #    # #    #  ####  ###### #    # 
                                                                                                                             
# Retry Output Parser

# Define data structure
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitute words based on context")
    reasons: List[str] = Field(description="reasoning for why this word fits the context provided")

parser = PydanticOutputParser(pydantic_object=Suggestions)

# Define prompt
template = """
Offer a list of suggestions to substitute the specificed target_word based on the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt_template = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt_template.format_prompt(target_word="behaviour", context="""The behaviour of the students in the classroom was disruptive
and made it difficult for the teacher to conduct the lesson.""")

from langchain.output_parsers import RetryWithErrorOutputParser

missformatted_output = '{"words": ["conduct", "manner"]}'

retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=llm)

retry_parser.parse_with_prompt(misformatted_output, model_input)