from dotenv import load_dotenv
import os

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b",temperature = 0.5,max_token = 25)

promptTemplate = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English to Odia."),
    ("human", "{input}"),
    ("ai", "The French translation of {input} is {output}")
])

finalPrompt = promptTemplate.invoke({"input":"I love programming."})

print(finalPrompt)

response = model.invoke(finalPrompt)

print(response.content)