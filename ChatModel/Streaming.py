from dotenv import load_dotenv
import os

load_dotenv()

from langchain.chat_models import init_chat_model


model = init_chat_model(
    "groq:openai/gpt-oss-120b",
    temperature = 0.9,
)

message = [

]

question = input("You : ")

for chunk in model.stream(question):
    print(chunk.content, end="", flush=True)