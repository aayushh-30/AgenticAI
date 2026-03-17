from dotenv import load_dotenv
import os

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model("groq:openai/gpt-oss-120b")

messages = [
    "Who is the President of India",
    "Weather of Mumbai",
    "Why is the sky blue"
]

response = model.batch(messages,config={
    "max_concurrency": 5
})

for res in response:
    print(res.content)