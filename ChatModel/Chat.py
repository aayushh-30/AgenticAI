from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model

load_dotenv()


model = init_chat_model("groq:openai/gpt-oss-120b",
temperature = 0.5,max_token = 250, max_retries = 5, timeout = 60)
messages = []
print("-*-*-*-*-*-*-*- Press 0 to exit -*-*-*-*-*-*-*-")
while True:
    prompt = input("You😎 : ")
    messages.append(prompt)
    if prompt == "0" : 
        print(messages)
        break
    response = model.invoke(messages)
    print("BOT🤖 : ",response.content)
    messages.append(response.content)




