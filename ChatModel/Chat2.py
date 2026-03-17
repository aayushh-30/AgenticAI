from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b",temperature = 0.5,max_token = 25)
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
]
print("-*-*-*-*-*-*-*- Press 0 to exit -*-*-*-*-*-*-*-")
while True:
    prompt = input("You😎 : ")
    messages.append(HumanMessage(content=prompt)) #messages.append(prompt)
    if prompt == "0" : 
        print(messages)
        break
    response = model.invoke(messages)
    print("BOT🤖 : ",response.content)
    messages.append(AIMessage(content=response.content)) #messages.append(response.content)