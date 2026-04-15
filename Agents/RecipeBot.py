from dotenv import load_dotenv
load_dotenv()

import os
import requests
from rich.console import Console
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

console = Console()

@tool
def getRequiredIngredients(dish: str) -> list :
    """This will return the list of ingredients in order to make the specific dish."""

    url = f"https://api.api-ninjas.com/v3/recipe?title={dish}"
    headers = {
        "X-Api-Key": os.getenv("FOOD_NINJA_API")
    }
    response = requests.get(url=url,headers=headers)
    response = response.json()
    data = response[0]
    ingredients = []
    for ingredient in data["ingredients"]:
        ingredients.append(ingredient)
    return ingredients

available_tools = {
    "getRequiredIngredients": getRequiredIngredients
}

model = init_chat_model("groq:openai/gpt-oss-120b")
model_with_tool = model.bind_tools([getRequiredIngredients])

# res = getRequiredIngredients.invoke("butter chicken")
# print(res)
messages = []
while True:
    user_input = console.input("[blue]User[/blue] : ")
    messages.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    model_res = model_with_tool.invoke(messages)
    messages.append(model_res)

    if model_res.tool_calls:
        for tool_call in model_res.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            confirmation = console.input(f"AI wants the permission to use {tool_name} tool (Y/N) : ")
            if confirmation.lower() == 'n':
                console.print("❌ Access Denied!", style="red")
            
            res = available_tools[tool_name].invoke(tool_call)
            messages.append(res)

            final_response = model_with_tool.invoke(messages)
            messages.append(final_response)
            console.print("[cyan]AI[/cyan] : " + final_response.content, style="cyan")

    else:
        console.print("[cyan]AI[/cyan] : " + model_res.content, style="cyan")

