from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b")

@tool('web_search', description = "Useful for when you need to answer questions about current events. You should ask targeted questions.")
def web_search(query: str) -> str:
    
    return f"Search result for {query}"



@tool
def getWeatherTool(city:str) -> str:
    """Finds the weather of the city"""
    return f"Weather is sunny at {city}"

@tool
def Add(a,b : int) -> int:
    """Finds sum of two number"""
    return a + b

@tool
def getAdminDetails()->str:
    """This will return the admin details"""
    return "Ayush, age : 23"

model_with_tool = model.bind_tools([getWeatherTool,Add,getAdminDetails,web_search])

messages = [{"role": "user", "content": "What's the weather in Boston? and tell me about the admin"}]
ai_response = model_with_tool.invoke(messages)
messages.append(ai_response)

# print(ai_response)

# Map tool names to tool functions
tools_map = {
    'getWeatherTool': getWeatherTool,
    'Add': Add,
    'getAdminDetails': getAdminDetails
}

for tool_call in ai_response.tool_calls:
    tool_name = tool_call['name']
    tool_args = tool_call['args']
    
    if tool_name in tools_map:
        tool = tools_map[tool_name]
        res = tool.invoke(tool_args)
        messages.append({"role": "tool", "content": str(res), "tool_call_id": tool_call['id']})

final_res = model_with_tool.invoke(messages)
print(final_res)