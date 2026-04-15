from dotenv import load_dotenv
load_dotenv()
from rich import print
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

@tool
def greetUser(name: str) -> str:
    """
    This tool greet the user, and takes a string as input and returns a string as well
    """
    return f"Hello, {name}"

tools = {
    "greetUser": greetUser
}

model = init_chat_model("groq:openai/gpt-oss-120b")
model_with_tool = model.bind_tools([greetUser])

# res = model.invoke("Ayush")
# print(res)

res2 = model_with_tool.invoke("Kanha")
if res2.tool_calls:
    tool_call = res2.tool_calls[0]
    tool_name = tool_call['name']
    tool_args = tool_call['args']
    finalRes = tools[tool_name].invoke(tool_args)
    print(finalRes)
    