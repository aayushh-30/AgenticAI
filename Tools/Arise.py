from dotenv import load_dotenv
load_dotenv()
from rich import print
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def countCharacters(sentence: str) -> int:
    """
    This tool counts the number of characters in a string , it acceps a string as input and returns an integer
    """
    return len(sentence)

tools = {
    "countCharacters": countCharacters
}

history = []

message = HumanMessage(content="Calculate the length of the sentence, 'My name is Ayush.'")

history.append(message)

model = init_chat_model("groq:openai/gpt-oss-120b")
model_with_tool = model.bind_tools([countCharacters])

res = model_with_tool.invoke(message.content)
history.append(res)

if res.tool_calls:
    tool_name = res.tool_calls[0]['name']
    tool_args = res.tool_calls[0]['args']
    tool_res = tools[tool_name].invoke(res.tool_calls[0])
    history.append(tool_res)
    
finalRes = model_with_tool.invoke(history)
print(finalRes.content)