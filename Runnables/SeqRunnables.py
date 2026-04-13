from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b")
outputParser = StrOutputParser()
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words."
)

chain = prompt | model | outputParser

res = chain.invoke("Love")
print(res)