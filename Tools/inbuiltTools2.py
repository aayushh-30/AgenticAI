from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from rich import print

model = init_chat_model("groq:openai/gpt-oss-120b")
parser = StrOutputParser()
search = DuckDuckGoSearchRun()

prompt = ChatPromptTemplate.from_template(
    """You will give me questions on the TCS NQT test that happen on the given date
    {Questions}
    """
)

searchResult = search.invoke("TCS NQT, Questions on 14th April, 2026")
chain = prompt | model | parser
res = chain.invoke({
    "Questions" : searchResult
})

print(res)