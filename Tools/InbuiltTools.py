from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser

model = init_chat_model("groq:openai/gpt-oss-120b")
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
    """
    You will get the news from a source and u have to make it short and crisp, pointwise.
    {latest_news}
    """
)

newsCollector = TavilySearchResults(max_result = 5)

chain = prompt | model | parser

latestNews = newsCollector.run("Latest AI news of March 2026 ")

res = chain.invoke({
    "latest_news":latestNews
})

print(res)

