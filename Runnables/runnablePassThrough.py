from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

model = init_chat_model("groq:openai/gpt-oss-120b")
parser = StrOutputParser()

codePrompt = ChatPromptTemplate.from_messages([
    ("system","You are a master code generator, and u will give code on the topic provided by the user"),
    ("human","{topic}")
])

explainPrompt = ChatPromptTemplate.from_messages([
    ("system","You are best at explaining code in a straight and simple manner."),
    ('human','explain the code in simple terms : {code}.')
])

# chain = codePrompt | model | parser | explainPrompt | model | parser

# res = chain.invoke({
#     "topic" : "Binary Search"
# })

# print(res)

seq = codePrompt | model | parser

seq2 = RunnableParallel({
    'codeAgent' : RunnablePassthrough(),
    'explainAgent' : explainPrompt | model | parser
})

chain = seq | seq2

res = chain.invoke({
    'topic': "linear search"
})

print(res)