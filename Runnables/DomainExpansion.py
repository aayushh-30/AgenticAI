from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.messages import SystemMessage

model = init_chat_model("groq:openai/gpt-oss-120b")

shortPrompt = PromptTemplate.from_template(
    "You are a DSA expert, u will provide details about the {algorithm_name} algorithm, in short like only it's TC , SC, tradeoffs."
)

longPrompt = PromptTemplate.from_template(
    "You are a DSA expert, u will provide details about the {algorithm_name} algorithm, in depth like only it's history, data structures it can be used, real life implementation, TC , SC, tradeoffs."
)

parser = StrOutputParser()

# Serial Execution : 
# Each prompt is called one by one, and the chains are invoked one by one.

# chain1 = shortPrompt | model | parser
# chain2 = longPrompt | model | parser

# response1 = chain1.invoke({"algorithm_name" : "Binary Search"})
# response2 = chain2.invoke({'algorithm_name' : "Linear Search"})

# print(response1)
# print("\n")
# print(response2)

# Parallel Execution : 

chain = RunnableParallel({
    'shortResponse' : RunnableLambda(lambda x: x["shortTopic"]) | shortPrompt | model | parser,
    'detailedResponse' : RunnableLambda(lambda x: x["detailedTopic"]) | longPrompt | model | parser
})

response = chain.invoke({
    'shortTopic' : {
        "algorithm_name":"Hashing"
    },
    'detailedTopic' : {
        "algorithm_name" : "Slow Fast Pointer"
    }
})

print(response)