from dotenv import load_dotenv
load_dotenv()


from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

model = init_chat_model("groq:openai/gpt-oss-120b")

shortPrompt = PromptTemplate.from_template(
    "Describe {topic} within 50 words."
)

longPrompt = PromptTemplate.from_template(
    "Describe {topic} within 250 words."
)

parser = StrOutputParser()

# single inputs, same input to all chains
# chain = RunnableParallel({
#     "short" : shortPrompt | model | parser,
#     "long" : longPrompt | model | parser
#     })

# res = chain.invoke({
#     "topic" : "Data Science"
# })

# multiple inputs
kidPrompt = PromptTemplate.from_template(
    "Explain {topic} to a kid who's name is {name}."
)

adultPrompt = PromptTemplate.from_template(
    "Explain {topic} to a adult who's name is {name}"
)
chain = RunnableParallel({
    'kid': RunnableLambda(lambda x : x['kid']) | kidPrompt | model | parser,
    "adult": RunnableLambda(lambda x : x['adult']) | kidPrompt | model | parser
})

res = chain.invoke({
    'kid': {
        'topic': 'time',
        'name': 'Riyansh'
    },
    'adult' : {
        'topic': 'time',
        'name': 'Ayush'
    }
})



print(res["kid"])
print('\n')
print(res['adult'])