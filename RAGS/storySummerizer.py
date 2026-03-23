from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader

model = init_chat_model("groq:openai/gpt-oss-120b")
loader = TextLoader("test.txt")

docs = loader.load()
basePrompt = ChatPromptTemplate.from_messages([
    ('system','You are a helpuful assistant, that has mastery on summerizing the stories'),
    ('human',"{story}")
])

finalPrompt = basePrompt.format_messages(story = docs[0].page_content)

response = model.invoke(finalPrompt)

print(response.content)

