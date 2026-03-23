from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()


basePrompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful, bot, that answers my questions, in a precise manner, not very long or not very short"),
    ("human","{story}"),
    ("human","{question}")
])

loader = TextLoader("ram.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20
)

chunks = splitter.split_documents(docs)

model = init_chat_model("groq:openai/gpt-oss-120b")
# finalPrompt = basePrompt.format_messages(story = docs[0].page_content,question = "Who is Ram?")
embeddingModel = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddingModel,
    persist_directory="chroma-db"
    )


#vector_store.add_documents(chunks)
# response = model.invoke(finalPrompt)
# print(response.content)

res = vector_store.similarity_search("Ram's father",k = 2)
print(res[0].page_content)