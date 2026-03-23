from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from dotenv import load_dotenv

load_dotenv()
# model initialisation
model = init_chat_model("groq:openai/gpt-oss-120b")

# document loading
loader = PyPDFLoader("./LHN.pdf")
docs = loader.load()

# splitting the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)
chunks = splitter.split_documents(docs)
# embedding model initialisation
# embeddingModel = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
embeddingModel = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en"
)

# storing into Chroma DB, it will first create the embedding and then store into it.
vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embeddingModel,
    persist_directory = "chroma-db-LHN"
)

# Prompt Template / Master Prompt
basePrompt = ChatPromptTemplate.from_messages([
    ('system','You are a helpful assistant, that will answer my questions on the basis of the context, that is provided by the user.Stick to the context provided and no other respons. If the context is not present please reply as NO CONTEXT FOUND '),
    ('human',"context : {context}, Question : {question}")
])

# Similar Search in the DB for the context
relatedContext = vector_store.similarity_search("Law of Rationality", k = 3)

# Attaching the context to my prompt template
finalPrompt = basePrompt.format_messages(context = relatedContext, question = "What is Law of Rationality ?")

# model calling for response
response = model.invoke(finalPrompt)

print(response.content)






