from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("GML.pdf")
docs = loader.load()

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap  = 20
# )

splitter = CharacterTextSplitter( 
    chunk_size=400,
    chunk_overlap=10
)

texts = splitter.split_documents(docs)

print(len(texts))