from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader

loader = CSVLoader(file_path="test.csv")
loader2 = TextLoader("test.txt")


docs = loader.load()
docs2 = loader2.load()

# print(docs[0].metadata)
# print(docs[0].metadata['source'])
# print(docs[0].page_content)
# print(docs[1].page_content)

# data_chunks = docs[1:3]
# for chunk in data_chunks:
#     print(chunk.page_content)

print(len(docs2))