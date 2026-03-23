from langchain_community.document_loaders import WebBaseLoader

url = "https://www.cricbuzz.com/live-cricket-scores/122720/nz-vs-rsa-4th-t20i-south-africa-tour-of-new-zealand-2026"

loader = WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content)