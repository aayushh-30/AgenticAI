from langchain_community.embeddings import HuggingFaceBgeEmbeddings

embedding = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en"
)

print(embedding.embed_query("Ayush"))