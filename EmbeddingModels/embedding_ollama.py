from langchain_ollama import OllamaEmbeddings
embedding_model = OllamaEmbeddings(model = "bge-m3")
query = "Delhi is the capital of india"
documents = [
    "this is my arena",
    "i'll be a multibillionoire super soon",
]
vector = embedding_model.embed_documents(documents)

print(str(vector))