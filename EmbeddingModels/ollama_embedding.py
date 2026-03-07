import time
from langchain_ollama import OllamaEmbeddings
start_time = time.time()

embedding_model = OllamaEmbeddings(model = 'nomic-embed-text')
query = "Hello how are you?"

print(embedding_model.embed_query(query))
print(f"Execution time: {time.time() - start_time}")