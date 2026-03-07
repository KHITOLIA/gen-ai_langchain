import time 
from langchain_huggingface import HuggingFaceEmbeddings
start_time = time.time()
model = 'BAAI/bge-large-en-v1.5'

embedding_model = HuggingFaceEmbeddings(model = model)
query = "hello how are you?"

print(embedding_model.embed_query(query))
print(f"Execution time: {time.time() - start_time}")