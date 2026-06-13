from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

ollama_embedding_model = OllamaEmbeddings(model = "bge-m3")
hf_embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "AS WE ALL KNOW BEST PLAYER JASPRIT BUMRAH "

# ollama_document_embeddings = ollama_embedding_model.embed_documents(documents)
# ollama_query_embedding = ollama_embedding_model.embed_query(query)

hf_document_embeddings = hf_embedding_model.embed_documents(documents)
hf_query_embedding = hf_embedding_model.embed_query(query)

# print(cosine_similarity([ollama_query_embedding], ollama_document_embeddings)[0])

# so from here i got to know that hf is working more accurately than ollama
index = np.argmax((cosine_similarity([hf_query_embedding], hf_document_embeddings)[0]))
score = cosine_similarity([hf_query_embedding], hf_document_embeddings)[0]
print(query)
print(" ")
print(documents[index])
print(" ")
print(f"Similarity score : {(score[index])*100}%")