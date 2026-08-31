from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

query = "Delhi is the capital of india"
documents = [
    "this is my arena",
    "i'll be a multibillionoire super soon"
]
vector = embedding_model.embed_documents(documents)

print(len(vector))