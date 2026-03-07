import time
start_time = time.time()
import ollama

response = ollama.generate(
    model = 'mistral',
    prompt = 'What is machine learning?'
)

print(response.response)
print()
print(f"Execution Time: {time.time() - start_time}")

print()
from langchain_ollama import OllamaLLM

start_time = time.time()

llm = OllamaLLM(model = 'mistral')

response = llm.invoke("What is machine learning?")
print(response)
print(f"Execution Time: {time.time() - start_time}")