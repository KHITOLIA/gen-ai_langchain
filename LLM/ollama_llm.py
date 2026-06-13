#Simple ollama implementation
import ollama

response = ollama.generate(
    model = 'mistral',
    prompt = "explain machine learningngg in 2 lines"
)
print(response['response'])

# Ollama with langchain implementation
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model = "mistral")
response = llm.invoke("explain machine learning in 2 lines")
print(response)