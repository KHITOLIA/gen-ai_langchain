from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from warnings import filterwarnings
filterwarnings("ignore")

model = ChatOllama(model = "mistral",
                   temperature=0.3, max_tokens = 100)
result = model.invoke("What is the capital of india?")
print(" ")
print(result.content)