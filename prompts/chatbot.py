import warnings
warnings.filterwarnings("ignore")
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.2-3B-Instruct",
    task = 'text-generation',
    temperature = 0,
    max_new_tokens = 400     
)
model = ChatHuggingFace(llm = llm)


chat_history = [
    SystemMessage(content = "you are a helpful AI assistant"),
]

while True:
    user_input = input("You : ")
    chat_history.append(HumanMessage(content = user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print(f"AI : {result.content}")
print(chat_history)