from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation',
    max_new_tokens = 1000,
    temperature = 0.5
)

chat_history = [
    SystemMessage(content = "You are a Helpful Assistant in Tech field")
]
model = ChatHuggingFace(llm = llm)
while True:
    user_input = input("You : ")
    chat_history.append(HumanMessage(content = user_input))

    if user_input == 'exit':
        break
    else:
        response = model.invoke(chat_history).content
        chat_history.append(AIMessage(content = response))
        print(f"AI : {response}") 
print(chat_history)