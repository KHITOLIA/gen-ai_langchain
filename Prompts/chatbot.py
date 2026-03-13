from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation',
    max_new_tokens = 1000,
    temperature = 0.5
)

chat_history = []
model = ChatHuggingFace(llm = llm)
while True:
    user_input = input("You : ")
    chat_history.append(user_input)
    if user_input == 'exit':
        break
    else:
        response = model.invoke(chat_history).content
        chat_history.append(response)
        print(f"AI : {response}") 