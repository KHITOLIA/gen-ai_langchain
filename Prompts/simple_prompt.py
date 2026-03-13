from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation',
    max_new_tokens = 1000,
    temperature = 0.5
)
model = ChatHuggingFace(llm = llm)

input_prompt = input("Enter your query : ")

response = model.invoke(input_prompt)
print(response.content)