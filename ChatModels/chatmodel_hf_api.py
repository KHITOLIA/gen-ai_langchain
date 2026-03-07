from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation',
    max_new_tokens = 100,
    temperature = 0.5
)

chatmodel = ChatHuggingFace(llm = llm)
response = chatmodel.invoke("What is Machine learning?")
print(response.content)