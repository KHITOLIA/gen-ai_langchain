from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import warnings
warnings.filterwarnings("ignore")
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

messages = [
    SystemMessage(content = "You are a Helpful assistant"),
    HumanMessage(content="tell me about langchain ")
]

result = model.invoke(messages)
messages.append(AIMessage(content = result.content))
print(messages)
