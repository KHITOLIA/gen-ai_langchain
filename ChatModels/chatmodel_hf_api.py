from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

# Initialize the chat model
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",  # Hugging Face model repo
    task = "text-generation",
    temperature = 0,
    max_new_tokens= 1000,
)

model = ChatHuggingFace(llm = llm)
response = model.invoke("who started anglo india moment?")
print(response.content)