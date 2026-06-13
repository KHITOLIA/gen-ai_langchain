from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

# Initialize the chat model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",  # Hugging Face model repo
    task = "text-generation",
    temperature = 0,
    max_new_tokens= 1000,
)

model = ChatHuggingFace(llm = llm)
response = model.invoke("Explain ANN in simple manner?")
print(response.content)
