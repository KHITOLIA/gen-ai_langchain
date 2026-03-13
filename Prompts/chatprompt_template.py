from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation',
    max_new_tokens = 1000,
    temperature = 0.5
)
model = ChatHuggingFace(llm = llm)

chat_template = ChatPromptTemplate([
    ('system', 'you are an {domain} expert'),
    ('human', 'Explain in simple terms what is {topic}')
])

prompt = chat_template.invoke({"domain" : "AI-ML", "topic" : "Machine learning and Deep learning"})
print(prompt)
