from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-Llama/Llama-3.2-3B-Instruct",
    task = "text-generation",
    temperature = 0,
    max_new_tokens = 1000
    )
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'write a joke about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'explain the meaning of {joke}',
    input_variables = ['joke']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
print(chain.invoke({"topic" : 'Machine learning'}))