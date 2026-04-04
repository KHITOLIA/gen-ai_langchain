from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-Llama/Llama-3.2-3B-Instruct",
    task = "text-generation",
    temperature = 0,
    max_new_tokens = 500
    )
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'generate a tweet on the {topic}',
    input_variables =  ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a linkdIn post content about the {topic}',
    input_variables = ['topic']
)
parallel_chain = RunnableParallel({
    'Tweet' : RunnableSequence(prompt1, model, parser),
    'LinkedIn' : RunnableSequence(prompt2, model, parser)
})

print(parallel_chain.invoke({"topic" : "AI"}))