from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch
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
def word_counter(text):
    return len(text.split())

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'generate a detailed report on the {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'generate a summary of the report {report}',
    input_variables = ['report']
)

report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 300, prompt2 | model | parser),
    RunnablePassthrough()
)

chain = report_gen_chain | branch_chain
result = chain.invoke({"Recently US attacked on Venezuela"})
print(result)