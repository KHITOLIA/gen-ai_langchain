from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel, RunnablePassthrough
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
    template = 'generate 2 jokes on the {topic}',
    input_variables = ['topic']
)

joke_generator_chain = RunnableSequence(prompt1, model, parser)

prompt2 = PromptTemplate(
    template = 'give me the explaination of the following joke {joke},',
    input_variables=['joke']
)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explaination' : RunnableSequence(prompt2, model , parser)
})

chain = RunnableSequence(joke_generator_chain, parallel_chain)
result = chain.invoke({"topic" : "Maths"})
print(result['joke'])
print(" ")
print(result['explaination'])