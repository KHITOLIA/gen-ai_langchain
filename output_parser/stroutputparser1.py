from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id = "meta-Llama/Llama-3.2-3B-Instruct",
    task = "text-generation",
    temperature = 0,
    max_new_tokens = 500
    )
model = ChatHuggingFace(llm = llm)

# template1 : detailed report
template1 = PromptTemplate(
    template = "Give me a detailed report about the {topic}",
    input_variables = ['topic']
)

# template2 : Summary of the generated report
template2 = PromptTemplate(
    template = "Summarize the following \n{text}",
    input_variables = ['text']
)

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic" : "Logistic Regression"})
print(result)