from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()


llm = HuggingFaceEndpoint(
    repo_id = "meta-Llama/Llama-3.2-3B-Instruct",
    task = "text-generation",
    temperature = 0,
    max_new_tokens = 500
    )

model = ChatHuggingFace(llm = llm)

# template1
template1 = PromptTemplate(
    template = "Give me the name, age, city of a fictional person \n {format_instruction}",
    input_variables = [],
    partial_variables = {"format_instruction" : parser.get_format_instructions()}
)

chain = template1 | model | parser

result = chain.invoke({})
print(result)
