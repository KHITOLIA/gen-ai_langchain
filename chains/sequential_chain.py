from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-Llama/Llama-3.2-3B-Instruct",
    task = "text-generation",
    temperature = 0,
    max_new_tokens = 500
    )

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

template1 = PromptTemplate(
    template = "Generate a detailed report on the {topic}",
    input_variables = ['topic']
)

template2  = PromptTemplate(
    template = "Give me the 5 points about the summary from the following {text}",
    input_variables = ['text']
) 

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic' : "Machine learning"})

chain.get_graph().print_ascii()

print(result)