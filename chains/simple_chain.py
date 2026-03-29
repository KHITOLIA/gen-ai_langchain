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

template = PromptTemplate(
    template = "Generate 5 facts about the {topic}",
    input_variables = ['topic']
)

parser = StrOutputParser()

chain = template | model | parser                       # | : this is a pipe operator used to make connection between two different components

result = chain.invoke({"topic" : 'Machine learning'})

chain.get_graph().print_ascii()

print(result)

