from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# ---------------- LLM ----------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    temperature = 1,      # VERY LOW for schema safety
    max_new_tokens=120
)

model = ChatHuggingFace(llm=llm)

# ---------------- Schema ----------------
class Person(BaseModel):
    name: str
    age: int = Field(gt=20)
    city: str

parser = PydanticOutputParser(pydantic_object=Person)

# ---------------- Prompt (CRITICAL FIX) ----------------
template = PromptTemplate(
    template="""
Create a fictional {place} person.

Return ONLY a JSON object with EXACTLY these keys:
name, age, city

do not return anything else except json object only nothing else is required

Return ONE flat JSON object only.

{format_instructions}
""",
    input_variables=["place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# ---------------- Chain ----------------
chain = template | model | parser

# ---------------- Invoke ---------------
result = chain.invoke({"place": "Indian"})
print(result)
