from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
# ---------------- LLM ----------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    temperature=0.1,      # VERY LOW for schema safety
    max_new_tokens=1000
)

model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description = "Give the sentiment of the feedback")


str_parser = StrOutputParser()

pydantic_parser = PydanticOutputParser(pydantic_object = Feedback)


prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback into positive and negative \n {feedback} \t
Return ONLY a JSON object sentiment
do not return anything else except json object only nothing else is required

Return ONE flat JSON object only.

{format_instructions}
""",
    input_variables=["place"],
    partial_variables={
        "format_instructions": pydantic_parser.get_format_instructions()
    }
)

classfier_chain = prompt1 | model | pydantic_parser

prompt2 = PromptTemplate(
    template = ''' Write an appropriate one line response to this positive feedback \n {feedback}''',
    input_variables = ['feedback'], 
)

prompt3 = PromptTemplate(
    template = ''' Write an appropriate one line response to this negative feedback \n {feedback}''',
    input_variables = ['feedback'], 
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive' , prompt2 | model | str_parser ),
    (lambda x:x.sentiment == 'negative', prompt3 | model | str_parser),
    RunnableLambda(lambda x:"Could not find the sentiment of the feedback")
)

chain = classfier_chain | branch_chain
feedback = '''I’ve been using the iQOO Z10x for a while now, and I’m extremely satisfied with the performance. The phone is super fast thanks to the powerful processor, and multitasking feels very smooth. Apps open instantly and gaming performance is impressive for this price range.

The battery life is one of the biggest highlights. It easily lasts a full day with heavy use, and the fast charging is a big bonus. The display quality is sharp, bright, and feels great for watching videos or browsing.

The camera performance is good for daily photos and videos, especially in daylight. The design also looks premium and feels comfortable in hand.

Overall, the iQOO Z10x is a fantastic option if you want strong performance, long battery life, and a smooth experience without spending too much. Definitely worth the money!'''
result = chain.invoke({"feedback" : feedback})
print(result)