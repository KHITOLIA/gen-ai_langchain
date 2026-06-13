from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import warnings
warnings.filterwarnings("ignore")

chat_template = ChatPromptTemplate.from_messages([
    ('system' , 'you are a helpful {domain} expert'),
    ('human' , 'Explain in simple terms, what is {topic}'),
])

prompt = chat_template.invoke({'topic' : 'SVM', 'domain' : 'Machine learning'})
print(prompt)