# chain_1 : generate a joke
# chain_2 : explain the joke
# and then connect both the chain for continue output

from abc import ABC, abstractmethod
import random

class Runnable(ABC):
    @abstractmethod
    def invoke(input_dict):
        pass

class NakliLLM(Runnable):

    def __init__(self):
        print("LLM created")
    
    def predict(self, prompt):
        print("This method has been deprecated")
    def invoke(self, prompt):
        response_list = [
            'Delhi is the capital of india',
            'AI stands for artificial intelligence',
            'Microsoft AI-102 certification'
        ]
        return {'response' : random.choice(response_list)}

class NakliPromptTemplate:
    def __init__(self,  template, input_variables):
       self.template = template
       self.input_variables = input_variables  

    def invoke(self, input_dict):
        return self.template.format(**input_dict)
    
    def format(self, input_dict):
        print("this method has been deprecated")

class RunnableConnector(Runnable):
    def __init__(self,runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):

        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data
    
class NakliStrOutputParser(Runnable):
    def __init__(self):
        pass
    def invoke(self, input_data):
        return input_data['response']

template_1 = NakliPromptTemplate(
    template = 'write a joke about {topic}',
    input_variables = ['topic']
)

template_2 = NakliPromptTemplate(
    template = 'explain the following joke {response}',
    input_variables = ['response']
)

llm = NakliLLM()
parser = NakliStrOutputParser()

chain1 = RunnableConnector([template_1, llm])
chain2 = RunnableConnector([template_2, llm])

chain = RunnableConnector([chain1, chain2])
print(chain.invoke({"topic" : "Noida"}))
