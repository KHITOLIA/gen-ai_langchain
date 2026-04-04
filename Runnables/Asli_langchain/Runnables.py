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

template = NakliPromptTemplate(
    template = 'write a {length} note on {topic}',
    input_variables = ['length', 'topic']
)

llm = NakliLLM()
parser = NakliStrOutputParser()
        
chain = RunnableConnector([template, llm, parser])
print(chain.invoke({'length' : "short" , 'topic' : 'machine learning'}))