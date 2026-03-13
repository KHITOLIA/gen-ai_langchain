from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation',
    max_new_tokens = 1000,
    temperature = 0.5
)
model = ChatHuggingFace(llm = llm)

template = '''
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  
1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
'''

template = PromptTemplate(
    template = template,
    input_variables = ['paper_input', 'style_input', 'length_input']
)

# template.save('template.json')

prompt = template.invoke({'paper_input' : input("enter the Research Paper Name: "), "style_input" : input("enter the style : "), "length_input" : input("Enter the length :")})
response = model.invoke(prompt)
print(response.content)