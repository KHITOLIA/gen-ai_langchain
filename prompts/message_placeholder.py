from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

# chat template
chat_template = ChatPromptTemplate([
    ('system' , 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name = 'chat_history'),
    ('human' , '{query}')
])

# load chat history
chat_history = []

with open('chat_history.txt', 'r') as f:
    chat_history.extend(f.readlines())

# create prompt 

prompt = chat_template.invoke({'query' : 'what is the position of my refund?', 'chat_history' : chat_history})
print(prompt)   