from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate

# chat tempalate
chat_template = ChatPromptTemplate([
    ('system',  'You are a helpfull customer support agent'),
    MessagesPlaceholder(variable_name = 'chat_history'),
    ('human', '{query}')
]
)
chat_history = []
#load chat history
with open('chat_history.txt', 'r') as f:
    chat_history.extend(f.readlines()) 


# create prompt
chat_template.invoke({'chat_history' : chat_history, 'query' : 'Where is my refund?'})
print(chat_template)