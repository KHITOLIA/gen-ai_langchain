from flask import Flask, render_template, request, redirect
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation',
    max_new_tokens = 1000,
    temperature = 0.5
)
model = ChatHuggingFace(llm = llm)

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['you']
        response = model.invoke(user_input).content
        return render_template('home.html', response = response)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug = True)