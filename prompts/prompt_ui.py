from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
load_dotenv()

# Initialize the chat model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",  # Hugging Face model repo
    task = "text-generation",
    temperature = 0,
    max_new_tokens= 5000,
)         
model = ChatHuggingFace(llm = llm)

st.header("Research Tool")

paper_input = st.selectbox("Select Research paper name ",["Attention is all you need", 
                                                         "BERT : Pre-training of Deep Bidirectional Transformers","GPT-3:Langauge modes are few-shot learners", "Diffusion model Beat GANs on Image synthesis"])
style_input = st.selectbox("Select Explaination Sytle", ["Begginer-friendly", "Technical", "Code-Oriented", "Mathematical","Intuitively"])

length_input = st.selectbox("Select Explaination length",["short(1-2) paragraphs","Medium (3-5)paragraphs", "Long (detailed explaination)"])

template = load_prompt('template.json')
# fill the placeholders
# prompt = template.invoke({
#     'paper_input' : paper_input,
#     'style_input' : style_input,
#     'length_input' : length_input
# })

if st.button("summarize"):
    chain = template | model
    result = chain.invoke({
    'paper_input' : paper_input,
    'style_input' : style_input,
    'length_input' : length_input})
    # result = model.invoke(prompt)
    st.write(result.content)