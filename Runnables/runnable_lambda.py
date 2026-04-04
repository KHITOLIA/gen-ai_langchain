from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel, RunnableLambda, RunnablePassthrough
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-Llama/Llama-3.2-3B-Instruct",
    task = "text-generation",
    temperature = 0,
    max_new_tokens = 500
    )
model = ChatHuggingFace(llm = llm)
def word_counter(text):
    return len(text.split())

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'generate a tweet on the {topic}',
    input_variables =  ['topic']
)

joke_generator_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_counter)
})

# parallel_chain = RunnableParallel({
#     'joke' : RunnablePassthrough(),
#     'word_count' : RunnableLambda(lambda x : x.split())
# })
chain = RunnableSequence(joke_generator_chain, parallel_chain)
result = chain.invoke({"topic" : "AI"})
final_result = '''{} \n word count - {}'''.format(result['joke'], result['word_count'])
print(final_result)