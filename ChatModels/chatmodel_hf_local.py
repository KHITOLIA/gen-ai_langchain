# # from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# # llm = HuggingFacePipeline(
# #     model_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
# #     task = "text-generation",
# #     pipeline_kwargs = 
# #     dict(temperature = 0.5,
# #      max_new_tokens = 100)
# # )

# # model = ChatHuggingFace(llm)

# # result = model.invoke("what is the capital of india?")
# # print(result.content)



# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM # or AutoModelForCausalLM for generative models

# model_name = "google/flan-t5-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name) # or AutoModelForCausalLM

# hf_pipeline = pipeline(
#         "text2text-generation", # or "text-generation"
#         model=model,
#         tokenizer=tokenizer,
#     )

# from langchain_community.llms import HuggingFacePipeline

# llm = HuggingFacePipeline(pipeline=hf_pipeline)

# prompt =  "Explain thermodynamics in brief?."
# generated_text = llm.invoke(prompt)
# print(generated_text)