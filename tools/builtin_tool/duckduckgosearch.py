from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke("Whats the latest news in AI /ML ? ")
print(results) # ai agent : LLm + tool