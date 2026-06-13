from langchain_classic.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader

loader = DirectoryLoader(
    path = "data/" ,
    glob = "*.csv",
    loader_cls = PyPDFLoader
)

loader_web = WebBaseLoader(web_path = 'https://icseindia.org/document/sample.pdf')
docs = loader_web.load()




print(len(docs))
print(docs)
