from langchain_classic.text_splitter import CharacterTextSplitter

from langchain_classic.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"D:\live_Projects\langchain\data\instruction_notification.pdf")

docs = loader.load()



splitter = CharacterTextSplitter(
    chunk_size = 1000,
    separator = "."
)

chunks = splitter.split_documents(docs)
print(len(chunks))
print(len(docs))