from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from langchain_classic.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"D:\live_Projects\langchain\data\instruction_notification.pdf")

docs = loader.load()



splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

chunks = splitter.split_documents(docs)
print(len(chunks))
print(len(docs))