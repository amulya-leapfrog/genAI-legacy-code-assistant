import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_code_files(path: str):
    documents = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith((".html", ".css", ".py", ".go", ".js", ".jsx", ".ts", ".tsx", ".md", ".yaml", ".yml", ".json", ".txt",".cs", ".c", ".cpp", ".h", ".hpp")):
                file_path = os.path.join(root, file)
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["file"] = file_path
                documents.extend(docs)
    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)
