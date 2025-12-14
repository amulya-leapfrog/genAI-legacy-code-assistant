from dotenv import load_dotenv 
load_dotenv()

import os
import shutil
import git
from utils.loader import load_code_files, split_documents
from utils.embedding import create_vectorstore


GITHUB_REPO = "https://github.com/amulya-leapfrog/Go-microservice-booking-system.git"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
TEMP_CODEBASE_PATH='./codebase'
MODEL_NAME = os.environ.get("EMBEDDING_MODEL_ID")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

# Clone the github repo and store it in temp codebase folder
def clone_repo(repo_url, target_path, token=None):
    # Check if codebase already exists and delete it
    if os.path.exists(target_path):
        print(f"Removing existing codebase at {target_path}...")
        shutil.rmtree(target_path)
    
    # Add token to URL if provided
    if token:
        repo_url = repo_url.replace('https://', f'https://{token}@')
    
    print(f"Cloning repository to {target_path}...")
    git.Repo.clone_from(repo_url, target_path)
    print("Repository cloned successfully!")
    
    return target_path
CODEBASE_PATH = clone_repo(GITHUB_REPO, TEMP_CODEBASE_PATH, GITHUB_TOKEN)

# Load and split code files
raw_documents = load_code_files(CODEBASE_PATH)
split_docs = split_documents(raw_documents)

# Create Pinecone vectorstore
vectorstore = create_vectorstore(split_docs, MODEL_NAME, PINECONE_INDEX_NAME)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

print("Vectorstore created and retriever ready for Runnable")
