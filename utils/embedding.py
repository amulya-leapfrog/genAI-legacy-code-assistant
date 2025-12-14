import os
from langchain_aws import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from utils.llm import bedrock_client

def create_vectorstore(split_docs, model_id, index_name):
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    """
    Create or connect to a Pinecone vector store with the given documents.
    
    Args:
        split_docs: List of document chunks to embed
        model_id: Bedrock embedding model ID (e.g., 'amazon.titan-embed-text-v1')
        index_name: Name of the Pinecone index
    
    Returns:
        PineconeVectorStore: The vector store instance
    """
    # Initialize embeddings
    embeddings = BedrockEmbeddings(client=bedrock_client, model_id=model_id)
    
    # Check if index exists, create if not
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,  # Titan embeddings dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # Change to your preferred region
            )
        )
    
    # Create vector store from documents
    vectorstore = PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        index_name=index_name
    )
    
    return vectorstore

def get_existing_vectorstore(model_id, index_name):
    """
    Connect to an existing Pinecone vector store without adding documents.
    
    Args:
        model_id: Bedrock embedding model ID
        index_name: Name of the existing Pinecone index
    
    Returns:
        PineconeVectorStore: The vector store instance
    """
    
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id=model_id
    )
    
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    
    return vectorstore