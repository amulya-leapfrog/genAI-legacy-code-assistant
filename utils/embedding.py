import os
from langchain_aws import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from utils.llm import bedrock_client


def create_vectorstore(split_docs, model_id, index_name):
    """
    Create or connect to a Pinecone vector store with the given documents.
    
    Args:
        split_docs: List of document chunks to embed
        model_id: Bedrock embedding model ID (e.g., 'amazon.titan-embed-text-v2:0')
        index_name: Name of the Pinecone index
    
    Returns:
        PineconeVectorStore: The vector store instance
    """
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    # Initialize embeddings
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id=model_id
    )
    
    # Check if index exists, create if not
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1024,  # amazon.titan-embed-text-v2:0 dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        # Wait for index to be ready
        import time
        time.sleep(10)
    else:
        print(f"Index {index_name} exists. Deleting all existing vectors...")
        # Connect to the index
        index = pc.Index(index_name)
        
        # Delete all vectors in the default namespace
        try:
            index.delete(delete_all=True, namespace="")
            print("All vectors deleted from default namespace.")
        except Exception as e:
            print(f"Warning during deletion: {e}")
            # Continue anyway - the namespace might just be empty
        
    # Create vector store from documents with explicit namespace
    print(f"Adding {len(split_docs)} documents to Pinecone...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        index_name=index_name,
        namespace=""  # Use default namespace
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
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    print(f"Connecting to existing index: {index_name}")

    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id=model_id
    )
    
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    
    return vectorstore