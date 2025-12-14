from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from utils.embedding import get_existing_vectorstore
from git_load import load_git_repo
from utils.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage


# Page config
st.set_page_config(
    page_title="Legacy Code Assistant",
    page_icon="💻",
    layout="wide"
)

# Title and description
st.title("💻 Legacy Code Assistant")
st.markdown("Ask questions about your codebase and get AI-powered answers!")

# Sidebar configuration --- INIT CONFIGURATION FOR MODEL AND PINECONE INDEX --- DEFAULTS TO ENV VARS
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection - different options for Bedrock and OpenAI
    llm_provider = os.environ.get("LLM_PROVIDER")
    
    if llm_provider == "bedrock":
        model_options = [
            "amazon.nova-lite-v1:0",
            "amazon.nova-pro-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0"
        ]
        default_model = os.environ.get("LLM_MODEL_ID", "amazon.nova-lite-v1:0")
    else:
        model_options = ["gpt-4", "gpt-3.5-turbo"]
        default_model = "gpt-4"
    
    selected_model = st.selectbox(
        "Select Model",
        model_options,
        index=model_options.index(default_model) if default_model in model_options else 0
    )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    
    num_results = st.slider("Number of Code Snippets", 1, 10, 5)
    
    st.divider()
    
    # GitHub Repository Section
    st.subheader("📦 Repository Setup")
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/user/repo.git",
        help="Enter the GitHub repository URL to analyze"
    )
    
    is_private = st.checkbox("Private Repository", help="Check if the repo requires authentication")
    
    github_token = None
    if is_private:
        github_token = st.text_input(
            "GitHub Token",
            type="password",
            help="Personal access token for private repos"
        )
    
    if st.button("🔄 Load Repository", type="primary"):
        if repo_url:
            with st.spinner("Cloning and processing repository..."):
                try:
                    vectorstore = load_git_repo(repo_url, github_token)

                    # Update session state with new vectorstore and retriever
                    st.session_state.vectorstore = vectorstore
                    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": num_results})
                    
                    # Clear old chat history since it's about a different repo
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    st.success(f"✅ Repository loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading repository: {str(e)}")
        else:
            st.warning("⚠️ Please enter a repository URL")
    
    st.divider()

    # Load existing vectorstore option
    st.subheader("📚 Existing Vectorstore")
    if st.button("📥 Load Existing Vectorstore"):
        with st.spinner("Loading vectorstore..."):
            try:
                embedding_model = os.environ.get("EMBEDDING_MODEL_ID")
                index_name = os.environ.get("PINECONE_INDEX_NAME")
                st.session_state.vectorstore = get_existing_vectorstore(embedding_model, index_name)
                st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": num_results}
                )
                st.success("✅ Vectorstore loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading vectorstore: {str(e)}")
    
    # Index info
    if "vectorstore" in st.session_state:
        st.markdown("**Status:** ✅ Vectorstore loaded")
    else:
        st.markdown("**Status:** ⚠️ No repository loaded")

    st.divider()

    st.markdown("**Index:** " + os.environ.get("PINECONE_INDEX_NAME", "N/A"))
    st.markdown("**Embedding Model:** " + os.environ.get("EMBEDDING_MODEL_ID", "N/A"))
    
    st.divider()

    # LangSmith tracing toggle
    if os.environ.get("LANGSMITH_API_KEY"):
        st.subheader("🔍 LangSmith")
        langsmith_enabled = st.checkbox(
            "Enable Tracing",
            value=os.environ.get("LANGSMITH_TRACING_V2") == "true",
            help="Track and debug LangChain calls in LangSmith"
        )
        if langsmith_enabled:
            os.environ["LANGSMITH_TRACING_V2"] = "true"
            st.markdown(f"**Project:** {os.environ.get('LANGSMITH_PROJECT', 'default')}")
        else:
            os.environ["LANGSMITH_TRACING_V2"] = "false"
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if "qa_chain" in st.session_state:
            st.session_state.qa_chain.clear_history()
        st.rerun()

# Initialize session state --- empty state for messages, chat history, and vectorstore
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    with st.spinner("Loading vectorstore..."):
        embedding_model = os.environ.get("EMBEDDING_MODEL_ID")
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        st.session_state.vectorstore = get_existing_vectorstore(embedding_model, index_name)
        st.session_state.retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": num_results}
        )

# Helper function to format retrieved documents into a single string for display
def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content
        formatted.append(f"[Source {i}: {source}]\n{content}")
    return "\n\n".join(formatted)

# Create the RAG chain
def create_rag_chain(retriever, model, temp, chat_history):
    llm = get_llm(model=model, temperature=temp)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
        "system", 
        """You are an expert code assistant helping developers understand a legacy codebase.
        Use the following code snippets to answer the question. If you don't know the answer based on the provided context, say so.
        Be specific and reference file names and code sections when relevant.

        Context:
        {context}
        """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # Create the chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: chat_history
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 View Source Code"):
                for i, doc in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:** `{doc.metadata.get('source', 'Unknown')}`")
                    st.code(doc.page_content, language="python")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about the codebase..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get relevant documents
            source_docs = st.session_state.retriever.invoke(prompt)
            
            # Create chain and get response
            rag_chain = create_rag_chain(
                st.session_state.retriever,
                selected_model,
                temperature,
                st.session_state.chat_history
            )
            
            answer = rag_chain.invoke(prompt)
            
            st.markdown(answer)
            
            # Show sources
            if source_docs:
                with st.expander("📄 View Source Code"):
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Source {i}:** `{doc.metadata.get('source', 'Unknown')}`")
                        st.code(doc.page_content, language="python")
                        st.divider()
            
            # Update chat history
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=answer))
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_docs
            })

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        💡 Tip: Ask specific questions about functions, classes, or implementation details for best results.
    </div>
    """,
    unsafe_allow_html=True
)