# 💻 Legacy Codebase Assistant

An AI-powered tool that helps developers understand and navigate legacy codebases through natural language conversations. Ask questions about any GitHub repository and get intelligent, context-aware answers with relevant code snippets.

## Features

- **GitHub Integration** - Analyze any public or private repository
- **Intelligent Q&A** - Ask questions in natural language about code structure and functionality
- **Conversation Memory** - Follow-up questions that remember context
- **Source Code Display** - View relevant code snippets with file references
- **Multiple LLM Options** - Choose between Amazon Nova, Claude, and other models
- **Real-time Processing** - Clone, index, and query repositories on-demand
- **LangSmith Integration** - Optional monitoring and debugging

## Prerequisites

Before you begin, ensure you have:

- Python 3.10 or higher
- AWS Account with Bedrock access
- Pinecone Account (free tier available)
- GitHub Account (for private repos, you'll need a Personal Access Token)
- Git installed on your system

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:amulya-leapfrog/genAI-legacy-code-assistant.git
cd genAI-legacyCode-helper
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Navigate to AWS Bedrock service and note your access key, secret key, etc.

### 5. Set Up Pinecone & note your pinecone API key and index name

### 6. Go to GitHub settings and generate the token for private repo

### 7. Set Up LangSmith & note your API key

### 8. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Or create manually
touch .env
```

### 9. Verify Setup

Test your environment variables:

```bash
python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); print('AWS Region:', os.getenv('AWS_REGION')); print('Pinecone Index:', os.getenv('PINECONE_INDEX_NAME'))"
```

## Running the Application

### Using Streamlit UI

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide

### Loading a Repository

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **In the sidebar, choose one option:**

   **Option A: Load New Repository**
   - Enter GitHub repository URL (e.g., `https://github.com/user/repo.git`)
   - For private repos: Check "Private Repository" and enter your token
   - Click "Load Repository"
   - Wait for processing (1-2 minutes depending on repo size)

   **Option B: Load Existing Vectorstore**
   - Click "Load Existing Vectorstore"
   - Connects to previously indexed repository (instant)

### Example Questions

```
User: "What is the main purpose of this codebase?"
Assistant: [Analyzes code and provides summary with file references]

User: "How does the authentication system work?"
Assistant: [Explains auth flow with relevant code snippets]

User: "Can you show me the database schema?"
Assistant: [Displays model definitions and relationships]
```

## Configuration Options

### Model Selection
In the sidebar, you can select different models depending on your LLM provider:
- **Amazon Nova Lite**
- **Amazon Nova Pro**
- **Claude 3 Sonnet**
- **Claude 3 Haiku** 
- **gpt-4**
- **gpt-3.5-turbo**

### Parameters
- **Temperature** (0.0 - 1.0): Controls creativity
  - 0.0 = Deterministic, focused
  - 1.0 = Creative, varied
- **Number of Code Snippets** (1-10): How many relevant chunks to retrieve

## Project Structure

```
genAI-legacyCode-helper/
├── app.py                  # Streamlit UI application
├── git_load.py                 # Load git repo
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── .env.example           # Example environment file
├── utils/
│   ├── llm.py             # LLM configuration (Bedrock/OpenAI)
│   ├── embedding.py       # Pinecone vectorstore management
│   ├── loader.py          # Code file loading and splitting
├── codebase/              # Cloned repositories (auto-created & deleted after vector embeddings are saved)
└── .gitignore         # Ignore files in the directory
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### AWS Bedrock Access Denied
- Ensure you've requested model access in AWS Bedrock console
- Verify your IAM user has `bedrock:InvokeModel` permissions
- Check your AWS credentials are correct in `.env`

### Pinecone Namespace Not Found
- This is usually harmless and can be ignored
- The system will create a fresh namespace automatically

### GitHub Clone Failed
- For private repos, ensure your token has correct permissions
- Check the repository URL is correct
- Verify your internet connection

### Streamlit Port Already in Use
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

## Monitoring with LangSmith

If you've enabled LangSmith:

1. Visit [smith.langchain.com](https://smith.langchain.com/)
2. Select your project: e.g. "legacy-code-assistant"
3. View traces for each query showing:
   - Execution time
   - Token usage
   - Retrieved documents
   - Model inputs/outputs
   - Error logs

## Security Notes

- **Never commit `.env` file** - It contains sensitive credentials
- Add `.env` to `.gitignore`
---
**Happy Code Exploration!**