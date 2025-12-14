import os
import boto3
from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.environ.get("AWS_SESSION_TOKEN")
AWS_REGION = os.environ.get("AWS_REGION")

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
)

llm_provider = os.environ.get("LLM_PROVIDER")

def get_llm(model=None, temperature=0):
    if llm_provider == "openai":
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        return ChatOpenAI(
            api_key=openai_api_key,
            model=model or "gpt-4.1-nano",
            temperature=temperature
        )
    elif llm_provider == "bedrock":
        return ChatBedrock(
            client=bedrock_client,
            model_id=model or "amazon.nova-lite-v1:0",
            temperature=temperature
        )
