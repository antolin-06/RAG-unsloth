import json

import boto3
from chromadb.utils.embedding_functions.google_embedding_function import GoogleGenerativeAiEmbeddingFunction
from langchain_aws import BedrockEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from torch.fx.experimental.migrate_gradual_types.constraint_generator import embedding_inference_rule

# Function to generate vector embeddings -> An embedding is a vector representation of a text that captures the meaning of the text
# In python the vector embedding are numbers an if we compare 2 embeddings and are close that means that have similar meanings.
''' # AWS NO TIRA POR LOS CREDENCIALES
def get_embedding_function():
    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='eu-west-1')
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1", client=bedrock_client
    )
    return embeddings
'''
## Bedrock Clients
def get_embedding_function_bedrock():
    bedrock = boto3.client(service_name="bedrock-runtime", region_name='eu-central-1')
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)
    return embeddings

 # Ollama
def get_embedding_function_ollama():
    embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    return embeddings

# Sentence Transformer - Hugging Face -- es más complejo de hacer la llamada, mas sencillo utilizando langchain que para algo lo crearon
def get_embedding_function_sentence_transformer():
    embeddings = SentenceTransformer("BAAI/bge-m3").encode()  # "sentence-transformers/msmarco-bert-base-dot-v5"
    return embeddings

# Hugging-Face, descargando un modelo de la web
def get_embedding_function_huggingface():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    return embeddings


# OTRA FORMA DE PODER IMPLEMENTARLO MEDIANTE LA LIBRERÍA DE CHROMA

# import chromadb.utils.embedding_functions as embedding_functions
# huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
#     api_key="YOUR_API_KEY",
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
'''
def get_embedding_function_openai():
    st.session_state.LLM_provider == "OpenAI":
    embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
    return embeddings
'''

def get_embedding_function_google(api_key:str):
    embedding = GoogleGenerativeAIEmbeddings(google_api_key= api_key, model="models/embedding-001")
    return embedding



