from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    model_name =  "vietnamese-embedding"
    embeddings = HuggingFaceEmbeddings( model_name = model_name)
    #embeddings = OllamaEmbeddings(model=model_name)
    #print("EMBEDDING MODEL:",model_name)
    return embeddings
