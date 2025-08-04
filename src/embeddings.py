import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger=logging.getLogger(__name__)

#   Initializing the embedding model for Gemini.

def get_embedding_model(model_name: str="text-embedding-004"):
    try:
        google_api_key=os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        model_mapping={
            "text-embedding-004": "models/text-embedding-004",
            "embedding-001": "models/embedding-001"
        }
        full_model_name=model_mapping.get(model_name, f"models/{model_name}")
        embeddings=GoogleGenerativeAIEmbeddings(
            model=full_model_name,
            google_api_key=google_api_key
        )
        logger.info(f"Successfully initialized Gemini embeddings with model: {full_model_name}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize Gemini embeddings: {str(e)}")
        raise