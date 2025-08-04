import os
import asyncio
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance
)
from langchain_qdrant import (
    FastEmbedSparse,
    RetrievalMode,
    QdrantVectorStore
)
from typing import Optional

logger=logging.getLogger(__name__)

#   Configuration for Qdrant client connection.

def get_qdrant_client(timeout: int=30):
    qdrant_url=os.getenv("QDRANT_CLOUD")
    return QdrantClient(
        url=qdrant_url,
        api_key=os.getenv("QDRANT_CLOUD_KEY"),
        port=None,
        prefer_grpc=False,
        timeout=timeout
    )

#   Initialize the Qdrant vector store with the given parameters.

async def initialize_vector_store(
    client: QdrantClient,
    collection_name: str,
    embedding_model,
    sparse_model: str="Qdrant/bm25"
) -> Optional[QdrantVectorStore]:
    try:
        vector_store=await asyncio.to_thread(
            QdrantVectorStore,
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
            sparse_embedding=FastEmbedSparse(model_name=sparse_model),
            sparse_vector_name="default",
            retrieval_mode=RetrievalMode.HYBRID
        )
        logger.info(f"Successfully initialized vector store for collection: {collection_name}")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        return None

#   Create a new Qdrant collection with vector store initialization.

async def create_qdrant_collection(
    collection_name: str,
    client: QdrantClient,
    vector_size: int,
) -> None:
    try:
        collections=await asyncio.to_thread(client.get_collections)
        logger.info(f"Existing collections: {collections}")
        
        if any(collection.name==collection_name for collection in collections.collections):
            await asyncio.to_thread(client.delete_collection, collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        await asyncio.to_thread(
            client.create_collection,
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
            sparse_vectors_config={
                "default": SparseVectorParams()
            }
        )
        logger.info(f"Successfully created collection: {collection_name}")
        await create_filter_indexes(client, collection_name)
    except Exception as e:
        logger.error(f"Error in collection creation: {str(e)}")
        raise

#   Create payload indexes for fields that will be used in filtering.

async def create_filter_indexes(client: QdrantClient, collection_name: str) -> None:
    try:

        #   Defining the fields to index for filtering.

        filter_fields=[
            ("airline", "keyword"),
            ("alliance", "keyword"),
            ("from_country", "keyword"),
            ("to_country", "keyword"),
            ("travel_class", "keyword"),
            ("price_usd", "integer"),
            ("refundable", "bool"),
            ("baggage_included", "bool"),
            ("wifi_available", "bool"),
            ("meal_service", "keyword"),
            ("aircraft_type", "keyword"),
        ]
        
        for field_name, field_type in filter_fields:
            try:
                await asyncio.to_thread(
                    client.create_payload_index,
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.info(f"Created index for field: {field_name} ({field_type})")
            except Exception as e:
                logger.warning(f"Failed to create index for field {field_name}: {str(e)}")
                
                #   Continue with other fields even if one fails.
        
        logger.info(f"Successfully created filter indexes for collection: {collection_name}")
    except Exception as e:
        logger.error(f"Error creating filter indexes: {str(e)}")
        raise

#   Ensure that payload indexes exist for fields that will be used in filtering.

async def ensure_filter_indexes(client: QdrantClient, collection_name: str) -> None:
    try:
        
        #   Checking if the collection exists before creating indexes.
        
        collections=await asyncio.to_thread(client.get_collections)
        collection_exists=any(collection.name==collection_name for collection in collections.collections)
        
        if not collection_exists:
            logger.warning(f"Collection {collection_name} does not exist, cannot create indexes")
            return
        await create_filter_indexes(client, collection_name)    #   Create indexes for the collection.
    except Exception as e:
        logger.error(f"Error ensuring filter indexes: {str(e)}")
        raise