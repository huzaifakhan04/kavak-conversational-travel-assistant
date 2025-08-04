import os
import json
import logging
from typing import (
    List,
    Optional
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.client_qdrant import (
    get_qdrant_client,
    initialize_vector_store,
    create_qdrant_collection
)
from src.models import FileType
from src.embeddings import get_embedding_model

logger=logging.getLogger(__name__)

#   Text splitter configuration.

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

#   Function to process JSON files and convert them into Document objects.

async def process_json_file(file_path: str) -> List[Document]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data=json.load(file)
        documents=[]
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    content=json.dumps(item, indent=2)
                    metadata=item.copy()
                    metadata.update({
                        "source": file_path,
                        "document_type": "json",
                        "item_index": i,
                        "total_items": len(data)
                    })
                    doc=Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)
        elif isinstance(data, dict):
            content=json.dumps(data, indent=2)
            metadata=data.copy()
            metadata.update({
                "source": file_path,
                "document_type": "json",
                "item_index": 0,
                "total_items": 1
            })
            doc=Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        else:
            raise ValueError(f"Unsupported JSON structure in {file_path}")
            
        logger.info(f"Successfully processed {len(documents)} JSON objects from {file_path}")
        return documents
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file {file_path}: {str(e)}")
        raise ValueError(f"Invalid JSON format in {file_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing JSON file {file_path}: {str(e)}")
        raise

#   Function to process a Markdown file by reading its content and chunking it.

async def process_markdown_file(file_path: str) -> List[Document]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content=file.read()
        
        chunks=text_splitter.split_text(content)
        documents=[]
        for i, chunk in enumerate(chunks):
            metadata={
                "source": file_path,
                "document_type": "markdown",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "filename": os.path.basename(file_path)
            }
            doc=Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(doc)
        logger.info(f"Successfully processed markdown file {file_path} into {len(documents)} chunks")
        return documents
    except Exception as e:
        logger.error(f"Error processing markdown file {file_path}: {str(e)}")
        raise

#   Function to ingest data from a file into Qdrant vector store.

async def ingest_data_to_qdrant(
    file_path: str,
    file_type: FileType,
    collection_name: str,
    embedding_model_name: str="text-embedding-004"
) -> int:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        file_extension=os.path.splitext(file_path)[1].lower()
        if file_type==FileType.JSON and file_extension != ".json":
            raise ValueError(f"File extension {file_extension} doesn't match declared type {file_type}")
        if file_type==FileType.MARKDOWN and file_extension not in [".md", ".markdown"]:
            raise ValueError(f"File extension {file_extension} doesn't match declared type {file_type}")
        if file_type==FileType.JSON:
            documents=await process_json_file(file_path)
        elif file_type==FileType.MARKDOWN:
            documents=await process_markdown_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        if not documents:
            logger.warning(f"No documents generated from file: {file_path}")
            return 0
        client=get_qdrant_client()
        embedding_model=get_embedding_model(embedding_model_name)
        vector_store=await initialize_vector_store(
            client=client,
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        if not vector_store:
            raise RuntimeError("Failed to initialize vector store")
        await vector_store.aadd_documents(documents=documents)
        logger.info(f"Successfully ingested {len(documents)} documents to collection '{collection_name}'")
        return len(documents)  
    except Exception as e:
        logger.error(f"Failed to ingest data from {file_path}: {str(e)}")
        raise
        
#   Function to create a new Qdrant collection with vector store initialization.

async def create_collection(
    collection_name: str
) -> dict:
    try:
        embedding_model_name="text-embedding-004"
        logger.info(f"Creating collection: {collection_name} with Gemini model: {embedding_model_name}")
        vector_size=768
        client=get_qdrant_client()
        await create_qdrant_collection(collection_name, client, vector_size)
        logger.info(f"Successfully created Qdrant collection: {collection_name}")
        embedding_model=get_embedding_model(embedding_model_name)
        vector_store=await initialize_vector_store(
            client=client,
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        if not vector_store:
            try:
                import asyncio
                await asyncio.to_thread(client.delete_collection, collection_name)
                logger.info(f"Cleaned up collection after vector store initialization failure: {collection_name}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to clean up collection after error: {str(cleanup_err)}")
            
            raise RuntimeError("Failed to initialize vector store")
        logger.info(f"Successfully initialized vector store for collection: {collection_name}")
        return {
            "success": True,
            "collection_name": collection_name,
            "vector_size": vector_size,
            "embedding_model": embedding_model_name,
            "message": f"Successfully created collection '{collection_name}' with Gemini embeddings (vector size {vector_size})"
        }
    except Exception as e:
        logger.error(f"Failed to create collection {collection_name}: {str(e)}")
        try:
            client=get_qdrant_client()
            import asyncio
            await asyncio.to_thread(client.delete_collection, collection_name)
            logger.info(f"Cleaned up partially created collection: {collection_name}")
        except Exception as cleanup_err:
            logger.warning(f"Failed to clean up collection after error: {str(cleanup_err)}")
        return {
            "success": False,
            "error": f"Collection creation failed: {str(e)}",
            "collection_name": collection_name
        }