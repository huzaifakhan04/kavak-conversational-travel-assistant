from pydantic import BaseModel, validator
from typing import (
    Literal,
    Optional
)
from enum import Enum

class FileType(str, Enum):
    JSON="json"
    MARKDOWN="markdown"
    TEXT="text"

class DataIngestionRequest(BaseModel):

    filename: str
    file_type: FileType
    collection_name: str

    @validator("filename")
    def validate_filename(cls, v):
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        return v.strip()
    
    @validator("collection_name")
    def validate_collection_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Collection name cannot be empty")
        return v.strip()

class DataIngestionResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int
    collection_name: str

class CreateCollectionRequest(BaseModel):

    collection_name: str
    
    @validator("collection_name")
    def validate_collection_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Collection name cannot be empty")
        clean_name=v.strip().lower().replace(" ", "_").replace("-", "_")    #   Ensuring collection name is valid.
        if not clean_name.replace("_", "").isalnum():
            raise ValueError("Collection name must contain only alphanumeric characters, spaces, hyphens, or underscores")
        return clean_name

class CreateCollectionResponse(BaseModel):
    success: bool
    message: str
    collection_name: str
    vector_size: int
    embedding_model: str

class SearchRequest(BaseModel):

    query: str
    collection_name: str
    
    @validator("query")
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    @validator("collection_name")
    def validate_collection_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Collection name cannot be empty")
        return v.strip()

class SearchResponse(BaseModel):
    success: bool
    message: str
    answer: str
    query_type: str
    filters_applied: Optional[dict]=None
    documents_used: int
    processing_time: float