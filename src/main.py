import os
import time
import logging
import nest_asyncio
from fastapi import (
    FastAPI,
    HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from models import (
    DataIngestionRequest,
    DataIngestionResponse,
    CreateCollectionRequest,
    CreateCollectionResponse,
    SearchRequest,
    SearchResponse
)
from ingestion import (
    ingest_data_to_qdrant,
    create_collection
)
from graph import run_search_and_answer

#   Creating a directory for logs if it doesn't exist.

log_directory="logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)


#   Configure logging.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(name)s-%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(log_directory, "app.log"))],
)

os.environ["TZ"]="Asia/Karachi"
time.tzset()

logger=logging.getLogger(__name__)
nest_asyncio.apply()

#   Initialize FastAPI application.

app_kwargs={"title": "KAVAK"}
if os.getenv("ENVIRONMENT")!="dev":
    app_kwargs["docs_url"]=None
    app_kwargs["redoc_url"]=None
    app_kwargs["openapi_url"]=None

app=FastAPI(**app_kwargs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the KAVAK's Conversational Travel Assistant Platform!"}

#   Endpoint to ingest data into Qdrant.

@app.post("/ingest", response_model=DataIngestionResponse)
async def ingest_data(request: DataIngestionRequest):
    try:
        logger.info(f"Starting data ingestion for file: {request.filename}, type: {request.file_type}")
        
        #   Validate file path and ensure it is within the project directory.

        if not os.path.isabs(request.filename):

            #   If the filename is relative, construct the absolute path.

            project_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path=os.path.join(project_root, request.filename)
        else:
            file_path=request.filename
        
        #   Check if the file exists.

        project_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.commonpath([file_path, project_root])==project_root:
            raise HTTPException(
                status_code=400,
                detail="File path must be within the project directory"
            )

        documents_processed=await ingest_data_to_qdrant(
            file_path=file_path,
            file_type=request.file_type,
            collection_name=request.collection_name
        )   #   Ingesting data from the file into Qdrant vector store.
        logger.info(f"Successfully ingested {documents_processed} documents from {request.filename}")
        return DataIngestionResponse(
            success=True,
            message=f"Successfully ingested {documents_processed} documents from {request.filename}",
            documents_processed=documents_processed,
            collection_name=request.collection_name
        )
    except FileNotFoundError:
        logger.error(f"File not found: {request.filename}")
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {request.filename}"
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error during data ingestion: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during data ingestion: {str(e)}"
        )
    
#   Endpoint to create a new collection in Qdrant.

@app.post("/create-collection", response_model=CreateCollectionResponse)
async def create_new_collection(request: CreateCollectionRequest):
    try:
        logger.info(f"Creating new collection: {request.collection_name}")
        result=await create_collection(
            collection_name=request.collection_name
        )
        if result["success"]:
            logger.info(f"Successfully created collection: {request.collection_name}")
            return CreateCollectionResponse(
                success=True,
                message=result["message"],
                collection_name=result["collection_name"],
                vector_size=result["vector_size"],
                embedding_model=result["embedding_model"]
            )
        else:
            logger.error(f"Failed to create collection: {result.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Collection creation failed")
            )   
    except ValueError as e:
        logger.error(f"Validation error during collection creation: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during collection creation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during collection creation: {str(e)}"
        )
    
'''

    Search using LangGraph agent with hybrid RAG capabilities.
    
    This endpoint uses the sophisticated LangGraph workflow that includes:

-  Query classification (flight/info/both)
-  Dynamic filter generation
-  Hard filtering with metadata
-  LLM reranking
-  Hybrid retrieval
-  Answer generation

'''
    
@app.post("/search", response_model=SearchResponse)
async def search_with_langgraph(request: SearchRequest):
    """
    Search using LangGraph agent with hybrid RAG capabilities.
    
    This endpoint uses the sophisticated LangGraph workflow that includes:
-Query classification (flight/info/both)
-Dynamic filter generation
-Hard filtering with metadata
-LLM reranking
-Hybrid retrieval
-Answer generation
    
    Args:
        request: Search request with query and collection name
        
    Returns:
        Search response with generated answer and processing details
    """
    try:
        logger.info(f"Starting LangGraph search for query: '{request.query}' in collection: {request.collection_name}")
        
        start_time=time.time()
        
        #   Running the LangGraph search workflow.

        result=await run_search_and_answer(
            query=request.query,
            collection_name=request.collection_name
        )
        
        processing_time=time.time()-start_time
        if result.get("success", False):
            logger.info(f"Successfully completed search in {processing_time:.2f}s")
            return SearchResponse(
                success=True,
                message="Search completed successfully",
                answer=result.get("answer", "No answer generated"),
                query_type=result.get("query_type", "unknown"),
                filters_applied=result.get("filters", {}),
                documents_used=result.get("documents_used", 0),
                processing_time=processing_time
            )
        else:
            error_msg=result.get('error', 'Unknown error')
            logger.error(f"Search failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Search operation failed: {error_msg}"
            )
    except ValueError as e:
        logger.error(f"Validation error during search: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during search: {str(e)}"
        )

if __name__=="__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except Exception as e:
        logger.exception(f"Error starting the server: {e}")