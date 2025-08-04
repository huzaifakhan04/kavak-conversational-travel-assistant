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
from src.models import (
    DataIngestionRequest,
    DataIngestionResponse,
    CreateCollectionRequest,
    CreateCollectionResponse
)
from src.ingestion import (
    ingest_data_to_qdrant,
    create_collection
)

#   Creating a directory for logs if it doesn't exist.

log_directory="logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)


#   Configure logging.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(log_directory, "app.log"))],
)

os.environ["TZ"]="Asia/Karachi"
time.tzset()

logger=logging.getLogger(__name__)
nest_asyncio.apply()

#   Initialize FastAPI application.

app_kwargs={"title": "KAVAK"}
if os.getenv("ENVIRONMENT") != "dev":
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
    return {"message": "Welcome to the KAVAK Conversational Travel Assistant Platform!"}