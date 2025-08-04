import streamlit as st
import requests
import json
import time
from typing import (
    Dict,
    Any
)
from pathlib import Path

#   Page configuration.

st.set_page_config(
    page_title="KAVAK - Conversational Travel Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#   Custom CSS for styling.

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .search-result {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

#   API Configuration.

API_BASE_URL="http://localhost:8001"

#   Function to check if the API is reachable.

def check_api_connection():
    try:
        response=requests.get(f'{API_BASE_URL}/', timeout=5)
        return response.status_code==200
    except:
        return False
    
#   Function to create a new vector store collection.

def create_collection(collection_name: str) -> Dict[str, Any]:
    try:
        response=requests.post(
            f'{API_BASE_URL}/create-collection',
            json={"collection_name": collection_name},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}
    
#   Function to ingest data from a file into the vector store.

def ingest_data(filename: str, file_type: str, collection_name: str) -> Dict[str, Any]:
    try:
        response=requests.post(
            f'{API_BASE_URL}/ingest',
            json={
                "filename": filename,
                "file_type": file_type,
                "collection_name": collection_name
            },
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}
    
#   Function to search using the LangGraph agent.

def search_with_langgraph(query: str, collection_name: str) -> Dict[str, Any]:
    try:
        response=requests.post(
            f'{API_BASE_URL}/search',
            json={
                "query": query,
                "collection_name": collection_name
            },
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}
    
#   Function to get the list of available files in the data directory.

def get_available_files():
    data_dir=Path("data")
    if not data_dir.exists():
        return []
    files=[]
    for file_path in data_dir.rglob("*"):
        if file_path.is_file():

            #   Determining file type based on extension.

            if file_path.suffix.lower()==".json":
                file_type="json"
            elif file_path.suffix.lower() in [".md", ".markdown"]:
                file_type="markdown"
            elif file_path.suffix.lower() in [".txt", ".text"]:
                file_type="text"
            else:
                continue
            
            files.append({
                "path": str(file_path),
                "name": file_path.name,
                "type": file_type,
                "size": file_path.stat().st_size
            })
    return files

#   Driver function.

def main():
    st.markdown('<h1 class="main-header">✈️ KAVAK - Conversational Travel Assistant</h1>', unsafe_allow_html=True)
    
    #   Checking API connection.

    if not check_api_connection():
        st.error("❌ Cannot connect to KAVAK API server. Please ensure the server is running on http://localhost:8001")
        st.info("💡 Start the server with: `python src/main.py`")
        return
    
    st.success("✅ Connected to KAVAK API server")
    st.sidebar.title("Navigation")  #   Sidebar title.
    page=st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "🗄️ Vector Store", "📁 Data Ingestion", "🔍 Search", "📊 Analytics"]
    )
    if page=="🏠 Dashboard":
        show_dashboard()
    elif page=="🗄️ Vector Store":
        show_vector_store()
    elif page=="📁 Data Ingestion":
        show_data_ingestion()
    elif page=="🔍 Search":
        show_search()
    elif page=="📊 Analytics":
        show_analytics()

#   Function to display the dashboard.

def show_dashboard():
    st.markdown("""<h2 class="sub-header">🚀 Welcome to KAVAK's Conversational Travel Assistant Platform!</h2>""", unsafe_allow_html=True)
    col1, col2, col3=st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Hybrid RAG</h3>
            <p>Advanced retrieval with query classification, dynamic filtering, and LLM reranking</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 LangGraph</h3>
            <p>Sophisticated workflow orchestration with intelligent routing and processing</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Vector Search</h3>
            <p>High-performance vector storage with Qdrant and Gemini embeddings</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    
    #   Quick start guide.

    st.markdown("<h3>🚀 Quick Start Guide</h3>", unsafe_allow_html=True)
    st.markdown("""
    1. **Create Vector Store**: Go to the Vector Store page to create a new collection.
    2. **Ingest Data**: Use the Data Ingestion page to upload and process your files.
    3. **Search**: Use the Search page to query your data with the LangGraph agent.
    4. **Monitor**: Check Analytics to see system performance and usage.
    """)
    
    #   System status overview.

    st.markdown("<h3>📊 System Status</h3>", unsafe_allow_html=True)
    col1, col2=st.columns(2)
    with col1:
        st.success("✅ API Server: Running")
        st.success("✅ Vector Store: Available")
    with col2:
        st.info("ℹ️ LangGraph: Ready")
        st.info("ℹ️ Embeddings: Gemini text-embedding-004")

#   Function to show vector store management.

def show_vector_store():
    st.markdown('<h2 class="sub-header">🗄️ Vector Store Management</h2>', unsafe_allow_html=True)
    
    #   Create new collection form.

    st.markdown("<h3>➕ Create New Collection</h3>", unsafe_allow_html=True)
    with st.form("create_collection_form"):
        collection_name=st.text_input(
            "Collection Name",
            placeholder="e.g., flights_data, travel_policies",
            help="Enter a unique name for your vector collection"
        )
        col1, col2=st.columns(2)
        with col1:
            vector_size=st.number_input("Vector Size", value=768, disabled=True)
        with col2:
            embedding_model=st.text_input("Embedding Model", value="Gemini text-embedding-004", disabled=True)
        submitted=st.form_submit_button("Create Collection", type="primary")
        if submitted:
            if collection_name:
                with st.spinner("Creating collection..."):
                    result=create_collection(collection_name)
                if result.get("success"):
                    st.success(f'✅ Collection "{collection_name}" created successfully!')
                    st.json(result)
                else:
                    st.error(f'❌ Failed to create collection: {result.get("error", "Unknown error")}')
            else:
                st.error("Please enter a collection name")

#   Function to show data ingestion interface.

def show_data_ingestion():
    st.markdown('<h2 class="sub-header">📁 Data Ingestion</h2>', unsafe_allow_html=True)
    st.markdown("<h3>📂 Select File to Ingest</h3>", unsafe_allow_html=True)
    available_files=get_available_files()
    if not available_files:
        st.warning('⚠️ No files found in the data directory. Please add files to the "data" folder.')
        st.info("Supported formats: JSON, Markdown (.md), Text (.txt)")
        return
    
    #   File selection dropdown.

    selected_file=st.selectbox(
        "Choose a file:",
        options=available_files,
        format_func=lambda x: f'{x["name"]} ({x["type"].upper()}, {x["size"]} bytes)'
    )
    if selected_file:
        st.info(f'Selected: {selected_file["name"]} ({selected_file["type"]})')
        
        #   Collection name input.

        collection_name=st.text_input(
            "Target Collection Name",
            value="default_collection",
            help="Enter the collection name where you want to store this data"
        )
        
        #   File preview option.

        if st.checkbox("Preview file content"):
            try:
                with open(selected_file["path"], "r", encoding="utf-8") as f:
                    content=f.read()
                    if len(content) > 1000:
                        st.text_area("File Preview (first 1000 chars):", content[:1000] + "...")
                    else:
                        st.text_area("File Preview:", content)
            except Exception as e:
                st.error(f'Error reading file: {e}')
        
        #   Ingestion button.

        if st.button("🚀 Ingest Data", type="primary"):
            if collection_name:
                with st.spinner("Ingesting data..."):
                    result=ingest_data(
                        filename=selected_file["path"],
                        file_type=selected_file["type"],
                        collection_name=collection_name
                    )
                if result.get("success"):
                    st.success(f'✅ Successfully ingested {result.get("documents_processed", 0)} documents!')
                    st.json(result)
                else:
                    st.error(f'❌ Ingestion failed: {result.get("error", "Unknown error")}')
            else:
                st.error("Please enter a collection name")

#   Function to show search interface.

def show_search():
    st.markdown('<h2 class="sub-header">🔍 LangGraph Search</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>💡 Search Capabilities:</strong><br>
        • <strong>Flight Queries:</strong> "Emirates business class flights to Dubai under $2000"<br>
        • <strong>Information Queries:</strong> "What are the refund policies for cancelled flights?"<br>
        • <strong>Mixed Queries:</strong> "Flights to Japan and visa requirements for US citizens"
    </div>
    """, unsafe_allow_html=True)
    
    #   Input fields for search query and collection name.

    collection_name=st.text_input(
        "Collection Name",
        value="default_collection",
        help="Enter the collection name to search in"
    )
    query=st.text_area(
        "Search Query",
        placeholder="Enter your search query here...",
        height=100,
        help="Ask questions about flights, travel policies, or any travel-related information"
    )
    
    #   Options to show filters and metrics.

    col1, col2=st.columns(2)
    with col1:
        show_filters=st.checkbox("Show Applied Filters", value=True)
    with col2:
        show_metrics=st.checkbox("Show Processing Metrics", value=True)
    
    #   Search button.

    if st.button("🔍 Search with LangGraph", type="primary"):
        if query and collection_name:
            with st.spinner("Searching with LangGraph agent..."):
                start_time=time.time()
                result=search_with_langgraph(query, collection_name)
                search_time=time.time() - start_time
            if result.get("success"):
                st.success("✅ Search completed successfully!")
                
                #   Displaying the search results.

                st.markdown('<div class="search-result">', unsafe_allow_html=True)
                st.markdown(f'**🤖 Generated Answer:**')
                st.markdown(result.get("answer", "No answer generated"))
                st.markdown("</div>", unsafe_allow_html=True)
                
                #   Displaying the documents used in the search.

                if show_filters or show_metrics:
                    col1, col2=st.columns(2)
                    with col1:
                        if show_filters and result.get("filters_applied"):
                            st.markdown("**🔧 Applied Filters:**")
                            st.json(result.get("filters_applied"))
                        elif show_filters:
                            st.info("No filters were applied to this query")
                    with col2:
                        if show_metrics:
                            st.markdown("**📊 Processing Metrics:**")
                            metrics_data={
                                "Query Type": result.get("query_type", "unknown"),
                                "Documents Used": result.get("documents_used", 0),
                                "Processing Time": f'{result.get("processing_time", 0):.2f}s',
                                "Total Time": f'{search_time:.2f}s'
                            }
                            st.json(metrics_data)
                
                #   Displaying the full response if requested.

                with st.expander("🔍 View Full Response"):
                    st.json(result)

            else:
                st.error(f'❌ Search failed: {result.get("error", "Unknown error")}')
        else:
            st.error("Please enter both a query and collection name")

#   Function to show analytics and system information.

def show_analytics():
    st.markdown('<h2 class="sub-header">📊 Analytics & System Info</h2>', unsafe_allow_html=True)
    
    #   System information section.

    st.markdown("<h3>⚙️ System Information</h3>", unsafe_allow_html=True)
    col1, col2=st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🔧 API Status</h4>
            <p>✅ Running</p>
            <p>Port: 8001</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <h4>🧠 LangGraph</h4>
            <p>✅ Active</p>
            <p>Workflow: Hybrid RAG</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🗄️ Vector Store</h4>
            <p>✅ Qdrant</p>
            <p>Embeddings: Gemini</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <h4>🔍 Search Engine</h4>
            <p>✅ Hybrid</p>
            <p>Dense + Sparse</p>
        </div>
        """, unsafe_allow_html=True)
    
    #   System architecture diagram.

    st.markdown("<h3>🏗️ System Architecture</h3>", unsafe_allow_html=True)
    st.markdown("""
    ```mermaid
    graph TD
        A[User Query] --> B[Query Classification]
        B --> C{Query Type}
        C -->|Flight| D[Generate Filters]
        C -->|Info| E[Hybrid Retrieval]
        C -->|Both| F[Both Paths]
        D --> G[Apply Hard Filters]
        G --> H[LLM Reranker]
        E --> I[Merge Documents]
        F --> I
        H --> I
        I --> J[Generate Answer]
        J --> K[Response]
    ```
    """)
    
    #   Key features section.

    st.markdown("<h3>✨ Key Features</h3>", unsafe_allow_html=True)
    features=[
        "🎯 **Intelligent Query Classification**: Automatically determines query type (flight/info/both)",
        "🔧 **Dynamic Filter Generation**: LLM-powered filter creation for precise results",
        "🎛️ **Hard Filtering**: Metadata-based document filtering",
        "🧠 **LLM Reranking**: GPT-4o-mini powered document reranking",
        "🔍 **Hybrid Retrieval**: Combines dense and sparse vector search",
        "📄 **Document Merging**: Intelligent combination of results from different paths",
        "⚡ **Async Processing**: Full async/await support for better performance",
        "🛡️ **Robust Fallbacks**: Multiple fallback strategies ensure reliability"
    ]
    for feature in features:
        st.markdown(f'• {feature}')

if __name__=="__main__":
    main()