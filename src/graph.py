import os
import logging
import asyncio
import json
from typing import (
    TypedDict,
    List,
    Dict,
    Any,
    Optional,
    Literal
)
from langgraph.graph import (
    StateGraph,
    START,
    END
)
from langgraph.types import Command
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)
from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain_core.documents import Document
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    Range
)
from src.client_qdrant import (
    get_qdrant_client,
    ensure_filter_indexes
)
from src.embeddings import get_embedding_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger=logging.getLogger(__name__)

embeddings=None
llm=None
client=None

#   Initialize the embedding model and Qdrant client asynchronously.

async def initialize_components():

    global embeddings, client
    if embeddings is None:
        embeddings=await asyncio.to_thread(get_embedding_model, "text-embedding-004")
    if client is None:
        client=await asyncio.to_thread(get_qdrant_client)

    pass

#   Initialize the LLM instance for answer generation.

async def get_gemini_llm():
    global llm
    if llm is None:
        try:
            google_api_key=os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            llm=await asyncio.to_thread(
                lambda: ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=google_api_key,
                    temperature=0.1
                )
            )
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {str(e)}")
            return None
    return llm

    pass

#   Represents the state of the minimal search and answer generation graph.

class GraphState(TypedDict):

    query: str
    collection_name: str
    query_type: str #   "flight_only", "info_only", "both"
    filters: Dict[str, Any] #   Hard filters to apply.
    filter_options: Dict[str, Any]  #   Available filter options.
    filtered_docs: List[Document]
    info_docs: List[Document]   #   Documents from simple retrieval.
    reranked_docs: List[Document]
    answer: str

#   Classifying the query to determine if it's flight-related, info-related, or both.

async def classify_query(state: GraphState) -> Command[Literal["generate_filters", "simple_retrieval"]]:
    logger.info("Starting query classification")
    try:
        query=state["query"]
        
        classification_prompt=ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier for a flight booking and travel information system. 
            Analyze the user's query and classify it into one of three categories:

            1. "flight_only" - Query is specifically about flight booking, searching, or flight details (e.g., "flights from NYC to London", "business class flights under $2000", "Emirates flights to Dubai")

            2. "info_only" - Query is about travel information, policies, rules, or general travel advice (e.g., "visa requirements for India", "refund policies", "baggage rules", "travel tips")

            3. "both" - Query contains both flight-specific requests and general information requests (e.g., "flights to Japan and visa requirements", "Emirates flights to Dubai and their baggage policy")

            Return only the classification string: "flight_only", "info_only", or "both"

            User Query: {query}"""),
            ("human", "Classify this query.")
        ])
        llm_instance=await get_gemini_llm()
        if llm_instance:
            try:
                response=await asyncio.to_thread(
                    lambda: llm_instance.invoke([
                        SystemMessage(content=classification_prompt.format(query=query)),
                        HumanMessage(content="Classify this query.")
                    ])
                )
                query_type=response.content.strip().lower()
                if query_type not in ["flight_only", "info_only", "both"]:
                    logger.warning(f"Invalid classification '{query_type}', defaulting to 'both'")
                    query_type="both"
                logger.info(f"Query classified as: {query_type}")
                if query_type in ["flight_only", "both"]:
                    return Command(goto="generate_filters", update={"query_type": query_type})
                else:
                    return Command(goto="simple_retrieval", update={"query_type": query_type})
            except Exception as e:
                logger.error(f"Error classifying query with LLM: {e}")
                return Command(goto="generate_filters", update={"query_type": "both"})
        else:
            logger.warning("LLM not available for query classification, defaulting to 'both'")
            return Command(goto="generate_filters", update={"query_type": "both"})
    except Exception as e:
        logger.error(f"Error in classify_query: {e}", exc_info=True)
        return Command(goto="generate_filters", update={"query_type": "both"})

#   Generate dynamic filters using LLM based on the query and available filter options.

async def generate_filters(state: GraphState) -> Dict[str, Any]:
    logger.info("Starting dynamic filter generation")
    try:
        await initialize_components()
        query=state["query"]
        filter_options=state.get("filter_options", get_filter_options())
        logger.info(f"Generating filters for query: {query}")
        filter_prompt=ChatPromptTemplate.from_messages([
            ("system", """You are a filter generation assistant for a flight booking system. Based on the user's query and available filter options, generate appropriate filters to narrow down the search results.

            Available filter options:
            {filter_options}

            Instructions:
            
            1. Analyze the user's query to identify relevant filters
            2. For layover queries (e.g., "flights to X with layover in Y"), focus on the destination country (X) and consider the layover country (Y) as additional context
            3. Select specific values from the available options that match the query intent
            4. Only include filters that are explicitly mentioned or strongly implied in the query
            5. Return a JSON object with the selected filters
            6. Use null for filters that are not applicable

            IMPORTANT: For queries about flights to a specific country, always set the "to_country" filter to that country.

            Example output format:
            
            {{
                "airline": null,
                "from_country": null, 
                "to_country": "Turkey",
                "travel_class": null,
                "max_price": null,
                "refundable": null,
                "baggage_included": null,
                "wifi_available": null,
                "meal_service": null,
                "aircraft_type": null
            }}
             
            Examples:
            
            - Query: "flights to Turkey with layover in London" → {{"to_country": "Turkey"}}
            - Query: "Emirates flights to Dubai" → {{"airline": "Emirates", "to_country": "UAE"}}
            - Query: "business class flights under $2000" → {{"travel_class": "business", "max_price": 2000}}

            User Query: {query}"""),
            ("human", "Generate filters for this query.")
        ])
        json_parser=JsonOutputParser()
        llm_instance=await get_gemini_llm()
        if llm_instance:
            try:
                chain=filter_prompt | llm_instance | json_parser
                filters=await asyncio.to_thread(
                    lambda: chain.invoke({
                        "query": query,
                        "filter_options": json.dumps(filter_options, indent=2)
                    })
                )
                
                cleaned_filters={k: v for k, v in filters.items() if v is not None}
                logger.info(f"Generated filters: {cleaned_filters}")
                return Command(goto="apply_hard_filters", update={"filters": cleaned_filters})
            except Exception as e:
                logger.error(f"Error generating filters with LLM: {e}")
                return Command(goto="apply_hard_filters", update={"filters": {}})
        else:
            logger.warning("LLM not available for filter generation")
            return Command(goto="apply_hard_filters", update={"filters": {}})
    except Exception as e:
        logger.error(f"Error in generate_filters: {e}", exc_info=True)
        return Command(goto="apply_hard_filters", update={"filters": {}})

#   Apply hard filters to the collection based on metadata and query.

async def apply_hard_filters(state: GraphState) -> Command[Literal["rerank_documents"]]:
    logger.info("Starting hard filter application")
    try:
        await initialize_components()
        collection_name=state["collection_name"]
        filters=state["filters"]
        query=state["query"]
        logger.info(f"Applying filters: {filters} to collection: {collection_name}")
        try:
            await ensure_filter_indexes(client, collection_name)
            logger.info("Ensured filter indexes exist")
        except Exception as e:
            logger.warning(f"Could not ensure filter indexes: {e}")
        try:
            sample_points, _=await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.scroll(
                    collection_name=collection_name,
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
            )
            if sample_points:
                sample_metadata=sample_points[0].payload
                logger.info(f"Sample document metadata keys: {list(sample_metadata.keys())}")
                logger.info(f"Sample document metadata: {sample_metadata}")
        except Exception as e:
            logger.warning(f"Could not get sample document: {e}")
        filter_conditions=[]
        if "airline" in filters:
            filter_conditions.append(
                FieldCondition(
                    key="airline",
                    match=MatchValue(value=filters["airline"])
                )
            )
        if "alliance" in filters:
            filter_conditions.append(
                FieldCondition(
                    key="alliance",
                    match=MatchValue(value=filters["alliance"])
                )
            )
        if "from_country" in filters:
            filter_conditions.append(
                FieldCondition(
                    key="from_country",
                    match=MatchValue(value=filters["from_country"])
                )
            )
        if "to_country" in filters:
            filter_conditions.append(
                FieldCondition(
                    key="to_country",
                    match=MatchValue(value=filters["to_country"])
                )
            )
        if "travel_class" in filters:
            filter_conditions.append(
                FieldCondition(
                    key="travel_class",
                    match=MatchValue(value=filters["travel_class"])
                )
            )
        if "max_price" in filters:
            filter_conditions.append(
                FieldCondition(
                    key="price_usd",
                    range=Range(lte=filters["max_price"])
                )
            )
        if "min_price" in filters:
            filter_conditions.append(
                FieldCondition(
                    key="price_usd",
                    range=Range(gte=filters["min_price"])
                )
            )
        if "refundable" in filters:
            filter_conditions.append(
                FieldCondition(
                    key="refundable",
                    match=MatchValue(value=filters["refundable"])
                )
            )
        if "baggage_included" in filters:
            filter_conditions.append(
                FieldCondition(
                    key="baggage_included",
                    match=MatchValue(value=filters["baggage_included"])
                )
            )
        if "wifi_available" in filters:
            filter_conditions.append(
                FieldCondition(
                    key="wifi_available",
                    match=MatchValue(value=filters["wifi_available"])
                )
            )
        if "meal_service" in filters:
            filter_conditions.append(
                FieldCondition(
                    key="meal_service",
                    match=MatchValue(value=filters["meal_service"])
                )
            )
        if "aircraft_type" in filters:
            filter_conditions.append(
                FieldCondition(
                    key="aircraft_type",
                    match=MatchValue(value=filters["aircraft_type"])
                )
            )
        if filter_conditions:
            if len(filter_conditions) > 1:
                filter_obj=Filter(should=filter_conditions)
            else:
                filter_obj=Filter(must=filter_conditions)
            logger.info(f"Created filter with {len(filter_conditions)} conditions: {filter_conditions}")
        else:
            filter_obj=None
            logger.info("No filter conditions created, will search without filters")
        original_store=await asyncio.to_thread(
            lambda: QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings,
                retrieval_mode=RetrievalMode.DENSE,
            )
        )

        #   First search with filters applied.

        retriever=original_store.as_retriever(
            search_kwargs={"k": 20, "filter": filter_obj}
        )
        logger.info(f"Searching with query: '{query}' and filter: {filter_obj}")
        filtered_docs=await retriever.ainvoke(query)
        if not filtered_docs:
            logger.warning(f"No documents found with filters: {filters}, trying without filters")
            retriever=original_store.as_retriever(search_kwargs={"k": 20})
            filtered_docs=await retriever.ainvoke(query)
            logger.info(f"Retrieved {len(filtered_docs)} documents without filters")
        else:
            logger.info(f"Retrieved {len(filtered_docs)} documents with filters")
        
        logger.info(f"Total documents retrieved: {len(filtered_docs)}")

        #   Logging the filtered documents for debugging.

        if filtered_docs:
            logger.info("First 3 documents retrieved:")
            for i, doc in enumerate(filtered_docs[:3]):
                logger.info(f"  Doc {i+1}: {doc.page_content[:100]}...")
                if hasattr(doc, 'metadata') and doc.metadata:
                    logger.info(f"    Metadata: {doc.metadata}")
        return Command(goto="rerank_documents", update={"filtered_docs": filtered_docs})
    except Exception as e:
        logger.error(f"Error in apply_hard_filters: {e}", exc_info=True)
        return Command(goto="rerank_documents", update={"filtered_docs": []})
    
#   Rerank the filtered documents using LLM reranker.

async def rerank_documents(state: GraphState) -> Command[Literal["merge_documents"]]:
    logger.info("Starting document reranking")
    try:
        filtered_docs=state["filtered_docs"]
        query=state["query"]
        if not filtered_docs:
            logger.warning("No documents to rerank.")
            return Command(goto="merge_documents", update={"reranked_docs": []})
        logger.info(f"Reranking {len(filtered_docs)} flight documents")
        
        #   Logging the original document order for debugging.

        logger.info("Original document order:")

        #   Logging the first 5 documents for context.

        for i, doc in enumerate(filtered_docs[:5]):
            logger.info(f"  Original {i+1}: {doc.page_content[:100]}...")
            if hasattr(doc, 'metadata') and doc.metadata:
                logger.info(f"    Metadata: {doc.metadata}")
        
        #   Using the LLM reranker – run in separate thread to avoid blocking.

        compressor=await asyncio.to_thread(
            lambda: RankLLMRerank(
                model="gpt", 
                gpt_model="gpt-4o-mini", 
                top_n=min(10, len(filtered_docs))
            )
        )
        
        reranked_docs=await compressor.acompress_documents(
            documents=filtered_docs,
            query=query
        )

        #   Logging the reranked documents for debugging.

        logger.info("Reranked document order:")

        #   Logging the first 5 reranked documents for context.

        for i, doc in enumerate(reranked_docs[:5]):
            logger.info(f"  Reranked {i+1}: {doc.page_content[:100]}...")
            if hasattr(doc, "metadata") and doc.metadata:
                logger.info(f"    Metadata: {doc.metadata}")
        logger.info(f"Reranked flight documents to {len(reranked_docs)} documents")
        return Command(goto="merge_documents", update={"reranked_docs": reranked_docs})
    except Exception as e:
        logger.error(f"Error in rerank_documents: {e}", exc_info=True)
        return Command(goto="merge_documents", update={"reranked_docs": []})
    
#   Generate the final answer based on the reranked documents and query.

async def generate_answer(state: GraphState) -> Command[Literal[END]]:
    logger.info("Starting answer generation")
    try:
        query=state["query"]
        reranked_docs=state["reranked_docs"]
        if not reranked_docs:
            logger.warning("No documents available for answer generation")
            return {"answer": "I couldn't find any relevant information to answer your query."}
        context="\n\n".join([doc.page_content for doc in reranked_docs])
        system_message=f"""You are a helpful assistant that answers questions based on the provided context.
        Context:
        {context}

        Please answer the following question based on the context above. If the context doesn't contain enough information to answer the question, say so. Be concise and accurate.

        Question: {query}"""
        llm_instance=await get_gemini_llm()
        if llm_instance:
            try:
                response=await asyncio.to_thread(
                    lambda: llm_instance.invoke([
                        SystemMessage(content=system_message),
                        HumanMessage(content=query)
                    ])
                )
                answer=response.content
            except Exception as e:
                logger.error(f"Error calling LLM: {e}")
                answer=f"Based on the {len(reranked_docs)} relevant documents found, here's what I can tell you about '{query}': [LLM generation failed]"
        else:
            answer=f"Based on the {len(reranked_docs)} relevant documents found, here's what I can tell you about '{query}': [LLM not available]"
        logger.info("Answer generation complete")
        return Command(goto=END, update={"answer": answer})
    except Exception as e:
        logger.error(f"Error in generate_answer: {e}", exc_info=True)
        return Command(goto=END, update={"answer": "Sorry, I encountered an error while generating the answer."})
    
#   Perform simple retrieval for info-only queries without hard filters.
    
async def simple_retrieval(state: GraphState) -> Command[Literal["merge_documents"]]:
    logger.info("Starting simple retrieval for info queries")
    try:
        await initialize_components()
        collection_name=state["collection_name"]
        query=state["query"]
        logger.info(f"Performing simple retrieval for query: '{query}'")
        original_store=await asyncio.to_thread(
            lambda: QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings,
                retrieval_mode=RetrievalMode.DENSE,
            )
        )
        retriever=original_store.as_retriever(search_kwargs={"k": 10})
        info_docs=await retriever.ainvoke(query)
        logger.info(f"Retrieved {len(info_docs)} documents from simple retrieval")
        return Command(goto="merge_documents", update={"info_docs": info_docs})
    except Exception as e:
        logger.error(f"Error in simple_retrieval: {e}", exc_info=True)
        return Command(goto="merge_documents", update={"info_docs": []})

#   Merge documents from both flight and info retrieval paths, reranking if needed.

async def merge_documents(state: GraphState) -> Command[Literal["generate_answer"]]:
    logger.info("Starting document merging")
    try:
        filtered_docs=state.get("filtered_docs", [])
        info_docs=state.get("info_docs", [])
        query_type=state.get("query_type", "both")
        query=state["query"]
        if query_type=="flight_only":
            merged_docs=filtered_docs
            logger.info(f"Flight-only query: using {len(merged_docs)} flight documents")
        elif query_type=="info_only":
            if info_docs:
                logger.info(f"Reranking {len(info_docs)} info documents")
                compressor=await asyncio.to_thread(
                    lambda: RankLLMRerank(
                        model="gpt", 
                        gpt_model="gpt-4o-mini", 
                        top_n=min(10, len(info_docs))
                    )
                )
                merged_docs=await compressor.acompress_documents(
                    documents=info_docs,
                    query=query
                )
                logger.info(f"Reranked info documents to {len(merged_docs)} documents")
            else:
                merged_docs=[]
                logger.info("No info documents to rerank")
        else:
            all_docs=filtered_docs+info_docs
            if all_docs:
                logger.info(f"Reranking combined {len(all_docs)} documents (flight and information)")
                compressor=await asyncio.to_thread(
                    lambda: RankLLMRerank(
                        model="gpt", 
                        gpt_model="gpt-4o-mini", 
                        top_n=min(15, len(all_docs))
                    )
                )
                merged_docs=await compressor.acompress_documents(
                    documents=all_docs,
                    query=query
                )
                logger.info(f"Reranked combined documents to {len(merged_docs)} documents")
            else:
                merged_docs=[]
                logger.info("No documents to rerank")
        return Command(goto="generate_answer", update={"reranked_docs": merged_docs})
    except Exception as e:
        logger.error(f"Error in merge_documents: {e}", exc_info=True)
        return Command(goto="generate_answer", update={"reranked_docs": []})

workflow=StateGraph(GraphState) #   Building the graph workflow.

#   Adding nodes to the workflow.

workflow.add_node("classify_query", classify_query)
workflow.add_node("generate_filters", generate_filters)
workflow.add_node("apply_hard_filters", apply_hard_filters)
workflow.add_node("rerank_documents", rerank_documents)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("simple_retrieval", simple_retrieval)
workflow.add_node("merge_documents", merge_documents)

#   Defining the workflow structure.

workflow.add_edge(START, "classify_query")

app=workflow.compile()  #   Compiling the workflow.

#   Loading filter options from the extracted data.

def get_filter_options():
    return {
        "airline": [
            "Aeromexico", "Air Canada", "Air France", "Alitalia", "All Nippon Airways",
            "American Airlines", "British Airways", "Cathay Pacific", "China Eastern",
            "Delta Air Lines", "Emirates", "Etihad Airways", "Finnair", "Iberia",
            "Japan Airlines", "JetBlue Airways", "KLM", "Korean Air", "Lufthansa",
            "Norwegian Air", "Qantas", "Qatar Airways", "Ryanair", "Singapore Airlines",
            "Southwest Airlines", "Spirit Airlines", "Swiss International", "Thai Airways",
            "Turkish Airlines", "United Airlines", "Vietnam Airlines", "Virgin Atlantic"
        ],
        "alliance": ["Non-Alliance", "OneWorld", "SkyTeam", "Star Alliance"],
        "from_country": [
            "Australia", "Canada", "Egypt", "France", "Germany", "Hong Kong", "India",
            "Italy", "Japan", "Netherlands", "Qatar", "Singapore", "South Korea",
            "Spain", "Thailand", "Turkey", "UAE", "UK", "USA"
        ],
        "to_country": [
            "Australia", "Canada", "Egypt", "France", "Germany", "Hong Kong", "India",
            "Italy", "Japan", "Netherlands", "Qatar", "Singapore", "South Korea",
            "Spain", "Thailand", "Turkey", "UAE", "UK", "USA"
        ],
        "travel_class": ["business", "economy", "first", "premium_economy"],
        "refundable": [False, True],
        "baggage_included": [False, True],
        "wifi_available": [False, True],
        "meal_service": ["meal", "none", "premium_meal", "snack"],
        "aircraft_type": [
            "Airbus A320", "Airbus A330", "Airbus A350", "Airbus A380",
            "Boeing 737", "Boeing 777", "Boeing 787"
        ],
        "price_ranges": {
            "min": 300,
            "max": 12430,
            "suggested_ranges": [
                {"min": 0, "max": 500, "label": "Budget (0-500 USD)"},
                {"min": 500, "max": 1000, "label": "Economy (500-1000 USD)"},
                {"min": 1000, "max": 2000, "label": "Mid-range (1000-2000 USD)"},
                {"min": 2000, "max": 5000, "label": "Premium (2000-5000 USD)"},
                {"min": 5000, "max": 12430, "label": "Luxury (5000+ USD)"}
            ]
        }
    }


#   Run the complete search and answer generation workflow with dynamic filter generation.

async def run_search_and_answer(
    query: str,
    collection_name: str
) -> Dict[str, Any]:
    initial_state={
        "query": query,
        "collection_name": collection_name,
        "query_type": "both",   #   Default to "both" until classified.
        "filters": {},
        "filter_options": get_filter_options(),
        "filtered_docs": [],
        "info_docs": [],
        "reranked_docs": [],
        "answer": ""
    }
    
    try:
        result=await app.ainvoke(initial_state)
        return result
    except Exception as e:
        logger.error(f"Error in run_search_and_answer: {e}", exc_info=True)
        return {"error": str(e)}