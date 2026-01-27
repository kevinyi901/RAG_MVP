"""FastAPI backend for RAG system."""

import os
import json
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from rag import EmbeddingService, Retriever, BM25Ranker, RAGPipeline


# Global instances
embedding_service: EmbeddingService = None
retriever: Retriever = None
ranker: BM25Ranker = None
pipeline: RAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global embedding_service, retriever, ranker, pipeline

    embedding_service = EmbeddingService()
    retriever = Retriever()
    ranker = BM25Ranker()
    pipeline = RAGPipeline(
        embedding_service=embedding_service,
        retriever=retriever,
        ranker=ranker
    )

    yield

    # Cleanup
    pipeline.close()


app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[int]] = None


class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = None
    document_ids: Optional[List[int]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    chain_of_thought: str
    query: str


class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: Optional[str]
    file_size: Optional[int]
    page_count: Optional[int]
    chunk_count: int
    created_at: str


class FeedbackRequest(BaseModel):
    query_id: Optional[int] = None
    query_text: str
    response_text: str
    sources: List[dict]
    chain_of_thought: Optional[str] = None
    feedback: str  # 'helpful', 'neutral', 'not_helpful'


class HealthResponse(BaseModel):
    status: str
    database: str
    ollama: str
    total_documents: int
    total_chunks: int


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    import httpx

    db_status = "healthy"
    ollama_status = "healthy"
    total_docs = 0
    total_chunks = 0

    # Check database
    try:
        docs = retriever.get_all_documents()
        total_docs = len(docs)
        total_chunks = retriever.get_total_chunks()
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    # Check Ollama
    try:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{ollama_host}/api/tags")
            response.raise_for_status()
    except Exception as e:
        ollama_status = f"unhealthy: {str(e)}"

    return HealthResponse(
        status="healthy" if db_status == "healthy" and ollama_status == "healthy" else "degraded",
        database=db_status,
        ollama=ollama_status,
        total_documents=total_docs,
        total_chunks=total_chunks
    )


# Document endpoints
@app.get("/api/documents", response_model=List[DocumentResponse])
async def list_documents():
    """List all uploaded documents."""
    try:
        docs = retriever.get_all_documents()
        return [
            DocumentResponse(
                id=d['id'],
                filename=d['filename'],
                file_type=d['file_type'],
                file_size=d['file_size'],
                page_count=d['page_count'],
                chunk_count=d['chunk_count'],
                created_at=str(d['created_at'])
            )
            for d in docs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: int):
    """Delete a document and its chunks."""
    try:
        deleted = retriever.delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "deleted", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Query endpoints
@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute a RAG query."""
    try:
        result = pipeline.query(
            question=request.question,
            document_ids=request.document_ids
        )
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            chain_of_thought=result.chain_of_thought,
            query=result.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """Execute a RAG query with streaming response."""

    def generate():
        try:
            for event in pipeline.query_streaming(
                question=request.question,
                document_ids=request.document_ids
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# Chat endpoint with conversation history
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Execute a RAG chat with conversation history and streaming response."""

    def generate():
        try:
            # Convert pydantic models to dicts for pipeline
            history = None
            if request.conversation_history:
                history = [{"role": m.role, "content": m.content} for m in request.conversation_history]

            for event in pipeline.chat_streaming(
                message=request.message,
                conversation_history=history,
                document_ids=request.document_ids
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# Feedback endpoint (placeholder - stores to database)
@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a query response."""
    try:
        with retriever.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_history
                    (query_text, response_text, sources, chain_of_thought, feedback)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    request.query_text,
                    request.response_text,
                    json.dumps(request.sources),
                    request.chain_of_thought,
                    request.feedback
                )
            )
            query_id = cur.fetchone()[0]
            retriever.conn.commit()

        return {"status": "recorded", "query_id": query_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Stats endpoint
@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    try:
        docs = retriever.get_all_documents()
        total_chunks = retriever.get_total_chunks()

        return {
            "total_documents": len(docs),
            "total_chunks": total_chunks,
            "documents": [
                {
                    "id": d['id'],
                    "filename": d['filename'],
                    "chunk_count": d['chunk_count']
                }
                for d in docs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
