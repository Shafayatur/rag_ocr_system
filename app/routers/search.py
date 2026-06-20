"""
Search & RAG Router

Exposes two endpoints:
  POST /api/search/query  — Full RAG: metadata filter → vector retrieval → LLM answer
  POST /api/search/chunks — Raw chunk retrieval (no LLM, for debugging/inspection)
"""

import logging
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from app.services.rag_service import rag_query

logger = logging.getLogger(__name__)

router = APIRouter()


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language question")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")

    # ── Metadata filters (all optional) ──────────────────────────────────────
    language: Optional[str] = Field(
        None,
        description="Filter by document language: 'en', 'bn', 'mixed', or 'all'",
    )
    date_from: Optional[str] = Field(
        None, description="Filter docs with date >= YYYY-MM-DD"
    )
    date_to: Optional[str] = Field(
        None, description="Filter docs with date <= YYYY-MM-DD"
    )
    file_type: Optional[str] = Field(
        None, description="Filter by file type: 'pdf', 'image', or 'all'"
    )


class ChunkSearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(5, ge=1, le=20)
    language: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    file_type: Optional[str] = None


@router.post("/query", summary="RAG query: natural language question with optional metadata filters")
def rag_search(body: RAGQueryRequest, request: Request):
    """
    Hybrid RAG search pipeline:

    1. **Metadata filtering** (hard constraint, applied first in SQL):
       - `language`: restricts to docs written in 'en', 'bn', or 'mixed'.
       - `date_from` / `date_to`: restricts to docs within a date range.
       - `file_type`: restricts to 'pdf' or 'image' docs.

    2. **Semantic vector search** (ChromaDB + multilingual sentence embeddings):
       - Searches only within the metadata-filtered chunk pool.
       - Returns top_k most relevant chunks.

    3. **LLM Answer Generation** (gemini-2.5-flash):
       - Retrieved chunks are passed as context.
       - Model generates a grounded, cited answer.
    """
    db = request.app.state.db
    vector_store = request.app.state.vector_store

    # Normalise "all" filter values to None (no filter)
    language = body.language if body.language and body.language != "all" else None
    file_type = body.file_type if body.file_type and body.file_type != "all" else None

    try:
        result = rag_query(
            query=body.query,
            db=db,
            vector_store=vector_store,
            top_k=body.top_k,
            language_filter=language,
            date_from=body.date_from,
            date_to=body.date_to,
            file_type_filter=file_type,
        )
        return result
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chunks", summary="Raw chunk retrieval (no LLM generation)")
def chunk_search(body: ChunkSearchRequest, request: Request):
    """
    Retrieve relevant chunks without calling the LLM. Useful for:
    - Inspecting what the retrieval pipeline returns.
    - Debugging filter behaviour.
    - Building custom frontends that handle their own generation.
    """
    db = request.app.state.db
    vector_store = request.app.state.vector_store

    language = body.language if body.language and body.language != "all" else None
    file_type = body.file_type if body.file_type and body.file_type != "all" else None

    # Apply metadata filters
    if any([language, body.date_from, body.date_to, file_type]):
        filtered_chunks = db.get_filtered_chunks(
            language=language,
            doc_date_from=body.date_from,
            doc_date_to=body.date_to,
            file_type=file_type,
        )
    else:
        filtered_chunks = None

    results = vector_store.search(body.query, top_k=body.top_k, filtered_chunks=filtered_chunks)

    return {
        "query": body.query,
        "chunks_found": len(results),
        "filters": {
            "language": language,
            "date_from": body.date_from,
            "date_to": body.date_to,
            "file_type": file_type,
        },
        "results": [
            {
                "doc_id": c["doc_id"],
                "filename": c["filename"],
                "chunk_index": c["chunk_index"],
                "language": c.get("doc_language"),
                "doc_date": c.get("doc_date"),
                "score": c["score"],
                "text": c["text"],
            }
            for c in results
        ],
    }
