"""
RAG (Retrieval-Augmented Generation) Query Service

Pipeline:
  1. Apply metadata filters via DatabaseService (hard SQL constraints).
  2. Retrieve top-k semantically relevant chunks via VectorStoreService (ChromaDB).
  3. Build a context-stuffed prompt with the retrieved chunks.
  4. Generate a grounded answer via a LOCAL LLM served by Ollama (default), so that
     retrieved document text never leaves the machine.

The RAG approach ensures answers are grounded in the uploaded documents rather
than relying on the model's parametric knowledge. Each answer cites the source
document and chunk it was drawn from.

Hybrid search flow:
  metadata_filter(SQL) → filtered_chunk_pool → vector_search(ChromaDB) → top_k → LLM

LLM backend:
  Default backend is Ollama (https://ollama.com), running fully on localhost with
  no network calls — this keeps the entire pipeline (OCR -> embedding -> retrieval
  -> generation) local, per the assessment's "no external commercial APIs" requirement.

  An optional cloud fallback (Google Gemini) is supported behind an explicit opt-in
  env var (LLM_BACKEND=gemini) for environments without enough RAM/CPU to run a local
  model comfortably. This is OFF by default.
"""

import os
import json
import logging
from typing import Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# ── Backend selection ────────────────────────────────────────────────────────
# "ollama" (default, fully local) or "gemini" (explicit opt-in, sends context to
# Google's API — only use this if you cannot run a local model).
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
MAX_CONTEXT_CHARS = 6000   # max chars of retrieved context to send to LLM


def _call_ollama(system_prompt: str, user_message: str) -> str:
    """Call a locally-running Ollama server and return the text response.

    Requires Ollama installed and running (`ollama serve`) with a model pulled,
    e.g. `ollama pull llama3.1:8b`. No data leaves the machine — this satisfies
    the assessment's requirement that no document content be sent to external
    commercial APIs.
    """
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }).encode("utf-8")

    url = f"{OLLAMA_BASE_URL}/api/chat"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        # Local generation can be slow on CPU-only machines; allow a generous timeout.
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["message"]["content"]
    except urllib.error.URLError as e:
        logger.error(f"Could not reach local Ollama server at {OLLAMA_BASE_URL}: {e}")
        raise RuntimeError(
            f"Could not reach local Ollama server at {OLLAMA_BASE_URL}. "
            f"Is it running? Start it with `ollama serve` and ensure the model "
            f"'{OLLAMA_MODEL}' is pulled (`ollama pull {OLLAMA_MODEL}`)."
        )
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        raise


def _call_gemini(system_prompt: str, user_message: str) -> str:
    """Call the Google Gemini API and return the text response.

    NOTE: this sends retrieved document context to an external commercial API.
    Only used if LLM_BACKEND=gemini is explicitly set; the default backend is
    local Ollama (see _call_ollama above).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    # Gemini doesn't have a separate system role in this endpoint version,
    # so we prepend the system instructions to the user message.
    full_prompt = f"{system_prompt}\n\n---\n\n{user_message}"

    payload = json.dumps({
        "contents": [{
            "parts": [{"text": full_prompt}]
        }]
    }).encode("utf-8")

    url = f"{GEMINI_API_URL}?key={api_key}"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logger.error(f"Gemini API error {e.code}: {body}")
        raise RuntimeError(f"LLM API error {e.code}: {body}")
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


def _call_llm(system_prompt: str, user_message: str) -> str:
    """Dispatch to the configured LLM backend (local by default)."""
    if LLM_BACKEND == "gemini":
        return _call_gemini(system_prompt, user_message)
    return _call_ollama(system_prompt, user_message)


def rag_query(
    query: str,
    db,
    vector_store,
    top_k: int = 5,
    # Metadata filters
    language_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    file_type_filter: Optional[str] = None,
) -> dict:
    """
    Execute a full RAG pipeline query.

    Args:
        query: Natural language question from the user.
        db: DatabaseService instance.
        vector_store: VectorStoreService instance.
        top_k: Number of chunks to retrieve.
        language_filter: 'en', 'bn', 'mixed', or None.
        date_from / date_to: ISO date strings for document date range filter.
        file_type_filter: 'pdf', 'image', or None.

    Returns:
        {
          "answer": str,
          "sources": [list of source chunk references],
          "query": str,
          "filters_applied": dict,
          "chunks_retrieved": int,
        }
    """

    # ── Step 1: Apply metadata filters ──────────────────────────────────────
    filters_applied = {}
    has_filters = any([language_filter, date_from, date_to, file_type_filter])

    if has_filters:
        filtered_chunks = db.get_filtered_chunks(
            language=language_filter,
            doc_date_from=date_from,
            doc_date_to=date_to,
            file_type=file_type_filter,
        )
        if language_filter and language_filter != "all":
            filters_applied["language"] = language_filter
        if date_from:
            filters_applied["date_from"] = date_from
        if date_to:
            filters_applied["date_to"] = date_to
        if file_type_filter and file_type_filter != "all":
            filters_applied["file_type"] = file_type_filter
    else:
        filtered_chunks = None  # search entire corpus

    # ── Step 2: Vector retrieval ─────────────────────────────────────────────
    results = vector_store.search(query, top_k=top_k, filtered_chunks=filtered_chunks)

    if not results:
        return {
            "answer": (
                "No relevant content found in the documents for your query"
                + (" with the applied filters." if filters_applied else ".")
                + " Try uploading documents first or broadening your filters."
            ),
            "sources": [],
            "query": query,
            "filters_applied": filters_applied,
            "chunks_retrieved": 0,
        }

    # ── Step 3: Build context ────────────────────────────────────────────────
    context_parts = []
    sources = []
    total_chars = 0

    for i, chunk in enumerate(results):
        chunk_text = chunk["text"]
        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break
        if len(chunk_text) > remaining:
            chunk_text = chunk_text[:remaining] + "…"

        context_parts.append(
            f"[Source {i+1}: '{chunk['filename']}', lang={chunk.get('doc_language','?')}, "
            f"score={chunk['score']:.3f}]\n{chunk_text}"
        )
        sources.append({
            "source_index": i + 1,
            "doc_id": chunk["doc_id"],
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "language": chunk.get("doc_language"),
            "score": chunk["score"],
            "snippet": chunk["text"][:200] + ("…" if len(chunk["text"]) > 200 else ""),
        })
        total_chars += len(chunk_text)

    context = "\n\n---\n\n".join(context_parts)

    # ── Step 4: Generate answer via LLM ─────────────────────────────────────
    system_prompt = (
        "You are a precise document QA assistant. You answer questions ONLY based on "
        "the provided document excerpts. The documents may be in English, Bangla, or both.\n\n"
        "Rules:\n"
        "- Answer using ONLY information present in the excerpts.\n"
        "- Cite your source numbers (e.g., [Source 1]) when referencing content.\n"
        "- If the answer cannot be found in the excerpts, say so clearly.\n"
        "- If relevant text is in Bangla, you may translate it in your answer.\n"
        "- Be concise and factual."
    )

    user_message = (
        f"Document excerpts:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}"
    )

    try:
        answer = _call_llm(system_prompt, user_message)
    except Exception as e:
        answer = (
            f"⚠️ LLM generation failed: {e}\n\n"
            f"Retrieved {len(results)} relevant chunks. Top match from '{results[0]['filename']}' "
            f"(score={results[0]['score']:.3f}):\n\n{results[0]['text'][:500]}"
        )

    return {
        "answer": answer,
        "sources": sources,
        "query": query,
        "filters_applied": filters_applied,
        "chunks_retrieved": len(results),
    }