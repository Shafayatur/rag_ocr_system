"""
Documents Router
Handles file upload, OCR processing, chunking, and indexing.
"""

import os
import shutil
import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse

from app.services.ocr_service import ocr_file
from app.services.chunker import chunk_text

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
MAX_FILE_SIZE_MB = 20


@router.post("/upload", summary="Upload and process a document")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    doc_date: Optional[str] = Form(None, description="Document date (YYYY-MM-DD)"),
    language_hint: Optional[str] = Form(None, description="Language hint: en, bn, mixed"),
):
    """
    Upload a scanned document or PDF.

    The pipeline:
    1. Validate file type and size.
    2. Save to uploads/ directory.
    3. Run local Tesseract OCR (Bangla + English).
    4. Chunk the extracted text.
    5. Store metadata in SQLite and update the vector index.
    """
    db = request.app.state.db
    vector_store = request.app.state.vector_store

    # Validate extension
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read and size-check
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB.",
        )

    # Save file
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(file_path, "wb") as f:
        f.write(content)

    logger.info(f"Saved '{safe_name}' ({size_mb:.2f} MB). Starting OCR...")

    # Run OCR
    try:
        ocr_result = ocr_file(file_path, filename, language_hint=language_hint)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    raw_text = ocr_result.get("text", "")
    detected_lang = ocr_result.get("language", "en")
    page_count = ocr_result.get("page_count", 1)
    ocr_engine = ocr_result.get("engine", "tesseract")
    ocr_confidence = ocr_result.get("confidence", 0.0)

    if not raw_text:
        logger.warning(f"OCR produced no text for '{filename}'")

    # Validate doc_date
    valid_date = None
    if doc_date:
        try:
            datetime.strptime(doc_date, "%Y-%m-%d")
            valid_date = doc_date
        except ValueError:
            pass  # ignore malformed dates

    # Store document in DB
    doc_id = db.insert_document(
        filename=filename,
        file_type=ocr_result.get("file_type", "unknown"),
        language=detected_lang,
        doc_date=valid_date,
        page_count=page_count,
        raw_text=raw_text,
        file_path=file_path,
        ocr_engine=ocr_engine,
    )

    # Chunk the text
    chunks = chunk_text(raw_text, doc_language=detected_lang)
    if chunks:
        db.insert_chunks(doc_id, chunks)
        # Attach document-level metadata so the vector store can filter by it
        enriched_chunks = [
            {
                **c,
                "doc_id": doc_id,
                "filename": filename,
                "doc_language": detected_lang,
                "doc_date": valid_date,
                "file_type": ocr_result.get("file_type", "unknown"),
            }
            for c in chunks
        ]
        vector_store.add_chunks(enriched_chunks, db)
        logger.info(f"Doc {doc_id}: {len(chunks)} chunks indexed.")

    return {
        "success": True,
        "doc_id": doc_id,
        "filename": filename,
        "file_type": ocr_result.get("file_type"),
        "language": detected_lang,
        "page_count": page_count,
        "ocr_engine": ocr_engine,
        "ocr_confidence": ocr_confidence,
        "ocr_method": ocr_result.get("method"),
        "text_length": len(raw_text),
        "chunks_created": len(chunks),
        "doc_date": valid_date,
        "message": (
            f"Document processed successfully. "
            f"{len(chunks)} chunks indexed for RAG search."
        ),
    }


@router.get("/", summary="List all uploaded documents")
def list_documents(request: Request):
    db = request.app.state.db
    docs = db.get_all_documents()
    return {"documents": docs, "total": len(docs)}


@router.get("/{doc_id}", summary="Get document details and chunks")
def get_document(doc_id: int, request: Request):
    db = request.app.state.db
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
    chunks = db.get_chunks_for_doc(doc_id)
    # Don't return raw_text inline (can be large); return chunk count instead
    doc_out = {k: v for k, v in doc.items() if k != "raw_text"}
    doc_out["text_length"] = len(doc.get("raw_text") or "")
    return {"document": doc_out, "chunks": chunks, "chunk_count": len(chunks)}


@router.delete("/{doc_id}", summary="Delete a document and its chunks")
def delete_document(doc_id: int, request: Request):
    db = request.app.state.db
    vector_store = request.app.state.vector_store

    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")

    # Remove file
    try:
        if os.path.exists(doc["file_path"]):
            os.remove(doc["file_path"])
    except Exception as e:
        logger.warning(f"Could not remove file {doc['file_path']}: {e}")

    db.delete_document(doc_id)

    # Rebuild vector index without deleted doc
    vector_store.load_from_db(db)

    return {"success": True, "deleted_doc_id": doc_id}
