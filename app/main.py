"""
Local OCR & Dynamic RAG System — Assessment 3
FastAPI backend for multilingual (Bangla + English) document processing,
vector-based semantic search, and metadata-filtered RAG.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from app.routers import documents, search, admin
from app.services.vector_store import VectorStoreService
from app.services.database import DatabaseService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    # Ensure upload directory exists
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("db", exist_ok=True)

    # Initialize database (SQLite for metadata)
    db = DatabaseService()
    db.initialize()

    # Initialize vector store (ChromaDB + multilingual sentence-transformer embeddings;
    # see app/services/vector_store.py for details)
    vs = VectorStoreService()
    vs.load_from_db(db)

    # Attach to app state
    app.state.db = db
    app.state.vector_store = vs

    print("✅ Services initialized. RAG system ready.")
    yield

    # Cleanup
    print("🔴 Shutting down.")


app = FastAPI(
    title="Local OCR & RAG System",
    description=(
        "A fully localized document processing pipeline supporting Bangla & English OCR, "
        "semantic vector search, and dynamic metadata-filtered RAG queries."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the frontend UI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Register routers
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(search.router, prefix="/api/search", tags=["Search & RAG"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])


@app.get("/", tags=["Health"], include_in_schema=False)
def root():
    """Serve the frontend UI."""
    return FileResponse("static/index.html")


@app.get("/status", tags=["Health"])
def status():
    return {
        "status": "running",
        "message": "Local OCR & RAG System — Assessment 3",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}