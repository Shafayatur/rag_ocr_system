"""Admin router for system stats and management."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/stats", summary="System statistics")
def get_stats(request: Request):
    db = request.app.state.db
    vs = request.app.state.vector_store
    stats = db.get_stats()
    stats["vector_index_size"] = vs.collection.count()
    stats["embedding_model"] = "paraphrase-multilingual-mpnet-base-v2"
    return stats


@router.post("/reindex", summary="Rebuild vector index from DB")
def reindex(request: Request):
    db = request.app.state.db
    vs = request.app.state.vector_store
    vs.load_from_db(db)
    return {
        "success": True,
        "message": f"Vector index has {vs.collection.count()} chunks indexed.",
    }