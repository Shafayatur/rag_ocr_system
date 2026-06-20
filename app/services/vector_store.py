"""
Vector Store Service

Embedding & retrieval strategy:
────────────────────────────────
We use ChromaDB (local, embedded vector database) with the
'paraphrase-multilingual-mpnet-base-v2' sentence-transformer model for embeddings.

Why this combination:
  ✅ ChromaDB runs fully embedded/local — no server, no network calls, persists to disk.
  ✅ paraphrase-multilingual-mpnet-base-v2 supports 50+ languages including Bangla,
     trained specifically so that semantically similar sentences in DIFFERENT languages
     land close together in vector space (critical for a bilingual Bangla/English corpus).
  ✅ Dense embeddings capture meaning, not just exact words — e.g. a query in English
     can retrieve a relevant Bangla chunk if the meaning matches.
  ✅ Both libraries are open-source, pip-installable, no API keys, no billing.

Trade-off vs lexical search (TF-IDF/BM25):
  - Dense embeddings are slower to compute (model forward pass) and need ~1GB RAM
    for the model in memory.
  - For very short / keyword-heavy queries, lexical methods can sometimes outperform
    dense embeddings. For natural-language questions (this assessment's use case),
    dense embeddings are the better fit.

Hybrid search architecture (metadata + vector):
  1. DatabaseService.get_filtered_chunks() applies hard SQL filters (language, date, type)
     BEFORE any vector math runs.
  2. VectorStoreService.search() restricts the ChromaDB query to only the filtered
     chunk IDs using a 'where' clause (ChromaDB's native metadata filtering).
  3. This guarantees manual filters are strict constraints, not a soft re-rank —
     a chunk that fails the filter is never considered, regardless of similarity score.
"""

import os
import logging
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CHROMA_PATH = os.path.join("db", "chroma")
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
COLLECTION_NAME = "document_chunks"


class VectorStoreService:
    """
    Persistent local vector store backed by ChromaDB + a multilingual
    sentence-transformer embedding model.
    """

    def __init__(self):
        os.makedirs(CHROMA_PATH, exist_ok=True)

        # ChromaDB persistent client — writes to local disk, no server needed
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Load the multilingual embedding model once at startup (kept in memory)
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'... (first run downloads ~1GB)")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded.")

    def load_from_db(self, db):
        """
        On startup, ChromaDB already persists to disk, so nothing needs to be
        rebuilt in memory — this method exists for interface parity / future use
        (e.g. could verify DB and Chroma collection counts match).
        """
        count = self.collection.count()
        logger.info(f"Vector store ready: {count} chunks already indexed in ChromaDB.")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into dense embedding vectors."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # required for cosine similarity to behave correctly
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def add_chunks(self, new_chunks: list[dict], db=None):
        """
        Embed and insert new chunks into ChromaDB.

        Args:
            new_chunks: list of dicts with keys:
                doc_id, chunk_index, text, language (chunk lang),
                filename, doc_language, doc_date, file_type
        """
        if not new_chunks:
            return

        ids = [f"{c['doc_id']}_{c['chunk_index']}" for c in new_chunks]
        texts = [c["text"] for c in new_chunks]
        embeddings = self._embed(texts)

        metadatas = [
            {
                "doc_id": c["doc_id"],
                "chunk_index": c["chunk_index"],
                "filename": c.get("filename", ""),
                "language": c.get("doc_language", c.get("language", "")),
                "doc_date": c.get("doc_date") or "",
                "file_type": c.get("file_type", ""),
            }
            for c in new_chunks
        ]

        # Chroma's upsert handles both insert and update cleanly
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"Indexed {len(new_chunks)} chunks into ChromaDB.")

    def _build_where_clause(self, filtered_chunks: Optional[list[dict]]) -> Optional[dict]:
        """
        Convert a pre-filtered chunk list (from DatabaseService.get_filtered_chunks)
        into a ChromaDB 'where' clause restricting search to those exact chunk IDs.
        This is how metadata filtering and vector search compose: SQL decides WHICH
        chunks are eligible, Chroma's where-clause enforces that same restriction
        natively during the similarity search itself.
        """
        if filtered_chunks is None:
            return None
        if not filtered_chunks:
            return {"doc_id": {"$in": []}}  # forces zero results

        doc_ids = list({c["doc_id"] for c in filtered_chunks})
        return {"doc_id": {"$in": doc_ids}}

    def search(
        self,
        query: str,
        top_k: int = 5,
        filtered_chunks: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Semantic search over (optionally pre-filtered) chunks using dense embeddings.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            filtered_chunks: If provided (from DatabaseService metadata filter),
                             restrict search to only these chunks' doc_ids.

        Returns:
            List of result dicts sorted by similarity score descending.
        """
        if self.collection.count() == 0:
            return []

        where_clause = self._build_where_clause(filtered_chunks)
        query_embedding = self._embed([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where=where_clause,
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        output = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # Chroma returns cosine DISTANCE (0=identical); convert to similarity score
            similarity = 1 - distance

            output.append({
                "doc_id": metadata["doc_id"],
                "chunk_index": metadata["chunk_index"],
                "filename": metadata["filename"],
                "doc_language": metadata["language"],
                "text": results["documents"][0][i],
                "score": round(similarity, 4),
            })

        return output
