"""
Database service using SQLite for storing document metadata and text chunks.
No external database required — fully local.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional
import os


DB_PATH = os.path.join("db", "rag_system.db")


class DatabaseService:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def initialize(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript("""
                -- Documents table: stores metadata about each uploaded file
                CREATE TABLE IF NOT EXISTS documents (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename    TEXT NOT NULL,
                    file_type   TEXT NOT NULL,          -- 'pdf' | 'image'
                    language    TEXT NOT NULL,          -- 'en' | 'bn' | 'mixed'
                    doc_date    TEXT,                   -- ISO date string, user-supplied or inferred
                    page_count  INTEGER DEFAULT 1,
                    raw_text    TEXT,
                    ocr_engine  TEXT DEFAULT 'tesseract',
                    created_at  TEXT NOT NULL,
                    file_path   TEXT NOT NULL
                );

                -- Chunks table: text chunks derived from each document for embedding
                CREATE TABLE IF NOT EXISTS chunks (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text        TEXT NOT NULL,
                    char_start  INTEGER,
                    char_end    INTEGER,
                    language    TEXT,                   -- chunk-level language hint
                    UNIQUE(doc_id, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_documents_language ON documents(language);
                CREATE INDEX IF NOT EXISTS idx_documents_date ON documents(doc_date);
            """)
        print(f"✅ Database initialized at {self.db_path}")

    # ─── Documents ────────────────────────────────────────────────────────────

    def insert_document(
        self,
        filename: str,
        file_type: str,
        language: str,
        doc_date: Optional[str],
        page_count: int,
        raw_text: str,
        file_path: str,
        ocr_engine: str = "tesseract",
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO documents
                   (filename, file_type, language, doc_date, page_count, raw_text, ocr_engine, created_at, file_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    filename, file_type, language, doc_date,
                    page_count, raw_text, ocr_engine,
                    datetime.utcnow().isoformat(), file_path,
                ),
            )
            return cursor.lastrowid

    def get_all_documents(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, filename, file_type, language, doc_date, page_count, created_at FROM documents ORDER BY id DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_document(self, doc_id: int) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def delete_document(self, doc_id: int) -> bool:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            return True

    # ─── Chunks ───────────────────────────────────────────────────────────────

    def insert_chunks(self, doc_id: int, chunks: list[dict]):
        """chunks: list of {chunk_index, text, char_start, char_end, language}"""
        with self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO chunks
                   (doc_id, chunk_index, text, char_start, char_end, language)
                   VALUES (:doc_id, :chunk_index, :text, :char_start, :char_end, :language)""",
                [{"doc_id": doc_id, **c} for c in chunks],
            )

    def get_chunks_for_doc(self, doc_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id=? ORDER BY chunk_index", (doc_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_chunks(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT c.*, d.filename, d.language as doc_language,
                       d.doc_date, d.file_type
                FROM chunks c
                JOIN documents d ON c.doc_id = d.id
                ORDER BY c.doc_id, c.chunk_index
            """).fetchall()
            return [dict(r) for r in rows]

    def get_filtered_chunks(
        self,
        language: Optional[str] = None,
        doc_date_from: Optional[str] = None,
        doc_date_to: Optional[str] = None,
        file_type: Optional[str] = None,
        doc_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        """Return chunks filtered by metadata constraints."""
        conditions = []
        params = []

        if language and language != "all":
            conditions.append("d.language = ?")
            params.append(language)
        if doc_date_from:
            conditions.append("d.doc_date >= ?")
            params.append(doc_date_from)
        if doc_date_to:
            conditions.append("d.doc_date <= ?")
            params.append(doc_date_to)
        if file_type and file_type != "all":
            conditions.append("d.file_type = ?")
            params.append(file_type)
        if doc_ids:
            placeholders = ",".join("?" * len(doc_ids))
            conditions.append(f"c.doc_id IN ({placeholders})")
            params.extend(doc_ids)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT c.*, d.filename, d.language as doc_language,
                           d.doc_date, d.file_type
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.id
                    {where}
                    ORDER BY c.doc_id, c.chunk_index""",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        with self._connect() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            lang_breakdown = conn.execute(
                "SELECT language, COUNT(*) as cnt FROM documents GROUP BY language"
            ).fetchall()
            return {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "language_breakdown": {r["language"]: r["cnt"] for r in lang_breakdown},
            }
