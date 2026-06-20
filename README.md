# Assessment 3: Local OCR & Dynamic RAG System

A fully localized, privacy-first document processing pipeline with multilingual OCR
(Bangla + English) and a hybrid semantic search + metadata-filtered RAG system.

---

## Architecture Overview

```
+---------------------------------------------------------------------+
|                        FastAPI Backend                              |
|                                                                       |
|  +----------+   +--------------+   +----------------------+        |
|  |  Upload  |-->|  OCR Service |-->|  Chunker Service     |        |
|  |  Router  |   |  (Tesseract) |   |  (Sliding Window)    |        |
|  +----------+   +--------------+   +----------+-----------+        |
|                                                |                     |
|                       +------------------------v----------+          |
|                       |        SQLite Database             |          |
|                       |   documents + chunks tables        |          |
|                       +------------------------+----------+          |
|                                                |                     |
|  +----------+   +--------------+   +----------v-----------+        |
|  |  Search  |-->|  RAG Service |<--|  Vector Store        |        |
|  |  Router  |   |  (Gemini)    |   |  (ChromaDB +         |        |
|  +----------+   +--------------+   |  multilingual        |        |
|                                     |  sentence embeddings) |        |
|                                     +-----------------------+        |
+---------------------------------------------------------------------+
              |
   +----------v----------+
   |   HTML/JS UI        |   (served at http://localhost:8000)
   +----------------------+
```

---

## Technical Decisions

### OCR Engine: Tesseract 5 (LSTM mode)

**Why Tesseract?**
- Open-source, runs 100% locally — no data sent to external APIs.
- Tesseract 5's LSTM engine significantly outperforms the legacy v4 classifier on Bangla.
- Native `ben` (Bengali) language pack handles Unicode Bangla script.

**Language packs installed:**
```bash
apt-get install tesseract-ocr-ben   # Bangla
# eng is bundled by default
```

**OCR Configuration:**
```
--oem 1   LSTM neural net mode (most accurate in Tesseract 5)
--psm 3   Automatic page segmentation (handles multi-column layouts)
Language: ben+eng   combined mode for bilingual documents
```

**Trade-offs:**
| Feature              | Tesseract (this system) | Surya / GOT-OCR2.0 |
|----------------------|------------------------|--------------------|
| Bangla accuracy      | ~75-88% printed        | ~90-95%            |
| English accuracy     | ~92-96%                | ~95-98%            |
| RAM requirement      | ~200 MB                | 4-8 GB GPU         |
| Setup complexity     | `apt install`          | Docker + GPU       |
| External dependency  | None                   | None (local)       |
| Handwriting support  | Limited                | Good               |

For scanned PDFs without embedded text, the system falls back to image-based OCR
using `pdf2image` (requires `poppler-utils` installed) to render each page then OCR it.

**Baseline Performance (Tesseract 5, LSTM):**
- Clean printed English: ~94% character accuracy
- Clear printed Bangla: ~80% character accuracy (degrades on low-DPI < 200 DPI scans)
- Mixed documents: ~82% overall (script-switching mid-line is the weak point)

---

### Text Chunking Strategy

**Why sentence-aware sliding window with overlap?**

RAG quality depends critically on chunk boundaries not cutting through semantically
coherent sentences. The design:

1. **Sentence-aware splitting** using both Latin (`.!?`) and Bangla (`।`) terminators.
   Bangla uses the *daari* (।, U+0964) as its standard sentence terminator.

2. **Window size: 400 chars** — approximately 60-80 English words or 40-60 Bangla words.
   Chosen to balance context richness vs. retrieval precision.

3. **Overlap: 80 chars** — ensures cross-boundary sentences appear in both adjacent chunks,
   preventing a key fact from being lost because it straddles a chunk border.

4. **Minimum chunk length: 50 chars** — filters out OCR noise (stray characters, page
   numbers, watermarks) that would otherwise pollute the vector index.

---

### Embedding Model: ChromaDB + Multilingual Sentence Transformers

**Why ChromaDB + `paraphrase-multilingual-mpnet-base-v2`?**

| Criterion           | This system (dense embeddings) | Lexical/TF-IDF alternative |
|--------------------|----------------------------------|------------------------------|
| Bangla support      | Native — trained on 50+ languages | Possible via char n-grams    |
| Semantic similarity | True semantic (car ≈ automobile, even cross-language) | Lexical (exact/stem match only) |
| Setup               | `pip install chromadb sentence-transformers` | Pure Python, no deps |
| Model size           | ~970 MB download (one-time)     | 0 MB                          |
| Persistence          | ChromaDB persists to disk automatically | Custom implementation needed |
| Cross-language match | Yes — Bangla query can retrieve English chunk if meaning matches | No |

This is the standard, production-grade approach for multilingual RAG. ChromaDB is an
embedded vector database (no separate server process) that persists its index to
local disk under `db/chroma/` — fully local, no network calls after the model is
downloaded once.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
embedding = model.encode(text, normalize_embeddings=True)
```

---

### Hybrid Search Architecture: Metadata + Vector

The key architectural decision is applying metadata filters **before** vector scoring,
not as a post-hoc re-rank:

```
User query + filters
        |
        v
+--------------------+
|  DatabaseService   |  <- SQL WHERE clause (hard constraint)
|  .get_filtered_    |     language = 'bn'
|   chunks()         |     doc_date BETWEEN '2024-01-01' AND '2024-12-31'
+--------+-----------+     file_type = 'pdf'
         |
         v (filtered chunk pool -> set of eligible doc_ids)
+--------------------+
|  VectorStoreService|  <- ChromaDB query with a native 'where' clause
|  .search(query,    |     restricting search to ONLY those doc_ids
|  filtered_chunks)  |
+--------+-----------+
         |
         v top-k results
+--------------------+
|  RAG Service       |  <- Gemini API (gemini-2.5-flash)
|  .rag_query()      |     Context-stuffed prompt generation
+--------------------+
```

**Why filter before vector search?**
- Guarantees filter compliance: a chunk cannot appear in results if it fails metadata
  criteria, regardless of how high its semantic score is.
- Reduces search space: ChromaDB's native `where` clause applies the filter during
  the similarity search itself, not as a separate pass.
- Enables strict document-level constraints (e.g., "only search documents uploaded
  this month, in Bangla, from PDFs").

---

## API Reference

### POST /api/documents/upload
Upload and process a document.

**Form fields:**
- `file` (required): PDF, PNG, JPG, TIFF
- `doc_date` (optional): `YYYY-MM-DD`
- `language_hint` (optional): `en`, `bn`, `mixed`

**Response:**
```json
{
  "doc_id": 1,
  "filename": "report.pdf",
  "language": "mixed",
  "page_count": 4,
  "ocr_engine": "tesseract-ben+eng",
  "ocr_confidence": 87.3,
  "chunks_created": 12
}
```

### POST /api/search/query
RAG query with optional metadata filters.

**Request:**
```json
{
  "query": "What is the crop yield forecast for 2024?",
  "top_k": 5,
  "language": "en",
  "date_from": "2024-01-01",
  "date_to": "2024-12-31",
  "file_type": "pdf"
}
```

**Response:**
```json
{
  "answer": "Based on [Source 1]...",
  "sources": [{"filename": "report.pdf", "score": 0.847}],
  "chunks_retrieved": 5,
  "filters_applied": {"language": "en", "date_from": "2024-01-01"}
}
```

### POST /api/search/chunks
Raw chunk retrieval without LLM (debugging).

### GET /api/documents/
List all documents.

### DELETE /api/documents/{id}
Delete a document and rebuild the vector index.

### GET /api/admin/stats
System statistics.

---

## Setup & Running

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-ben poppler-utils

# macOS
brew install tesseract tesseract-lang poppler
```

### Installation
```bash
git clone <repo-url>
cd rag_ocr_system
pip install -r requirements.txt
```

### Set your Gemini API key
```bash
export GEMINI_API_KEY=your_actual_key_here
```
Get a free key at https://aistudio.google.com/apikey

### Run
```bash
python server.py
# or
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` — the UI is served from `/static/index.html`.
API docs at `http://localhost:8000/docs`.

**Note**: First run downloads the multilingual embedding model (~970 MB) from
HuggingFace. This requires internet access once; after that, everything (OCR,
embedding, vector search) runs fully offline.

### Run Tests
```bash
python tests/test_system.py
```

---

## Docker Setup

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-ben poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t rag-ocr-system .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key rag-ocr-system
```

---

## Database Schema

```sql
CREATE TABLE documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL,     -- 'pdf' | 'image'
    language    TEXT NOT NULL,     -- 'en' | 'bn' | 'mixed'
    doc_date    TEXT,              -- ISO date (user-supplied)
    page_count  INTEGER DEFAULT 1,
    raw_text    TEXT,
    ocr_engine  TEXT DEFAULT 'tesseract',
    created_at  TEXT NOT NULL,
    file_path   TEXT NOT NULL
);

CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    char_start  INTEGER,
    char_end    INTEGER,
    language    TEXT,
    UNIQUE(doc_id, chunk_index)
);
```

ChromaDB maintains its own separate vector index under `db/chroma/`, keyed by
`{doc_id}_{chunk_index}` IDs that map back to this SQLite schema.

---

## File Structure

```
rag_ocr_system/
├── app/
│   ├── main.py                 # FastAPI app, lifespan, router registration
│   ├── routers/
│   │   ├── documents.py        # Upload, list, delete endpoints
│   │   ├── search.py           # RAG query + raw chunk search
│   │   └── admin.py            # Stats, reindex
│   └── services/
│       ├── ocr_service.py      # Tesseract OCR (PDF + image)
│       ├── chunker.py          # Sliding window chunker (bilingual)
│       ├── vector_store.py     # ChromaDB + sentence-transformers
│       ├── database.py         # SQLite metadata store
│       └── rag_service.py      # End-to-end RAG pipeline (Gemini)
├── static/
│   └── index.html              # Single-page frontend
├── uploads/                    # Uploaded files (gitignored)
├── db/                         # SQLite + ChromaDB data (gitignored)
├── tests/
│   └── test_system.py          # Unit/integration tests
├── server.py                   # Entry point
├── requirements.txt
├── Dockerfile
└── README.md
```
