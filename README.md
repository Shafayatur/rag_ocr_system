# Local OCR & Dynamic RAG System

A fully localized, privacy-first document processing pipeline with multilingual OCR
(Bangla + English) and a hybrid semantic search + metadata-filtered RAG system.

> **Fully local by default.** OCR (Tesseract), embeddings (sentence-transformers),
> vector search (ChromaDB), and answer generation (Ollama) all run on-device. No
> document content is sent to any external commercial API in the default
> configuration. An optional, explicitly opt-in cloud LLM fallback exists for
> low-resource machines (see "Answer Generation" below) but is off by default.

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
|  |  Router  |   |  (Ollama,    |   |  (ChromaDB +         |        |
|  +----------+   |   local LLM) |   |  multilingual        |        |
|                  +--------------+   |  sentence embeddings) |        |
|                                     +-----------------------+        |
+---------------------------------------------------------------------+
              |
   +----------v----------+
   |   HTML/JS UI        |   (served at http://localhost:8000)
   +----------------------+
```

Every stage — OCR, embedding, vector search, and answer generation — runs on
localhost. The only one-time network calls are downloading the embedding model
weights from HuggingFace and the LLM weights via `ollama pull`; after that, the
system serves queries with zero outbound network calls, satisfying the "no
external commercial APIs" requirement end-to-end (not just for OCR).

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

### Answer Generation: Local LLM via Ollama

**Why a local LLM instead of a cloud API?**

The assessment requires that documents be processed "without sending data to
external commercial APIs." OCR and embeddings already satisfy this, but the final
answer-generation step also needs to stay local — otherwise the retrieved chunk
text (i.e. private document content) would leave the machine on every query. This
system therefore defaults to **Ollama**, a local model server, for that step.

```bash
# One-time setup
ollama pull qwen2.5:3b       # ~2 GB download, runs comfortably on CPU
ollama serve                  # starts the local server on :11434
```

No `GEMINI_API_KEY` or any other cloud credential is required for the default
configuration. The backend is configurable via environment variables:

| Variable          | Default                  | Purpose                              |
| ------------------ | ------------------------ | ------------------------------------- |
| `LLM_BACKEND`       | `ollama`                 | `ollama` (local) or `gemini` (cloud)  |
| `OLLAMA_BASE_URL`   | `http://localhost:11434` | Local Ollama server address           |
| `OLLAMA_MODEL`      | `qwen2.5:3b`              | Any pulled Ollama model               |

**Why `qwen2.5:3b` specifically?** It was chosen over larger models (e.g. an 8B
Llama variant) deliberately for hardware reach: at ~2GB it runs comfortably on
CPU-only laptops without the system becoming unresponsive, while an 8B model
can spike memory/CPU hard enough on modest hardware to feel like a freeze.
Qwen's multilingual training also tends to handle Bangla reasonably well for
its size. `OLLAMA_MODEL` is fully swappable — on a machine with more RAM/a
GPU, a larger model (e.g. `llama3.1:8b`, `qwen2.5:7b`) will generally produce
stronger, more nuanced answers, especially on Bangla synthesis.

**Trade-offs vs. cloud APIs:**

| Factor              | Local (Ollama, this system) | Cloud (Gemini, optional opt-in) |
| -------------------- | ---------------------------- | --------------------------------- |
| Data privacy         | Document text never leaves the machine | Retrieved chunks sent to Google |
| Setup                | `ollama pull` + local RAM/CPU/GPU | API key only |
| Latency (CPU-only)   | Slower (~3-10s for a 3B model) | Faster (~1-2s) |
| Answer quality       | Good for a small 3B instruction model; noticeably weaker than frontier cloud models, especially on nuanced Bangla. Swapping in a larger Ollama model narrows this gap at the cost of speed/RAM. | Generally stronger |
| Cost                 | Free after download | Per-token billing |

A cloud fallback (`LLM_BACKEND=gemini`) is left in the code, explicitly opt-in,
for cases where the deployment machine can't comfortably run a local model — but
the default behavior, and what's demonstrated in the demo video, is fully local
end-to-end.

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
|  RAG Service       |  <- Local LLM via Ollama (default: qwen2.5:3b)
|  .rag_query()      |     Context-stuffed prompt generation, fully on-device
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

# Install Ollama (local LLM server) — https://ollama.com/download
# Linux:
curl -fsSL https://ollama.com/install.sh | sh
# macOS: download the app from ollama.com, or `brew install ollama`
```

### Installation
```bash
git clone <repo-url>
cd rag_ocr_system
pip install -r requirements.txt
```

### Pull and start the local LLM
```bash
ollama pull qwen2.5:3b    # one-time download, ~2 GB
ollama serve              # starts the local server on :11434 (default backend)
```

No cloud API key is required for the default setup. If your machine can't run a
local model comfortably, you can opt into the cloud fallback instead:
```bash
export LLM_BACKEND=gemini
export GEMINI_API_KEY=your_actual_key_here   # https://aistudio.google.com/apikey
```

### Run
```bash
python server.py
# or
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` — the UI is served from `/static/index.html`.
API docs at `http://localhost:8000/docs`.

**Note**: First run downloads the multilingual embedding model (~970 MB) from
HuggingFace, and `ollama pull` downloads the LLM weights (~4.7 GB) once. Both
require internet access only for that initial download; after that, everything
(OCR, embedding, vector search, and answer generation) runs fully offline.

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
```

Since the LLM now runs locally via Ollama, the recommended way to run this is
`docker-compose`, which starts an Ollama service alongside the app container so
the app can reach it on the Docker network (see `docker-compose.yml`):

```bash
docker compose up --build
# first run only: pull the model into the ollama container
docker compose exec ollama ollama pull qwen2.5:3b
```

This brings up two containers — `app` (FastAPI + Tesseract + ChromaDB) and
`ollama` (the local LLM server) — with no traffic to any external API at
runtime. If you'd rather run the app container standalone against an Ollama
instance already running on the host:

```bash
docker run -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  rag-ocr-system
```

Or, to use the optional cloud fallback instead of a local model:
```bash
docker run -p 8000:8000 -e LLM_BACKEND=gemini -e GEMINI_API_KEY=your_key rag-ocr-system
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
│       └── rag_service.py      # End-to-end RAG pipeline (local Ollama LLM, optional Gemini fallback)
├── static/
│   └── index.html              # Single-page frontend
├── uploads/                    # Uploaded files (gitignored)
├── db/                         # SQLite + ChromaDB data (gitignored)
├── tests/
│   └── test_system.py          # Unit/integration tests
├── server.py                   # Entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml           # app + local Ollama service
└── README.md
```
