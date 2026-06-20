"""
Text Chunking Service

Chunking strategy for bilingual (Bangla + English) documents:
─────────────────────────────────────────────────────────────
1. Sentence-aware sliding window with overlap.
   - Window size: 400 characters (~60-80 words in English; fewer in Bangla due to longer words).
   - Overlap: 80 characters — ensures cross-sentence context is preserved at boundaries.
   - Why overlap? RAG retrieval quality degrades when a relevant sentence is split across
     two non-overlapping chunks. Overlap lets each chunk carry enough context for embedding.

2. Sentence splitting uses both Latin (. ! ?) and Bangla (।  ?) terminators.
   - Bangla uses the 'daari' (।) as a sentence terminator.
   - We split on these punctuation marks while preserving the delimiter.

3. Minimum chunk length: 50 characters — avoids embedding noise from OCR artifacts
   (stray characters, page headers, watermarks).

4. Language tagging per chunk: each chunk inherits the document language tag.
   Could be made finer-grained in future (character-level script detection per chunk).
"""

import re
from typing import Optional


# Sentence terminator pattern: Latin (.!?) and Bangla (।?)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?।\u09F7])\s+")

CHUNK_SIZE = 400       # target characters per chunk
CHUNK_OVERLAP = 80     # overlap between consecutive chunks
MIN_CHUNK_LEN = 50     # skip very short chunks (OCR noise)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, handling both English and Bangla terminators."""
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Split on sentence boundaries
    parts = SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, doc_language: Optional[str] = "en") -> list[dict]:
    """
    Produce overlapping text chunks from a document string.

    Returns list of:
      {chunk_index, text, char_start, char_end, language}
    """
    if not text or len(text.strip()) < MIN_CHUNK_LEN:
        return []

    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = ""
    current_start = 0
    chunk_index = 0
    char_cursor = 0  # tracks absolute position in original text

    # Build a mapping of sentence → approximate start position
    sentence_positions = []
    pos = 0
    for sent in sentences:
        idx = text.find(sent, pos)
        if idx == -1:
            idx = pos
        sentence_positions.append(idx)
        pos = idx + len(sent)

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_start = sentence_positions[i]

        if not current_chunk:
            current_chunk = sent
            current_start = sent_start
        elif len(current_chunk) + len(sent) + 1 <= CHUNK_SIZE:
            current_chunk += " " + sent
        else:
            # Emit current chunk
            if len(current_chunk) >= MIN_CHUNK_LEN:
                chunks.append({
                    "chunk_index": chunk_index,
                    "text": current_chunk.strip(),
                    "char_start": current_start,
                    "char_end": current_start + len(current_chunk),
                    "language": doc_language,
                })
                chunk_index += 1

            # Start new chunk with overlap: go back until we have ~CHUNK_OVERLAP chars
            # Find the boundary sentence for overlap
            overlap_text = ""
            j = i - 1
            while j >= 0 and len(overlap_text) < CHUNK_OVERLAP:
                overlap_text = sentences[j] + " " + overlap_text
                j -= 1

            if j + 1 < i:
                # Re-start from overlap start
                current_chunk = overlap_text.strip() + " " + sent
                current_start = sentence_positions[max(0, j + 1)]
            else:
                current_chunk = sent
                current_start = sent_start

        i += 1

    # Emit final chunk
    if current_chunk and len(current_chunk) >= MIN_CHUNK_LEN:
        chunks.append({
            "chunk_index": chunk_index,
            "text": current_chunk.strip(),
            "char_start": current_start,
            "char_end": current_start + len(current_chunk),
            "language": doc_language,
        })

    return chunks
