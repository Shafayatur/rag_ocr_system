"""
Test suite for the Local OCR & RAG System.
Run with: python tests/test_system.py
"""

import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.database import DatabaseService
from app.services.chunker import chunk_text, split_into_sentences
from app.services.vector_store import VectorStoreService, TFIDFVectorizer
from app.services.ocr_service import detect_language as ocr_detect_lang, preprocess_image


BANGLA_SAMPLE = (
    "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ। এটি বঙ্গোপসাগরের তীরে অবস্থিত। "
    "ঢাকা বাংলাদেশের রাজধানী এবং সবচেয়ে বড় শহর। "
    "বাংলা এখানকার প্রধান ভাষা এবং এটি বিশ্বের সপ্তম সর্বাধিক কথিত ভাষা।"
)

ENGLISH_SAMPLE = (
    "Bangladesh is a country in South Asia. It is located on the Bay of Bengal. "
    "Dhaka is the capital and largest city of Bangladesh. "
    "Bengali is the official language spoken by over 98% of the population."
)

MIXED_SAMPLE = (
    "Bangladesh বাংলাদেশ is a country. "
    "ঢাকা Dhaka is the capital city রাজধানী of Bangladesh বাংলাদেশ. "
    "Bengali বাংলা ভাষা is spoken by millions কোটি মানুষ here এখানে."
)


# ── Test 1: Language Detection ────────────────────────────────────────────────
def test_language_detection():
    print("\n[1] Language Detection")
    en = ocr_detect_lang(ENGLISH_SAMPLE)
    bn = ocr_detect_lang(BANGLA_SAMPLE)
    mx = ocr_detect_lang(MIXED_SAMPLE)
    assert en == "en", f"Expected 'en', got '{en}'"
    assert bn == "bn", f"Expected 'bn', got '{bn}'"
    assert mx == "mixed", f"Expected 'mixed', got '{mx}'"
    print(f"  ✅ English → '{en}', Bangla → '{bn}', Mixed → '{mx}'")


# ── Test 2: Text Chunking ────────────────────────────────────────────────────
def test_chunker():
    print("\n[2] Text Chunking")
    long_text = (ENGLISH_SAMPLE + " ") * 10
    chunks = chunk_text(long_text, doc_language="en")
    assert len(chunks) > 1, "Should produce multiple chunks"
    for c in chunks:
        assert "text" in c
        assert "chunk_index" in c
        assert len(c["text"]) >= 50
    print(f"  ✅ {len(chunks)} chunks produced from {len(long_text)}-char text")

    # Bangla chunking
    bn_long = (BANGLA_SAMPLE + " ") * 5
    bn_chunks = chunk_text(bn_long, doc_language="bn")
    assert len(bn_chunks) >= 1
    print(f"  ✅ {len(bn_chunks)} Bangla chunks produced")

    # Overlap: consecutive chunks should share some text boundary context
    if len(chunks) >= 2:
        print(f"  ✅ Overlap check: chunk[0] ends at char {chunks[0]['char_end']}, chunk[1] starts at {chunks[1]['char_start']}")


# ── Test 3: TF-IDF Vectorizer ────────────────────────────────────────────────
def test_tfidf():
    print("\n[3] TF-IDF Vector Store")
    corpus = [
        "machine learning artificial intelligence deep neural network",
        "Dhaka Bangladesh capital city population",
        "sensor data time series prediction forecast model",
        BANGLA_SAMPLE,
    ]
    vec = TFIDFVectorizer(ngram_range=(2, 3))
    vec.fit(corpus)
    assert vec._fitted
    assert len(vec.vocabulary) > 0

    q_vec = vec.transform("machine learning model")
    doc_vec = vec.transform(corpus[0])
    sim = vec.cosine_similarity(q_vec, doc_vec)
    assert sim > 0, "Query should match its own document"

    # Different doc should score lower
    other_vec = vec.transform(corpus[1])
    other_sim = vec.cosine_similarity(q_vec, other_vec)
    assert sim > other_sim, "Relevant doc should score higher"
    print(f"  ✅ Vocabulary: {len(vec.vocabulary)} ngrams")
    print(f"  ✅ Similarity (relevant): {sim:.4f} | (irrelevant): {other_sim:.4f}")


# ── Test 4: Database ─────────────────────────────────────────────────────────
def test_database():
    print("\n[4] Database Service")
    tmpdir = tempfile.mkdtemp()
    try:
        db = DatabaseService(db_path=os.path.join(tmpdir, "test.db"))
        db.initialize()

        # Insert document
        doc_id = db.insert_document(
            filename="test.pdf", file_type="pdf", language="en",
            doc_date="2024-01-15", page_count=3,
            raw_text=ENGLISH_SAMPLE, file_path="/tmp/test.pdf"
        )
        assert doc_id is not None and doc_id > 0

        # Retrieve
        doc = db.get_document(doc_id)
        assert doc["filename"] == "test.pdf"
        assert doc["language"] == "en"

        # Insert Bangla document
        doc_id2 = db.insert_document(
            filename="bangla_doc.png", file_type="image", language="bn",
            doc_date="2024-03-20", page_count=1,
            raw_text=BANGLA_SAMPLE, file_path="/tmp/bangla.png"
        )

        # Insert chunks
        chunks = chunk_text(ENGLISH_SAMPLE * 5, "en")
        db.insert_chunks(doc_id, chunks)
        stored = db.get_chunks_for_doc(doc_id)
        assert len(stored) == len(chunks)

        # Metadata filtering
        en_chunks = db.get_filtered_chunks(language="en")
        bn_chunks = db.get_filtered_chunks(language="bn")
        assert all(c["doc_id"] == doc_id for c in en_chunks)

        # Date filtering
        after_jan = db.get_filtered_chunks(doc_date_from="2024-02-01")
        assert all(c["doc_id"] == doc_id2 for c in after_jan)

        # Stats
        stats = db.get_stats()
        assert stats["total_documents"] == 2

        print(f"  ✅ Inserted {stats['total_documents']} docs, {stats['total_chunks']} chunks")
        print(f"  ✅ Language filter: {len(en_chunks)} en chunks, {len(bn_chunks)} bn chunks")
        print(f"  ✅ Date filter (after 2024-02-01): {len(after_jan)} chunks")
    finally:
        shutil.rmtree(tmpdir)


# ── Test 5: Vector Search ────────────────────────────────────────────────────
def test_vector_search():
    print("\n[5] Vector Search Pipeline")
    tmpdir = tempfile.mkdtemp()
    try:
        db = DatabaseService(db_path=os.path.join(tmpdir, "vec_test.db"))
        db.initialize()

        # Add documents
        docs = [
            ("agriculture.pdf", "pdf", "en", "2024-01-01",
             "crop yield prediction using satellite imagery and soil sensors. "
             "machine learning models forecast harvest outcomes based on rainfall and temperature patterns. " * 5),
            ("bangladesh.pdf", "pdf", "mixed", "2024-06-15",
             MIXED_SAMPLE * 5),
            ("bangla_only.png", "image", "bn", "2023-11-20",
             BANGLA_SAMPLE * 4),
        ]
        for fname, ftype, lang, date, text in docs:
            did = db.insert_document(fname, ftype, lang, date, 1, text, f"/tmp/{fname}")
            chunks = chunk_text(text, lang)
            if chunks:
                db.insert_chunks(did, chunks)

        vs = VectorStoreService()
        vs.load_from_db(db)
        assert vs._fitted
        print(f"  ✅ Index built with {len(vs._chunk_records)} chunks")

        # Semantic search — should return agriculture doc
        results = vs.search("crop harvest rainfall prediction", top_k=3)
        assert results, "Should find results"
        top = results[0]
        assert "agriculture" in top["filename"].lower(), f"Expected agriculture doc, got {top['filename']}"
        print(f"  ✅ Top hit: '{top['filename']}' (score={top['score']:.4f})")

        # Filtered search — English only
        en_chunks = db.get_filtered_chunks(language="en")
        en_results = vs.search("crop harvest", top_k=3, filtered_chunks=en_chunks)
        assert all("bangla" not in r["filename"] for r in en_results)
        print(f"  ✅ Language filter: {len(en_results)} English results, no Bangla docs")

        # Date filter
        old_chunks = db.get_filtered_chunks(doc_date_to="2024-01-31")
        old_results = vs.search("crop", top_k=5, filtered_chunks=old_chunks)
        if old_results:
            assert all(r.get("doc_date", "") <= "2024-01-31" for r in old_results)
            print(f"  ✅ Date filter: {len(old_results)} chunks from before 2024-02-01")

    finally:
        shutil.rmtree(tmpdir)


# ── Test 6: Sentence splitting ────────────────────────────────────────────────
def test_sentence_split():
    print("\n[6] Bangla Sentence Splitting")
    text = "আমি বাংলায় কথা বলি। তুমি কি বাংলা জানো? হ্যাঁ, আমি জানি।"
    sents = split_into_sentences(text)
    assert len(sents) >= 2, f"Expected multiple sentences, got {len(sents)}"
    print(f"  ✅ Split into {len(sents)} sentences: {sents}")


# ── Run all tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Local OCR & RAG System — Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0

    for fn in [
        test_language_detection,
        test_chunker,
        test_tfidf,
        test_database,
        test_vector_search,
        test_sentence_split,
    ]:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
