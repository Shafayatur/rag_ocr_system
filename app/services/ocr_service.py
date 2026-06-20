"""
OCR Service — Local extraction using Tesseract (open-source, no external API calls).

Engine choice: Tesseract 5.x with LSTM neural network mode.
  - Supports English (eng) and Bangla/Bengali (ben) out of the box via tessdata packs.
  - Trade-offs: Tesseract handles printed text well; handwritten or stylised fonts
    (especially complex Bangla conjuncts/matras) may produce errors. For production,
    Surya or a local vision-language model (e.g., GOT-OCR2.0) would improve accuracy
    on degraded Bangla scripts, but requires significantly more GPU/RAM.
  - Baseline accuracy: ~92-96% on clean printed English; ~75-88% on clear printed Bangla
    (accuracy degrades on low-DPI scans or cursive ligatures).

Language detection strategy:
  - We attempt OCR with combined 'ben+eng' mode, then count Unicode script ranges
    to classify the document as 'en', 'bn', or 'mixed'.
"""

import os
import io
import re
import logging
from typing import Optional

import pytesseract
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)


# Unicode ranges for Bangla characters (U+0980 – U+09FF)
BANGLA_RANGE = re.compile(r"[\u0980-\u09FF]")
# Rough English/Latin range
LATIN_RANGE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """
    Classify document language based on character script ratios.
    Returns 'en', 'bn', or 'mixed'.
    """
    if not text:
        return "en"
    bangla_chars = len(BANGLA_RANGE.findall(text))
    latin_chars = len(LATIN_RANGE.findall(text))
    total = bangla_chars + latin_chars

    if total == 0:
        return "en"

    bangla_ratio = bangla_chars / total
    if bangla_ratio > 0.75:
        return "bn"
    elif bangla_ratio > 0.2:
        return "mixed"
    else:
        return "en"


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Image preprocessing pipeline to improve OCR accuracy.
    Steps:
      1. Convert to grayscale — colour information is irrelevant for OCR.
      2. Resize to at least 300 DPI equivalent — Tesseract performs better at higher resolution.
      3. Sharpen — reduces blur from scanning/photography.
      4. Enhance contrast — helps distinguish ink from background.
    """
    # Convert to grayscale
    img = img.convert("L")

    # Upscale if small (simulate 300 DPI)
    min_dim = 1800
    w, h = img.size
    if max(w, h) < min_dim:
        scale = min_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Sharpen slightly
    img = img.filter(ImageFilter.SHARPEN)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)

    return img


def ocr_image(img: Image.Image, language_hint: Optional[str] = None) -> dict:
    """
    Run Tesseract OCR on a PIL Image.

    Args:
        img: PIL Image object.
        language_hint: 'en', 'bn', 'mixed', or None (auto-detect with combined mode).

    Returns:
        dict with 'text', 'language', 'confidence' (average word confidence).
    """
    img = preprocess_image(img)

    # Map language hints to Tesseract language strings
    # Always try combined mode first for mixed docs
    lang_map = {
        "en": "eng",
        "bn": "ben",
        "mixed": "ben+eng",
        None: "ben+eng",  # default: try both
    }
    tess_lang = lang_map.get(language_hint, "ben+eng")

    # Tesseract config: LSTM engine (mode 1), auto page segmentation (mode 3)
    custom_config = r"--oem 1 --psm 3"

    try:
        # Get text
        text = pytesseract.image_to_string(img, lang=tess_lang, config=custom_config)

        # Get per-word confidence data
        try:
            data = pytesseract.image_to_data(
                img, lang=tess_lang, config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            confs = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
            avg_confidence = round(sum(confs) / len(confs), 1) if confs else 0.0
        except Exception:
            avg_confidence = 0.0

        detected_lang = detect_language(text)
        return {
            "text": text.strip(),
            "language": detected_lang,
            "confidence": avg_confidence,
            "engine": f"tesseract-{tess_lang}",
        }

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return {"text": "", "language": "en", "confidence": 0.0, "engine": "tesseract"}


def ocr_pdf(file_path: str) -> dict:
    """
    Extract text from a PDF file.
    Strategy:
      1. Try direct text extraction with PyPDF2 (fast, lossless for digital PDFs).
      2. If extracted text is empty/too short, fall back to image-based OCR
         (renders each page as an image then runs Tesseract).
    Returns dict with 'text', 'language', 'page_count', 'method'.
    """
    # Try text extraction first
    try:
        import PyPDF2
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages_text = []
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
            full_text = "\n\n".join(pages_text).strip()
            page_count = len(reader.pages)

        # If we got meaningful text (>50 chars), use it
        if len(full_text) > 50:
            lang = detect_language(full_text)
            return {
                "text": full_text,
                "language": lang,
                "page_count": page_count,
                "method": "pdf-text-extraction",
                "confidence": 100.0,
                "engine": "PyPDF2",
            }
    except Exception as e:
        logger.warning(f"PDF text extraction failed: {e}")

    # Fall back to image-based OCR (handles scanned PDFs)
    return _ocr_pdf_as_images(file_path)


def _ocr_pdf_as_images(file_path: str) -> dict:
    """Convert PDF pages to images and OCR each page."""
    try:
        # pdf2image converts PDF pages to PIL Images
        from pdf2image import convert_from_path
        images = convert_from_path(file_path, dpi=300)
    except ImportError:
        # If pdf2image not available, try ghostscript via subprocess
        logger.warning("pdf2image not available; attempting PIL PDF open")
        try:
            img = Image.open(file_path)
            images = [img]
        except Exception as e:
            logger.error(f"Cannot open PDF as image: {e}")
            return {"text": "", "language": "en", "page_count": 0, "method": "failed", "confidence": 0.0, "engine": "none"}
    except Exception as e:
        logger.error(f"PDF to image conversion failed: {e}")
        return {"text": "", "language": "en", "page_count": 0, "method": "failed", "confidence": 0.0, "engine": "none"}

    all_text = []
    confidences = []
    for img in images:
        result = ocr_image(img)
        all_text.append(result["text"])
        confidences.append(result["confidence"])

    full_text = "\n\n".join(all_text).strip()
    avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0.0

    return {
        "text": full_text,
        "language": detect_language(full_text),
        "page_count": len(images),
        "method": "pdf-image-ocr",
        "confidence": avg_conf,
        "engine": "tesseract-ben+eng",
    }


def ocr_file(file_path: str, filename: str, language_hint: Optional[str] = None) -> dict:
    """
    Main entry point: dispatch to PDF or image OCR based on file extension.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        result = ocr_pdf(file_path)
        result["file_type"] = "pdf"
    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"):
        img = Image.open(file_path)
        result = ocr_image(img, language_hint=language_hint)
        result["file_type"] = "image"
        result["page_count"] = 1
        result["method"] = "image-ocr"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    logger.info(
        f"OCR complete for '{filename}': lang={result.get('language')}, "
        f"chars={len(result.get('text',''))}, conf={result.get('confidence')}%"
    )
    return result
