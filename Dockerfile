FROM python:3.12-slim

# Install Tesseract with Bangla language pack and PDF rendering dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ben \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart \
    pytesseract Pillow PyPDF2 pdf2image numpy scikit-learn

COPY . .

# Create runtime directories
RUN mkdir -p uploads db static

EXPOSE 8000

# ANTHROPIC_API_KEY must be supplied at runtime:
# docker run -e ANTHROPIC_API_KEY=sk-... -p 8000:8000 rag-ocr-system
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
