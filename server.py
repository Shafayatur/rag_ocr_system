"""Entry point: run with `python server.py` or `uvicorn app.main:app --reload`"""
import uvicorn
import os
import sys

# Add project root to path so `app` package is importable
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
