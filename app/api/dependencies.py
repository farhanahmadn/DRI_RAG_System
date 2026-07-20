"""Dependency injection untuk endpoint — retriever via Depends(), BUKAN hardcode di route.

Swap ke retriever asli (RAG teman) nanti = ganti isi get_retriever() saja, atau override lewat
app.dependency_overrides saat test. Kode reasoning/endpoint tidak perlu berubah (SEAM, lihat
CLAUDE.md § Retrieval).
"""

from functools import lru_cache

from app.retrieval.base import Retriever
from app.retrieval.mock import MockRetriever


@lru_cache
def get_retriever() -> Retriever:
    return MockRetriever()
