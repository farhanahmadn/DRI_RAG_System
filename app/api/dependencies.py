"""Dependency injection untuk endpoint — retriever via Depends(), BUKAN hardcode di route.

Pilihan retriever via env `RETRIEVER` (default "asli"):
- "asli" -> RetrieverAsli (Postgres+pgvector sungguhan), `default_wilayah` dari `RETRIEVER_WILAYAH`
  (default "Sleman Tengah") supaya demo satu wilayah tak perlu filter dokumen manual tiap request.
- "mock" -> MockRetriever (tanpa DB). Tests offline SELALU pakai ini (lihat tests/conftest.py) —
  jangan sampai default "asli" bikin `pytest -m "not live"` butuh koneksi DB.

Override lewat app.dependency_overrides juga tetap bisa dipakai saat test (kode reasoning/endpoint
tidak perlu berubah, SEAM, lihat CLAUDE.md § Retrieval).
"""

import os
from functools import lru_cache

from app.retrieval.base import Retriever
from app.retrieval.mock import MockRetriever
from app.retrieval.retriever import RetrieverAsli


@lru_cache
def get_retriever() -> Retriever:
    mode = os.getenv("RETRIEVER", "asli")
    if mode == "mock":
        return MockRetriever()
    wilayah = os.getenv("RETRIEVER_WILAYAH", "Sleman Tengah")
    return RetrieverAsli(default_wilayah=wilayah)
