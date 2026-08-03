"""Kontrak SEAM antara reasoning (saya) dan RAG (teman).

Protocol `Retriever` dan skema `Chunk`/`RetrievalFilters` di file ini adalah kontrak bersama —
jarang diubah. Implementasi asli (embed bge-m3 + pgvector + rerank) ada di
`app/retrieval/retriever.py` (milik teman). Sampai itu siap, reasoning memakai
`app/retrieval/mock.py::MockRetriever` yang mematuhi Protocol yang sama.
"""

from datetime import date
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel


class Chunk(BaseModel):
    id: str
    level: Literal["pasal", "ayat", "tabel"]
    parent_id: str | None = None
    teks: str
    dokumen: str
    pasal: str | None = None
    ayat: str | None = None
    halaman: int | None = None
    skor: float = 0.0
    tanggal_berlaku: date | None = None

    istilah_kode: str | None = None
    zona: str | None = None
    jenis: str | None = None
    tanggal_dicabut: date | None = None


class RetrievalFilters(BaseModel):
    as_of: date | None = None
    zona: str | None = None
    dokumen: str | None = None
    jenis: str | None = None


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, filters: RetrievalFilters, top_k: int = 5) -> list[Chunk]: ...

    def get_by_reference(self, referensi: list[str]) -> list[Chunk]: ...

    def get_parent(self, chunk_id: str) -> Chunk | None: ...
