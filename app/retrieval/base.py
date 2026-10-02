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
    # Filter KELUARGA zona (mis. "R" utk cocok "R-2"/"R-3"/"R-4") — TERPISAH dari `zona` (exact
    # match). Ditambahkan setelah bukti live (APP-2026-6191): input back-end (`lokasi.rdtr_zone`)
    # cuma kasih nama zona induk ("Zona Perumahan"), bukan kode sub-zona presisi, jadi filter exact
    # `zona` tak bisa dipakai — prefix ini cukup utk cegah kontaminasi lintas-KELUARGA zona (mis.
    # Lampiran Zona Perkantoran "KT" ikut kepentar utk pemohon Zona Perumahan "R").
    zona_prefix: str | None = None


@runtime_checkable
class Retriever(Protocol):
    # `tanpa_lexical`: lewati leg lexical (FTS) sehingga kandidat murni dari pencarian vektor.
    # Dipakai per-poin oleh generator — lihat generator._TANPA_LEXICAL_PER_POIN utk ukurannya.
    # Default False = perilaku hibrida penuh, jadi implementasi lain tak wajib peduli.
    def search(self, query: str, filters: RetrievalFilters, top_k: int = 5, *,
               tanpa_lexical: bool = False) -> list[Chunk]: ...

    def get_by_reference(self, referensi: list[str]) -> list[Chunk]: ...

    def get_parent(self, chunk_id: str) -> Chunk | None: ...
