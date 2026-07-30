"""app/retrieval/retriever.py — implementasi ASLI Protocol `Retriever` (base.py).

Pipeline search: query expansion -> embed bge-m3 (dense) + FTS lexical -> Reciprocal Rank Fusion
-> rerank bge-reranker-v2-m3 -> top_k. Filter tanggal_berlaku/zona/dokumen (versi & daerah).
get_by_reference: parse rujukan (dokumen + pasal/ayat/lampiran) -> lookup presisi (fallback dokumen-level).
get_parent: small-to-big (ayat -> pasal induk).

Swap dari MockRetriever: ganti isi app/api/dependencies.py::get_retriever menjadi
`return RetrieverAsli()` (opsional `default_wilayah="Sleman Timur"` utk demo satu wilayah). Tidak ada
file reasoning lain yang perlu berubah (SEAM). Lihat docs/INTEGRASI_RETRIEVER.md.
"""

from __future__ import annotations

import re

from app.retrieval import db, fusion, rerank
from app.retrieval.base import Chunk, RetrievalFilters
from app.retrieval.embeddings import encode_dense_one

# Query expansion: query reasoning berupa nama kategori indikator yang pendek (mis. "KDB").
# Perkaya jadi istilah regulasi supaya dense+FTS menemukan pasal/ketentuan yang tepat.
_EXPANSION = {
    "kdb": "KDB Koefisien Dasar Bangunan intensitas pemanfaatan ruang tata bangunan",
    "klb": "KLB Koefisien Lantai Bangunan intensitas pemanfaatan ruang",
    "kdh": "KDH Koefisien Dasar Hijau ruang terbuka hijau",
    "lp2b": "LP2B lahan pertanian pangan berkelanjutan alih fungsi",
    "lokasional lp2b": "LP2B lahan pertanian pangan berkelanjutan alih fungsi",
    "sempadan": "garis sempadan sungai perlindungan setempat",
    "sempadan sungai": "garis sempadan sungai perlindungan setempat",
    "lokasional sempadan sungai": "garis sempadan sungai perlindungan setempat",
    "banjir": "rawan bencana banjir genangan kawasan rawan",
    "lokasional banjir": "rawan bencana banjir genangan kawasan rawan",
    "resapan": "kawasan resapan air",
    "lokasional resapan": "kawasan resapan air",
    "kegiatan": "ketentuan kegiatan penggunaan lahan diizinkan terbatas bersyarat peraturan zonasi",
    "dampak tata guna lahan": (
        "limpasan air hujan runoff sumur resapan kolam retensi zero delta Q koefisien dasar hijau "
        "RTH drainase kawasan resapan air"
    ),
}

_RE_PASAL = re.compile(r"pasal\s+(\d+)", re.I)
_RE_AYAT = re.compile(r"ayat\s*\(?(\d+)\)?", re.I)
_RE_LAMPIRAN = re.compile(r"(lampiran\s+[IVXLC]+(?:\.\w+)?)", re.I)


def _expand(query: str) -> str:
    key = query.strip().lower()
    if key in _EXPANSION:
        return f"{query} {_EXPANSION[key]}"
    return query


class RetrieverAsli:
    """Implementasi Protocol Retriever atas Postgres+pgvector (bge-m3 + FTS + RRF + reranker)."""

    def __init__(self, dsn: str | None = None, *, default_wilayah: str | None = None,
                 candidate_k: int = 30, rerank_pool: int = 30) -> None:
        self._dsn = dsn
        self._conn = None
        self._default_wilayah = default_wilayah  # mis. "Sleman Timur" -> filters.dokumen default
        self._candidate_k = candidate_k
        self._rerank_pool = rerank_pool

    # -- koneksi (lazy, reuse; reconnect bila putus) --
    def _c(self):
        if self._conn is None or self._conn.closed:
            self._conn = db.connect(self._dsn)
        return self._conn

    def _apply_default_wilayah(self, filters: RetrievalFilters) -> RetrievalFilters:
        if self._default_wilayah and filters.dokumen is None:
            return filters.model_copy(update={"dokumen": self._default_wilayah})
        return filters

    # ---------------------------------------------------------------- search
    def search(self, query: str, filters: RetrievalFilters, top_k: int = 5) -> list[Chunk]:
        filters = self._apply_default_wilayah(filters or RetrievalFilters())
        q = _expand(query)
        conn = self._c()

        qvec = encode_dense_one(q)
        dense = db.dense_search(conn, qvec, filters, self._candidate_k)
        lexical = db.fts_search(conn, q, filters, self._candidate_k)

        fused = fusion.reciprocal_rank_fusion([[i for i, _ in dense], [i for i, _ in lexical]])
        if not fused:
            return []
        cand_ids = [i for i, _ in fused[:self._rerank_pool]]
        cand = db.hydrate(conn, cand_ids)

        scored = rerank.rerank(q, [c.teks for c in cand], top_k=top_k)
        out: list[Chunk] = []
        for idx, score in scored:
            c = cand[idx]
            out.append(c.model_copy(update={"skor": float(score)}))
        return out

    # ------------------------------------------------------- get_by_reference
    def get_by_reference(self, referensi: list[str]) -> list[Chunk]:
        conn = self._c()
        found: list[Chunk] = []
        seen: set[str] = set()
        for ref in referensi or []:
            m_pasal = _RE_PASAL.search(ref)
            pasal = m_pasal.group(1) if m_pasal else None
            m_ayat = _RE_AYAT.search(ref)
            ayat = m_ayat.group(1) if m_ayat else None
            m_lamp = _RE_LAMPIRAN.search(ref)
            lampiran = m_lamp.group(1) if m_lamp else None

            dok_ids = db.match_dokumen(conn, set(db._tok(ref)))
            if not dok_ids:
                continue
            hits = db.lookup_reference(conn, dok_ids, pasal, ayat, lampiran)
            # fallback: kalau minta pasal spesifik tapi kosong, ambil dokumen-level apa adanya
            if not hits and (pasal or lampiran):
                hits = db.lookup_reference(conn, dok_ids, None, None, None)
            for c in hits:
                if c.id not in seen:
                    seen.add(c.id)
                    found.append(c)
        return found

    # ------------------------------------------------------------- get_parent
    def get_parent(self, chunk_id: str) -> Chunk | None:
        return db.get_parent(self._c(), chunk_id)
