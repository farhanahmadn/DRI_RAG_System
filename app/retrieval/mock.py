"""MockRetriever — stand-in RAG dev/testing, mematuhi Protocol `Retriever` di `base.py`.

PERINGATAN: Data di bawah adalah CONTOH/DUMMY untuk mengembangkan & menguji pipeline reasoning
(parent-child pasal/ayat, filter, dsb). Teksnya BUKAN kutipan hukum resmi yang sudah diverifikasi
kata-per-kata — jangan dipakai sebagai sitasi produksi. Saat retrieval asli (RAG teman, embed
bge-m3 + pgvector + rerank) siap, modul ini digantikan tanpa mengubah kode reasoning (lihat SEAM
di CLAUDE.md).
"""

import re
from datetime import date

from app.retrieval.base import Chunk, RetrievalFilters


def _build_mock_chunks() -> list[Chunk]:
    pasal_44 = Chunk(
        id="uu41-2009-p44",
        level="pasal",
        parent_id=None,
        teks=(
            "Pasal 44: Lahan Pertanian Pangan Berkelanjutan yang sudah ditetapkan dilindungi dan "
            "dilarang dialihfungsikan, kecuali untuk kepentingan umum dengan syarat tertentu "
            "(kajian kelayakan strategis, penyusunan rencana alih fungsi, dan penyediaan lahan "
            "pengganti)."
        ),
        dokumen="UU No. 41 Tahun 2009 tentang Perlindungan Lahan Pertanian Pangan Berkelanjutan",
        pasal="44",
        halaman=21,
        skor=0.0,
        tanggal_berlaku=date(2009, 8, 14),
        zona="LP2B",
        jenis="UU",
    )
    pasal_44_ayat1 = Chunk(
        id="uu41-2009-p44-a1",
        level="ayat",
        parent_id=pasal_44.id,
        teks="(1) Lahan Pertanian Pangan Berkelanjutan yang sudah ditetapkan dilarang dialihfungsikan.",
        dokumen=pasal_44.dokumen,
        pasal="44",
        ayat="1",
        halaman=21,
        skor=0.0,
        tanggal_berlaku=pasal_44.tanggal_berlaku,
        zona="LP2B",
        jenis="UU",
    )
    pasal_44_ayat2 = Chunk(
        id="uu41-2009-p44-a2",
        level="ayat",
        parent_id=pasal_44.id,
        teks=(
            "(2) Pengalihfungsian sebagaimana dimaksud pada ayat (1) dapat dilakukan karena "
            "kepentingan umum dengan syarat: kajian kelayakan strategis, rencana alih fungsi lahan, "
            "pembebasan hak, dan penyediaan lahan pengganti."
        ),
        dokumen=pasal_44.dokumen,
        pasal="44",
        ayat="2",
        halaman=21,
        skor=0.0,
        tanggal_berlaku=pasal_44.tanggal_berlaku,
        zona="LP2B",
        jenis="UU",
    )

    pasal_50 = Chunk(
        id="uu41-2009-p50",
        level="pasal",
        parent_id=None,
        teks=(
            "Pasal 50: Setiap orang yang mengalihfungsikan Lahan Pertanian Pangan Berkelanjutan "
            "tanpa izin dapat dikenai sanksi administratif berupa penghentian kegiatan, pencabutan "
            "izin, dan/atau kewajiban pemulihan fungsi lahan."
        ),
        dokumen="UU No. 41 Tahun 2009 tentang Perlindungan Lahan Pertanian Pangan Berkelanjutan",
        pasal="50",
        halaman=24,
        skor=0.0,
        tanggal_berlaku=date(2009, 8, 14),
        zona="LP2B",
        jenis="UU",
    )
    pasal_50_ayat1 = Chunk(
        id="uu41-2009-p50-a1",
        level="ayat",
        parent_id=pasal_50.id,
        teks=(
            "(1) Alih fungsi Lahan Pertanian Pangan Berkelanjutan tanpa izin dikenai sanksi "
            "administratif berupa penghentian kegiatan dan/atau kewajiban pemulihan fungsi lahan."
        ),
        dokumen=pasal_50.dokumen,
        pasal="50",
        ayat="1",
        halaman=24,
        skor=0.0,
        tanggal_berlaku=pasal_50.tanggal_berlaku,
        zona="LP2B",
        jenis="UU",
    )

    return [pasal_44, pasal_44_ayat1, pasal_44_ayat2, pasal_50, pasal_50_ayat1]


_PASAL_RE = re.compile(r"pasal\s+(\d+)", re.IGNORECASE)


class MockRetriever:
    """Implementasi dummy Protocol `Retriever` untuk dev/testing sebelum RAG teman siap."""

    def __init__(self) -> None:
        self._chunks = _build_mock_chunks()
        self._by_id = {c.id: c for c in self._chunks}

    def search(self, query: str, filters: RetrievalFilters, top_k: int = 5) -> list[Chunk]:
        query_tokens = set(re.findall(r"\w+", query.lower()))
        candidates: list[Chunk] = []
        for chunk in self._chunks:
            if not self._passes_filters(chunk, filters):
                continue
            haystack = f"{chunk.teks} {chunk.pasal or ''} {chunk.dokumen}".lower()
            overlap = sum(1 for tok in query_tokens if tok in haystack)
            if overlap > 0:
                candidates.append(chunk.model_copy(update={"skor": float(overlap)}))
        candidates.sort(key=lambda c: c.skor, reverse=True)
        return candidates[:top_k]

    def get_by_reference(self, referensi: list[str]) -> list[Chunk]:
        found: list[Chunk] = []
        seen_ids: set[str] = set()
        for ref in referensi:
            ref_lower = ref.lower()
            match = _PASAL_RE.search(ref_lower)
            pasal_num = match.group(1) if match else None
            for chunk in self._chunks:
                if chunk.id in seen_ids:
                    continue
                dokumen_hit = any(
                    tok in ref_lower for tok in ("41/2009", "41 tahun 2009", "lp2b")
                ) or chunk.dokumen.lower() in ref_lower
                pasal_hit = pasal_num is not None and chunk.pasal == pasal_num
                if dokumen_hit and (pasal_num is None or pasal_hit):
                    found.append(chunk)
                    seen_ids.add(chunk.id)
        return found

    def get_parent(self, chunk_id: str) -> Chunk | None:
        chunk = self._by_id.get(chunk_id)
        if chunk is None or chunk.parent_id is None:
            return None
        return self._by_id.get(chunk.parent_id)

    @staticmethod
    def _passes_filters(chunk: Chunk, filters: RetrievalFilters) -> bool:
        if filters.as_of is not None:
            if chunk.tanggal_berlaku is not None and chunk.tanggal_berlaku > filters.as_of:
                return False
            if chunk.tanggal_dicabut is not None and chunk.tanggal_dicabut <= filters.as_of:
                return False
        if filters.zona is not None and chunk.zona is not None and chunk.zona != filters.zona:
            return False
        if filters.dokumen is not None and filters.dokumen.lower() not in chunk.dokumen.lower():
            return False
        if filters.jenis is not None and chunk.jenis is not None and chunk.jenis != filters.jenis:
            return False
        return True
