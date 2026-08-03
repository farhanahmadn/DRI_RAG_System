"""Inti reasoning — generate_poin() untuk satu PoinKonteks (dari app/adapter.py).

Neuro-simbolik (CLAUDE.md): LLM diberi skema SEMPIT — hanya bagian bahasa (reasoning, kutipan,
saran, disclaimer). Semua metadata (poin_id, kategori, status, tipe rekomendasi, target) dan
metadata sitasi (dokumen/pasal/halaman) dirakit DI SINI dari poin/calculator/chunk yang benar-benar
diretrieve — bukan dipercaya dari output LLM. citation_id yang tidak dikenal (halusinasi) dibuang.
"""

from app.reasoning import llm_client
from app.reasoning.calculator import pilih_target_utama_intensitas
from app.reasoning.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.base import Chunk, RetrievalFilters, Retriever
from app.schemas import MetaL2, PoinKonteks, PoinOutput, RekomendasiOutput, SitasiOutput

# APP-2026-3468: saran deterministik utk poin aman/lolos — reasoning_pendek/panjang & sitasi TETAP
# dari LLM (poin aman WAJIB tetap dijelaskan KENAPA lolos, bukan reasoning generik tanpa sitasi),
# hanya bagian "tidak ada tindakan lanjut" ini yang aman ditemplate (tak ada apa pun utk direkomendasikan).
_SARAN_AMAN = "Tidak diperlukan tindakan khusus; poin ini telah memenuhi ketentuan."

_LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning_pendek": {"type": "string"},
        "reasoning_panjang": {"type": "string"},
        "sitasi": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "citation_id": {"type": "string"},
                    "kutipan": {"type": "string"},
                },
                "required": ["citation_id", "kutipan"],
                "additionalProperties": False,
            },
        },
        "saran": {"type": "string"},
        "disclaimer": {"type": ["string", "null"]},
    },
    "required": ["reasoning_pendek", "reasoning_panjang", "sitasi", "saran", "disclaimer"],
    "additionalProperties": False,
}


# Kata kunci PENDEK utk jalur search fallback — selaras dgn `_EXPANSION` di
# app/retrieval/retriever.py (RAG asli teman: exact-match lowercase key -> istilah regulasi,
# TIDAK fuzzy/substring). `poin.kategori` sekarang string deskriptif panjang ("Klasifikasi
# Kegiatan (ITBX)" dst) yang TIDAK match kunci `_EXPANSION` manapun persis — pakai kata kunci
# pendek di sini supaya RetrieverAsli tetap dapat manfaat ekspansi query.
_QUERY_FALLBACK_PER_POIN = {
    "itbx": "kegiatan",
    "intensitas": "kdb",
    # Fix #5: entri "dampak tata guna lahan" ditambahkan ke _EXPANSION retriever.py (limpasan/
    # runoff/sumur resapan/kolam retensi/zero delta Q/RTH) — kata kunci pendek di sini persis
    # match key itu, jangan pakai poin.kategori panjang ("Dampak Tata Guna Lahan" apa adanya
    # kebetulan lowercase-match juga, tapi eksplisit lebih tahan kalau kategori berubah nanti).
    "dampak": "dampak tata guna lahan",
}


def ambil_chunks_pendukung(
    poin: PoinKonteks,
    retriever: Retriever,
    top_k_dukungan: int = 3,
) -> list[Chunk]:
    """Retrieval chunk pendukung utk satu poin: dasar_hukum dulu, fallback search kata kunci pendek."""
    referensi = [f"{d.dokumen} {d.pasal}" for d in poin.dasar_hukum if d.pasal]
    chunks = retriever.get_by_reference(referensi) if referensi else []
    # Investigasi ITBX APP-2026-6191 (live thd DB nyata): `dasar_hukum` back-end kadang berupa label
    # non-pasal ("Matriks ITBX", tanpa nomor) + dokumen generik ("RDTR Sleman") — get_by_reference
    # (tidak bisa parse nomor pasal, jatuh ke fallback dokumen-level) bisa mengembalikan RATUSAN chunk
    # TAK TERBATAS (terbukti live: seluruh korpus, bukan cuma 6 chunk kecil di MockRetriever). Prompt
    # yang membanjiri LLM bikin ia gagal memilih sitasi sama sekali. Batasi ke top_k_dukungan di sini
    # (bukan di retriever.py — bukan file saya) — anchor (dasar_hukum asli) tetap utuh dikirim terpisah
    # ke prompt (lihat build_user_prompt), jadi pembatasan ini TIDAK mengurangi sitasi yang faithful.
    if len(chunks) > top_k_dukungan:
        chunks = chunks[:top_k_dukungan]
    if not chunks:
        query_fallback = _QUERY_FALLBACK_PER_POIN.get(poin.poin_id, poin.kategori)
        chunks = retriever.search(query_fallback, RetrievalFilters(), top_k=top_k_dukungan)
    return chunks


def apakah_aman(poin: PoinKonteks) -> bool:
    """Poin jelas aman/lolos -> boleh template tanpa LLM (hemat kuota). REUSE fakta yang sudah
    dihitung adapter/calculator, tidak menurunkan ulang di sini.
    """
    if poin.poin_id == "itbx":
        return (
            poin.status == "I"
            and bool(poin.fakta.get("lolos"))
            and not poin.fakta.get("fallback_data_kosong", False)
        )
    if poin.poin_id == "intensitas":
        return not poin.fakta.get("target")
    if poin.poin_id == "dampak":
        mitigasi = poin.fakta.get("mitigasi") or {}
        return not mitigasi.get("perlu_mitigasi")
    return False


def generate_poin(
    poin: PoinKonteks,
    retriever: Retriever,
    meta: MetaL2 | None = None,
    *,
    top_k_dukungan: int = 3,
    catatan_perbaikan: str | None = None,
    temperature: float = 0.0,
) -> PoinOutput:
    """Hasilkan PoinOutput untuk satu poin — SELALU lewat retrieval + LLM, termasuk poin aman/lolos
    (APP-2026-3468: poin aman WAJIB tetap dijelaskan KENAPA lolos + sitasi pasal, bukan reasoning
    generik tanpa sitasi seperti sebelumnya). Poin aman hanya dapat template pada
    `rekomendasi.saran`/`target` (lihat apakah_aman di bawah) — reasoning & sitasi TETAP dari LLM.
    """
    chunks = ambil_chunks_pendukung(poin, retriever, top_k_dukungan)
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    anchor_by_id = {f"anchor-{i}": d for i, d in enumerate(poin.dasar_hukum)}

    prompt = build_user_prompt(poin, chunks, meta, catatan_perbaikan)

    llm_out = llm_client.generate(
        prompt,
        _LLM_RESPONSE_SCHEMA,
        schema_name="poin_reasoning",
        system=SYSTEM_PROMPT,
        temperature=temperature,
    )

    sitasi: list[SitasiOutput] = []
    for item in llm_out.get("sitasi", []):
        citation_id = item.get("citation_id")
        kutipan = item.get("kutipan", "")

        anchor = anchor_by_id.get(citation_id)
        if anchor is not None:
            sitasi.append(
                SitasiOutput(
                    citation_id=citation_id,
                    dokumen=anchor.dokumen,
                    pasal=anchor.pasal or "",
                    halaman=0,
                    kutipan=kutipan or anchor.kutipan,
                    terverifikasi=True,
                )
            )
            continue

        chunk = chunk_by_id.get(citation_id)
        if chunk is None:
            continue  # citation_id tak dikenal (halusinasi) -> dibuang
        sitasi.append(
            SitasiOutput(
                citation_id=chunk.id,
                dokumen=chunk.dokumen,
                pasal=chunk.pasal or "",
                halaman=chunk.halaman or 0,
                # SELALU chunk.teks apa adanya (BUKAN `kutipan or chunk.teks`) — kutipan dari LLM di
                # sini cuma dipakai utk MEMILIH chunk mana yang relevan, isi teksnya sendiri wajib
                # verbatim dari hasil retrieval, LLM tak boleh menulis ulang/mengarang kutipan pasal.
                kutipan=chunk.teks,
                terverifikasi=True,
            )
        )

    target: float | str | None = None
    if poin.tipe_rekomendasi == "numerik":
        target = pilih_target_utama_intensitas(poin.fakta.get("target") or {})

    saran = llm_out["saran"]
    if apakah_aman(poin):
        # Tidak ada pelanggaran utk ditindaklanjuti -> saran & target deterministik, TAPI
        # reasoning_pendek/panjang & sitasi di atas tetap murni dari LLM (lihat docstring).
        saran = _SARAN_AMAN
        target = None

    return PoinOutput(
        poin_id=poin.poin_id,
        kategori=poin.kategori,
        status=poin.status,
        reasoning_pendek=llm_out["reasoning_pendek"],
        reasoning_panjang=llm_out["reasoning_panjang"],
        sitasi=sitasi,
        rekomendasi=RekomendasiOutput(
            tipe=poin.tipe_rekomendasi,
            target=target,
            saran=saran,
            disclaimer=llm_out.get("disclaimer"),
        ),
        low_confidence=False,
    )
