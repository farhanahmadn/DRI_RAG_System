"""Inti reasoning — generate_poin() untuk satu indikator jejak aturan.

Neuro-simbolik (CLAUDE.md): LLM diberi skema SEMPIT — hanya bagian bahasa (reasoning, kutipan,
saran, disclaimer). Semua angka (poin_id, kategori, kontribusi, status, target) dan metadata sitasi
(dokumen/pasal/halaman) dirakit DI SINI dari jejak/calculator/chunk yang benar-benar diretrieve —
bukan dipercaya dari output LLM. citation_id yang tidak dikenal (halusinasi) dibuang.
"""

from app.reasoning import llm_client
from app.reasoning.calculator import (
    hitung_target_rekomendasi,
    klasifikasi_tipe_rekomendasi,
    pilih_target_utama,
    rakit_status_numerik,
)
from app.reasoning.prompts import SYSTEM_PROMPT, build_user_prompt
from app.reasoning.templates import template_aman
from app.retrieval.base import Chunk, RetrievalFilters, Retriever
from app.schemas import IndikatorJejak, PoinOutput, RekomendasiOutput, SitasiOutput

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


def ambil_chunks_pendukung(
    indikator: IndikatorJejak,
    retriever: Retriever,
    top_k_dukungan: int = 3,
) -> list[Chunk]:
    """Retrieval chunk pendukung untuk satu indikator: referensi_hukum dulu, fallback search teks."""
    chunks = retriever.get_by_reference(indikator.referensi_hukum)
    if not chunks:
        chunks = retriever.search(indikator.kategori, RetrievalFilters(), top_k=top_k_dukungan)
    return chunks


def generate_poin(
    indikator: IndikatorJejak,
    retriever: Retriever,
    *,
    top_k_dukungan: int = 3,
    catatan_perbaikan: str | None = None,
    temperature: float = 0.0,
) -> PoinOutput:
    """Hasilkan PoinOutput untuk satu indikator. Skor 0 -> template (tanpa retrieval/LLM)."""
    if indikator.skor == 0:
        return template_aman(indikator)

    chunks = ambil_chunks_pendukung(indikator, retriever, top_k_dukungan)
    chunk_by_id = {chunk.id: chunk for chunk in chunks}

    target = hitung_target_rekomendasi(indikator)
    fakta_verdict = rakit_status_numerik(indikator)
    prompt = build_user_prompt(indikator, chunks, target, catatan_perbaikan, fakta_verdict)

    llm_out = llm_client.generate(
        prompt,
        _LLM_RESPONSE_SCHEMA,
        schema_name="poin_reasoning",
        system=SYSTEM_PROMPT,
        temperature=temperature,
    )

    sitasi: list[SitasiOutput] = []
    for item in llm_out.get("sitasi", []):
        chunk = chunk_by_id.get(item.get("citation_id"))
        if chunk is None:
            continue
        sitasi.append(
            SitasiOutput(
                citation_id=chunk.id,
                dokumen=chunk.dokumen,
                pasal=chunk.pasal or "",
                halaman=chunk.halaman or 0,
                kutipan=item.get("kutipan", chunk.teks),
                terverifikasi=True,
            )
        )

    return PoinOutput(
        poin_id=indikator.poin_id,
        kategori=indikator.kategori,
        status="Tidak Aman",
        kontribusi=indikator.kontribusi,
        reasoning_pendek=llm_out["reasoning_pendek"],
        reasoning_panjang=llm_out["reasoning_panjang"],
        sitasi=sitasi,
        rekomendasi=RekomendasiOutput(
            tipe=klasifikasi_tipe_rekomendasi(indikator.kategori),
            target=pilih_target_utama(target),
            saran=llm_out["saran"],
            disclaimer=llm_out.get("disclaimer"),
        ),
        low_confidence=False,
    )
