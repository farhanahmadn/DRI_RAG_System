"""Inti reasoning — generate_poin() untuk satu PoinKonteks (dari app/adapter.py).

Neuro-simbolik (CLAUDE.md): LLM diberi skema SEMPIT — hanya bagian bahasa (reasoning, kutipan,
saran, disclaimer). Semua metadata (poin_id, kategori, status, tipe rekomendasi, target) dan
metadata sitasi (dokumen/pasal/halaman) dirakit DI SINI dari poin/calculator/chunk yang benar-benar
diretrieve — bukan dipercaya dari output LLM. citation_id yang tidak dikenal (halusinasi) dibuang.
"""

from app.reasoning import llm_client
from app.reasoning.calculator import pilih_target_utama_intensitas
from app.reasoning.prompts import SYSTEM_PROMPT, build_user_prompt
from app.reasoning.templates import template_aman
from app.retrieval.base import Chunk, RetrievalFilters, Retriever
from app.schemas import MetaL2, PoinKonteks, PoinOutput, RekomendasiOutput, SitasiOutput

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
    poin: PoinKonteks,
    retriever: Retriever,
    top_k_dukungan: int = 3,
) -> list[Chunk]:
    """Retrieval chunk pendukung utk satu poin: dasar_hukum dulu, fallback search teks kategori."""
    referensi = [f"{d.dokumen} {d.pasal}" for d in poin.dasar_hukum if d.pasal]
    chunks = retriever.get_by_reference(referensi) if referensi else []
    if not chunks:
        chunks = retriever.search(poin.kategori, RetrievalFilters(), top_k=top_k_dukungan)
    return chunks


def _apakah_aman(poin: PoinKonteks) -> bool:
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
    """Hasilkan PoinOutput untuk satu poin. Poin jelas aman -> template (tanpa retrieval/LLM)."""
    if _apakah_aman(poin):
        return template_aman(poin)

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
                kutipan=kutipan or chunk.teks,
                terverifikasi=True,
            )
        )

    target: float | str | None = None
    if poin.tipe_rekomendasi == "numerik":
        target = pilih_target_utama_intensitas(poin.fakta.get("target") or {})

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
            saran=llm_out["saran"],
            disclaimer=llm_out.get("disclaimer"),
        ),
        low_confidence=False,
    )
