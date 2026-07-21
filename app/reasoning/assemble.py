"""Perakit OutputPreCheck penuh — entrypoint terakhir sebelum JSON dikirim ke web.

Prinsip 3 CLAUDE.md: skor & level komposit adalah ranah rule engine (back-end), BUKAN dihitung
ulang atau dikarang di sini. Kalimat ringkasan & langkah_berdampak dirakit deterministik dari data
poin yang sudah ada (termasuk rekomendasi.saran yang sudah lolos guardrail) — tanpa panggilan LLM
tambahan di jalur ini.

Indikator diproses PARALEL (ThreadPoolExecutor) — panggilan Groq itu I/O-bound, GIL dilepas saat
menunggu socket, jadi threading beri speedup nyata tanpa perlu menulis ulang seluruh chain jadi
async (yang akan memaksa ubah SEAM Retriever Protocol, kontrak dengan teman).
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from app.logging_util import log_precheck
from app.reasoning import observability
from app.reasoning.guardrail import generate_poin_dengan_guardrail
from app.reasoning.templates import template_low_confidence
from app.retrieval.base import Retriever
from app.schemas import (
    IndikatorJejak,
    JejakAturanRequest,
    KesimpulanOutput,
    OutputPreCheck,
    PoinOutput,
    RingkasanOutput,
)

logger = logging.getLogger(__name__)

_LEVEL_BELUM_TERSEDIA = "BELUM_DITENTUKAN (menunggu level dari back-end)"

_MAX_WORKERS = int(os.getenv("REASONING_MAX_WORKERS", "8"))
_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="reasoning-poin")


def _generate_poin_aman(indikator: IndikatorJejak, retriever: Retriever) -> PoinOutput:
    """Lapis pertahanan tambahan: guardrail.py seharusnya tidak pernah raise, tapi kalau suatu
    saat ada bug tak terduga, batch tidak boleh gagal total gara-gara 1 indikator."""
    try:
        return generate_poin_dengan_guardrail(indikator, retriever)
    except Exception:
        logger.exception(
            "generate_poin_dengan_guardrail gagal tak terduga utk %s — fallback low_confidence.",
            indikator.poin_id,
        )
        return template_low_confidence(indikator)


def _rakit_ringkasan(request: JejakAturanRequest, poin_list: list[PoinOutput]) -> RingkasanOutput:
    if request.level is not None:
        level = request.level
    else:
        level = _LEVEL_BELUM_TERSEDIA
        logger.warning(
            "JejakAturanRequest tidak membawa 'level' — pakai sentinel %r. Minta back-end kirim "
            "level komposit, jangan dihitung ulang di sini.",
            _LEVEL_BELUM_TERSEDIA,
        )

    berisiko = [p for p in poin_list if p.status != "Aman"]
    if not berisiko:
        kalimat = (
            f"Berdasarkan hasil pemeriksaan, tidak ditemukan indikator berisiko dari "
            f"{len(poin_list)} indikator yang diperiksa."
        )
    else:
        daftar_kategori = ", ".join(p.kategori for p in berisiko)
        kalimat = (
            f"Ditemukan {len(berisiko)} dari {len(poin_list)} indikator berisiko: "
            f"{daftar_kategori}. Skor risiko total: {request.skor_total}."
        )

    return RingkasanOutput(skor_total=request.skor_total, level=level, kalimat=kalimat)


def _rakit_kesimpulan(poin_list: list[PoinOutput]) -> KesimpulanOutput:
    berisiko = [p for p in poin_list if p.status != "Aman"]

    langkah_berdampak = [f"{p.kategori}: {p.rekomendasi.saran}" for p in berisiko]

    lokasional_berisiko = [p for p in berisiko if p.rekomendasi.tipe == "lokasional"]
    if lokasional_berisiko:
        daftar_kategori = ", ".join(p.kategori for p in lokasional_berisiko)
        catatan_lokasi = f"Lokasi berkaitan dengan faktor risiko lokasional: {daftar_kategori}."
    else:
        catatan_lokasi = None

    return KesimpulanOutput(langkah_berdampak=langkah_berdampak, catatan_lokasi=catatan_lokasi)


def jalankan_precheck(request: JejakAturanRequest, retriever: Retriever) -> OutputPreCheck:
    """Jalankan precheck penuh: generate tiap poin PARALEL (dengan guardrail), rakit ringkasan/kesimpulan."""
    futures = [
        _executor.submit(_generate_poin_aman, indikator, retriever)
        for indikator in request.indikator
    ]
    poin_list = [f.result() for f in futures]  # urutan submit == urutan hasil, walau selesai konkuren

    output = OutputPreCheck(
        ringkasan=_rakit_ringkasan(request, poin_list),
        poin=poin_list,
        kesimpulan=_rakit_kesimpulan(poin_list),
    )

    try:
        log_precheck(request, output)
    except Exception:
        logger.exception("Gagal menulis log precheck — melanjutkan tanpa menggagalkan respons.")

    observability.catat_precheck_trace(
        "jalankan_precheck", request.model_dump(mode="json"), output.model_dump(mode="json")
    )

    return output
