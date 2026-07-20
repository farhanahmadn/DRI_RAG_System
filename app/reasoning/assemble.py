"""Perakit OutputPreCheck penuh — entrypoint terakhir sebelum JSON dikirim ke web.

Prinsip 3 CLAUDE.md: skor & level komposit adalah ranah rule engine (back-end), BUKAN dihitung
ulang atau dikarang di sini. Kalimat ringkasan & langkah_berdampak dirakit deterministik dari data
poin yang sudah ada (termasuk rekomendasi.saran yang sudah lolos guardrail) — tanpa panggilan LLM
tambahan di jalur ini.
"""

import logging

from app.logging_util import log_precheck
from app.reasoning.guardrail import generate_poin_dengan_guardrail
from app.retrieval.base import Retriever
from app.schemas import (
    JejakAturanRequest,
    KesimpulanOutput,
    OutputPreCheck,
    PoinOutput,
    RingkasanOutput,
)

logger = logging.getLogger(__name__)

_LEVEL_BELUM_TERSEDIA = "BELUM_DITENTUKAN (menunggu level dari back-end)"


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
    """Jalankan precheck penuh: generate tiap poin (dengan guardrail), rakit ringkasan/kesimpulan."""
    poin_list = [
        generate_poin_dengan_guardrail(indikator, retriever) for indikator in request.indikator
    ]

    output = OutputPreCheck(
        ringkasan=_rakit_ringkasan(request, poin_list),
        poin=poin_list,
        kesimpulan=_rakit_kesimpulan(poin_list),
    )

    try:
        log_precheck(request, output)
    except Exception:
        logger.exception("Gagal menulis log precheck — melanjutkan tanpa menggagalkan respons.")

    return output
