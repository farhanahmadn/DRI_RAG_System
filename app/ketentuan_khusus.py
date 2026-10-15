"""Penentuan status ketentuan khusus RDTR dari data lokasi (blok & overlay BE).

RDTR Sleman Tengah menetapkan ketentuan khusus PER BLOK, bukan per zona:
- Pasal 53 ayat (1): kawasan resapan air.
- Pasal 50 ayat (10): kawasan rawan bencana kekeringan.

BE (permintaan 2026-10-15, docs/PERMINTAAN_BE_L2_2026-10-15.md) mengirim ID blok dan/atau hasil overlay
peta Lampiran VII. Urutan keputusan, dari yang paling otoritatif:

1. Hasil overlay BE yang eksplisit (`lokasi.ketentuan_khusus.*` lalu `lokasi.kawasan_resapan`).
2. Dasar ambang intensitas dari BE (`tahapan.intensitas.kawasan_resapan`), khusus resapan.
3. Daftar blok dari teks peraturan (hanya bila wilayah yang dilayani Sleman Tengah, karena daftar blok
   itu milik dokumen Sleman Tengah).
4. Tidak diketahui (None) -> perilaku lama, TIDAK menebak.
"""

from __future__ import annotations

import logging
import os

from app.schemas import L2Assessment

logger = logging.getLogger(__name__)

# Pasal 53 ayat (1) Perbup 80/2023 (RDTR Kawasan Sleman Tengah).
BLOK_RESAPAN_SLEMAN_TENGAH = frozenset({
    "I.B.1", "I.B.2", "I.B.3", "I.B.4", "I.B.5", "I.B.6",
    "I.C.1", "I.C.2", "I.C.3", "I.C.5", "I.C.6", "I.C.7",
    "I.F.3",
})

# Pasal 50 ayat (10) Perbup 80/2023: kawasan rawan bencana kekeringan tingkat tinggi.
BLOK_KEKERINGAN_SLEMAN_TENGAH = frozenset({"I.E.1", "I.E.2"})


def _wilayah_tengah() -> bool:
    return "tengah" in os.getenv("RETRIEVER_WILAYAH", "Sleman Tengah").lower()


def normalisasi_blok(blok: str | None) -> str | None:
    """"i.b.2" / " I.B.2 " / "B.2" -> "I.B.2". None bila kosong."""
    if not blok or not blok.strip():
        return None
    b = blok.strip().upper().replace(" ", "")
    # "B.2" (tanpa awalan "I.") -> "I.B.2". Pola RDTR: <I>.<SWP>.<nomor>.
    bagian = b.split(".")
    if len(bagian) == 2 and len(bagian[0]) == 1 and bagian[0].isalpha():
        b = f"I.{b}"
    return b


def _daftar_blok(assessment: L2Assessment) -> list[str]:
    lokasi = assessment.lokasi
    mentah = list(lokasi.blok_list or [])
    if lokasi.blok:
        mentah.insert(0, lokasi.blok)
    hasil: list[str] = []
    for b in mentah:
        n = normalisasi_blok(b)
        if n and n not in hasil:
            hasil.append(n)
    return hasil


def _dari_blok(assessment: L2Assessment, himpunan: frozenset[str]) -> bool | None:
    """True bila SEMUA blok lokasi ada di himpunan, False bila tak satu pun, None bila campuran atau
    blok tak diketahui atau wilayah bukan Sleman Tengah."""
    if not _wilayah_tengah():
        return None
    blok = _daftar_blok(assessment)
    if not blok:
        return None
    di_dalam = [b in himpunan for b in blok]
    if all(di_dalam):
        return True
    if not any(di_dalam):
        return False
    return None  # poligon menyilang blok resapan dan non-resapan -> tak bisa dipastikan


def tentukan_kawasan_resapan(assessment: L2Assessment) -> bool | None:
    lokasi = assessment.lokasi
    khusus = lokasi.ketentuan_khusus or {}
    kandidat = [
        khusus.get("kawasan_resapan_air"),
        lokasi.kawasan_resapan,
        getattr(assessment.gate_hukum.tahapan.intensitas, "kawasan_resapan", None)
        if assessment.gate_hukum.tahapan.intensitas else None,
    ]
    eksplisit = next((k for k in kandidat if k is not None), None)
    turunan = _dari_blok(assessment, BLOK_RESAPAN_SLEMAN_TENGAH)
    if eksplisit is not None:
        if turunan is not None and turunan != eksplisit:
            logger.warning(
                "kawasan_resapan dari BE (%s) berbeda dari daftar blok Pasal 53 (%s) untuk blok %s — "
                "memakai nilai dari BE.", eksplisit, turunan, _daftar_blok(assessment),
            )
        return eksplisit
    return turunan


def tentukan_rawan_kekeringan(assessment: L2Assessment) -> bool | None:
    eksplisit = (assessment.lokasi.ketentuan_khusus or {}).get("rawan_kekeringan")
    if eksplisit is not None:
        return eksplisit
    return _dari_blok(assessment, BLOK_KEKERINGAN_SLEMAN_TENGAH)
