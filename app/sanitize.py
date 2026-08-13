"""app/sanitize.py — pagar EKSPLISIT sebelum teks dikirim keluar (query retrieval ke provider
embedding/rerank eksternal — Jina — DAN prompt LLM Groq).

KENAPA modul terpisah (bukan cuma "aman by design"): arsitektur saat ini KEBETULAN tidak pernah
mengalirkan field sensitif ke titik ini — `app/adapter.py` tidak pernah menyalin
`application_number`/`application_id`/`lokasi.koordinat` ke `PoinKonteks.fakta`, dan
`retriever.search()`/`get_by_reference()` hanya menerima kategori indikator pendek / rujukan pasal
(lihat app/reasoning/generator.py::ambil_chunks_pendukung). Tapi itu properti EMERGENT dari cara
kode kebetulan ditulis, bukan sesuatu yang DIJAGA aktif — field baru yang ditambahkan ke
`PoinKonteks.fakta` suatu hari nanti (developer lain, tanpa sadar risikonya) bisa bocor ke prompt/API
eksternal tanpa ada apa pun yang mencegahnya. Modul ini adalah gerbang yang DIPANGGIL di titik
pembentukan teks keluar, bukan diasumsikan.

Field TERLARANG (nama field — di-strip dari dict/di-tolak eksplisit kalau muncul; sebagian belum ada
di skema saat ini tapi didaftar PREVENTIF untuk kalau ditambah nanti):
  - nama pemohon, NIK
  - application_number, application_id
  - koordinat presisi (lat, lon, geojson coordinates)

Field free-text yang WAJIB di-scan pola PII (di-scrub, bukan ditolak mentah — field ini memang bahan
reasoning yang sah, ditulis petugas, cuma BERPOTENSI memuat catatan personal):
  - itbx.reason, itbx.keterangan_ketentuan (app/schemas.py::ItbxTahap)

Field DIIZINKAN ikut ke query retrieval / prompt (allowlist per poin_id di `ALLOWED_FAKTA_FIELDS`):
  - kategori indikator / poin_id, zona, angka fakta poin (KDB/KLB/KDH usulan-ambang, skor kategori
    dampak — skor mentah SUDAH sengaja tak disuntik LLM, lihat app/reasoning/prompts.py), dasar_hukum
    (dokumen/pasal/kutipan pasal — kutipan REGULASI PUBLIK, bukan data personal pemohon).
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------- nama field terlarang
# Dicek case-insensitive, exact match nama key (dict) — TIDAK substring, supaya field legit seperti
# "kategori"/"koordinat_zona_x" (kalau ada) tak salah kena. Preventif: sebagian (nama/nik) belum ada
# di app/schemas.py saat ini, didaftar untuk kalau field itu ditambahkan nanti.
FORBIDDEN_FIELD_NAMES = frozenset({
    "nama", "nama_pemohon", "applicant_name", "nik",
    "application_number", "application_id",
    "koordinat", "lat", "lon", "latitude", "longitude", "coordinates", "geojson",
})

# --------------------------------------------------------------------- allowlist fakta per poin
# Dipakai app/reasoning/prompts.py sebelum menyusun prompt — apa pun key di `poin.fakta` yang TIDAK
# ada di sini otomatis DIBUANG (fail-safe: field baru yang lupa didaftar = tidak pernah lolos, bukan
# tidak sengaja lolos). Selaras persis dengan yang dibaca app/adapter.py per poin_id saat ini.
ALLOWED_FAKTA_FIELDS: dict[str, frozenset[str]] = {
    "itbx": frozenset({
        "lolos", "kbli_diusulkan", "kegiatan_diusulkan", "kegiatan_diizinkan", "kegiatan_terbatas",
        "kegiatan_bersyarat", "kegiatan_terbatas_bersyarat", "keterangan_ketentuan", "reason",
        "fallback_data_kosong",
    }),
    "intensitas": frozenset({
        "dinilai", "parameter", "luas_tapak_m2", "jumlah_lantai", "luas_rth_usulan_m2", "target",
    }),
    "dampak": frozenset({
        "dinilai", "impact_score", "runoff_change_index", "c_before", "c_after", "threshold_bands",
        "existing_surface_details", "proposed_surface_details", "c_coefficients", "data_confidence",
        "limitations", "mitigasi", "target_mitigasi",
    }),
}

# Key di dalam `fakta` yang isinya free-text tulisan petugas (bukan angka/enum) — WAJIB lewat
# `sanitize_freetext` sebelum masuk prompt, terlepas dari poin_id.
_FREETEXT_FAKTA_KEYS = frozenset({"reason", "keterangan_ketentuan", "limitations"})

# --------------------------------------------------------------------- pola PII (heuristik, regex)
# Best-effort, BUKAN deteksi PII sempurna (tidak ada NLP/NER di sini) — cukup untuk menangkap pola
# yang JELAS (NIK 16-digit, nomor aplikasi APP-YYYY-NNNN, pasangan lat/lon presisi, email, telp ID).
_RE_NIK = re.compile(r"\b\d{16}\b")
_RE_APPLICATION_NUMBER = re.compile(r"\bAPP-\d{4}-\d+\b", re.IGNORECASE)
_RE_COORD_PAIR = re.compile(r"-?\d{1,3}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}")
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_RE_PHONE_ID = re.compile(r"\b(?:\+62|62|0)8\d{8,11}\b")

_PII_PATTERNS = (
    ("NIK", _RE_NIK),
    ("application_number", _RE_APPLICATION_NUMBER),
    ("koordinat", _RE_COORD_PAIR),
    ("email", _RE_EMAIL),
    ("telepon", _RE_PHONE_ID),
)


class PIIDetectedError(ValueError):
    """Teks/field yang akan dikirim keluar (query retrieval / prompt LLM) mengandung pola PII."""


# --------------------------------------------------------------------- API publik
def sanitize_query_text(text: str, *, context: str = "query") -> str:
    """Validasi string query retrieval (dikirim ke provider embedding eksternal) SEBELUM dipakai.

    Dipanggil di titik pembentukan query: `app/retrieval/retriever.py::RetrieverAsli.search()` &
    `get_by_reference()`. Query di sistem ini SEHARUSNYA selalu kategori pendek (mis. "kdb") atau
    rujukan pasal (mis. "RDTR Sleman Tengah Pasal 43") — kalau pola PII ketemu di sini, itu tanda
    ada field salah yang bocor ke pemanggil (bug di layer atas), bukan sesuatu yang boleh diloloskan
    diam-diam. Raise `PIIDetectedError`, JANGAN scrub-lalu-lanjut (query yang di-scrub diam-diam bisa
    salah arti & retrieval jadi salah — lebih aman gagal keras & ketahuan dari log/test).
    """
    for label, pat in _PII_PATTERNS:
        if pat.search(text):
            raise PIIDetectedError(f"Pola PII ({label}) terdeteksi di {context}: {text[:80]!r}...")
    return text


def sanitize_freetext(text: str, *, context: str = "freetext") -> str:
    """Scrub (redaksi) pola PII dari teks bebas yang MEMANG boleh masuk prompt LLM (mis.
    `itbx.reason`/`keterangan_ketentuan` — tulisan petugas, bahan reasoning yang sah). Beda dgn
    `sanitize_query_text`: di sini SCRUB lalu lanjut (bukan raise) — field ini legit isinya bahan
    regulasi, menolak mentah-mentah bisa merusak reasoning; redaksi cukup untuk mencegah kebocoran.
    """
    if not text:
        return text
    out = text
    for label, pat in _PII_PATTERNS:
        out = pat.sub(f"[REDACTED:{label}]", out)
    return out


def sanitize_fakta(poin_id: str, fakta: dict[str, Any]) -> dict[str, Any]:
    """Filter `poin.fakta` sebelum masuk prompt LLM (app/reasoning/prompts.py) — ALLOWLIST per
    poin_id (`ALLOWED_FAKTA_FIELDS`): key yang tidak terdaftar DIBUANG (fail-safe terhadap field
    baru yang lupa diaudit). Key free-text yang tersisa (`_FREETEXT_FAKTA_KEYS`) di-scrub via
    `sanitize_freetext` (list of str di-scrub per-item).
    """
    allowed = ALLOWED_FAKTA_FIELDS.get(poin_id)
    if allowed is None:
        # poin_id tak dikenal allowlist-nya -> JANGAN tebak, JANGAN loloskan apa pun by default.
        return {}
    out: dict[str, Any] = {}
    for key, value in fakta.items():
        if key not in allowed:
            continue
        # Nama field terlarang tidak boleh lolos meski (secara keliru) sempat masuk allowlist masa
        # depan — pertahanan lapis kedua.
        if key.lower() in FORBIDDEN_FIELD_NAMES:
            continue
        if key in _FREETEXT_FAKTA_KEYS:
            if isinstance(value, str):
                value = sanitize_freetext(value, context=f"fakta.{key}")
            elif isinstance(value, list):
                value = [sanitize_freetext(v, context=f"fakta.{key}[]") if isinstance(v, str) else v
                         for v in value]
        out[key] = value
    return out
