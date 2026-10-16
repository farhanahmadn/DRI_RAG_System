"""Anggaran waktu per permintaan /reasoning.

Cloudflare (paket Free/Pro/Business) memutus permintaan yang origin-nya belum membalas dalam 100 detik
(HTTP 524, tidak bisa diubah di paket itu), dan reverse proxy VPS bisa lebih ketat lagi (nginx bawaan 60
detik). Tanpa anggaran waktu, satu permintaan yang lambat (Groq kena rate limit, kunci mati, Jina lambat,
percobaan guardrail berulang) bisa memakan lebih dari itu dan BE hanya menerima halaman error HTML
Cloudflare, bukan hasil.

Cara kerja: `jalankan_precheck` membuka `anggaran()`. Setiap lapisan yang bisa lama (panggilan LLM, panggilan
provider retrieval, tidur menunggu rate limit, percobaan ulang guardrail, sintesis) memeriksa `sisa()` dan
berhenti lebih awal dengan `BatasWaktuTerlampaui` alih-alih menunggu. Poin yang belum selesai jatuh ke
fallback deterministik (`low_confidence=True`), jadi respons 200 yang JUJUR tetap datang sebelum Cloudflare
memutus. Tanpa `anggaran()` aktif (mis. skrip, tes), `sisa()` adalah None dan perilaku lama tidak berubah.

`ContextVar` dipakai supaya anggaran ikut ke thread pekerja tanpa mengubah tanda tangan fungsi: pemanggil
yang memindahkan kerja ke thread harus memakai `contextvars.copy_context().run(...)`.
"""

from __future__ import annotations

import contextvars
import os
import time
from contextlib import contextmanager
from typing import Iterator

# 85 detik: menyisakan ~15 detik dari batas 100 detik Cloudflare untuk perakitan akhir, jaringan, dan
# serialisasi. 0 atau negatif = anggaran dimatikan.
DURASI_BAWAAN_S = 85.0

_batas: contextvars.ContextVar[float | None] = contextvars.ContextVar("batas_waktu_reasoning", default=None)


class BatasWaktuTerlampaui(RuntimeError):
    """Anggaran waktu permintaan habis (atau terlalu sedikit untuk satu langkah lagi)."""


def durasi_dari_env() -> float | None:
    mentah = os.getenv("REASONING_DEADLINE_S")
    try:
        nilai = float(mentah) if mentah not in (None, "") else DURASI_BAWAAN_S
    except ValueError:
        nilai = DURASI_BAWAAN_S
    return nilai if nilai > 0 else None


@contextmanager
def anggaran(detik: float | None = None) -> Iterator[None]:
    """Buka anggaran waktu untuk blok ini. `detik=None` -> baca REASONING_DEADLINE_S."""
    durasi = durasi_dari_env() if detik is None else (detik if detik > 0 else None)
    token = _batas.set(None if durasi is None else time.monotonic() + durasi)
    try:
        yield
    finally:
        _batas.reset(token)


def sisa() -> float | None:
    """Detik tersisa, atau None bila tak ada anggaran aktif. Bisa negatif bila sudah lewat."""
    batas = _batas.get()
    return None if batas is None else batas - time.monotonic()


def habis(minimal: float = 0.0) -> bool:
    """True bila anggaran aktif dan sisanya <= `minimal`."""
    s = sisa()
    return s is not None and s <= minimal


def periksa(minimal: float, langkah: str) -> None:
    """Raise BatasWaktuTerlampaui bila sisa anggaran kurang dari `minimal` detik."""
    s = sisa()
    if s is not None and s < minimal:
        raise BatasWaktuTerlampaui(
            f"Sisa anggaran waktu {max(s, 0):.1f}s kurang dari {minimal:.1f}s yang dibutuhkan untuk: {langkah}"
        )
