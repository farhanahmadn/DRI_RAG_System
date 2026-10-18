"""Kumpulan API key Jina dengan failover berurutan (bukan round-robin).

Satu key dipakai terus sampai HABIS (saldo/kuota token Jina habis atau key ditolak), baru pindah ke key
berikutnya. Key yang sudah ditandai habis tidak dicoba lagi sampai masa tunggu (`JINA_KEY_COOLDOWN_S`,
bawaan 1 jam) lewat, supaya key yang sudah diisi ulang bisa dipakai kembali tanpa restart.

Beda dari rotasi Groq (`app/reasoning/llm_client.py`) yang round-robin: kuota Jina berupa saldo token
total per key, tidak per menit, jadi menghabiskan satu key dulu lebih sederhana dan mudah dipantau.

Konfigurasi: `JINA_API_KEYS` (dipisah koma, urutan = urutan pemakaian), fallback ke `JINA_API_KEY` tunggal.
Yang dicatat ke log hanya nomor urut key, tidak pernah isinya.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_COOLDOWN_BAWAAN_S = 3600.0


class SemuaKunciJinaHabisError(RuntimeError):
    """Tidak ada key Jina yang bisa dipakai: semua habis saldo/kuota atau ditolak."""


def kunci_habis(status_code: int, teks: str) -> bool:
    """Apakah balasan HTTP menandakan key ini HABIS/DITOLAK (bukan gangguan sesaat)?

    402 = saldo habis. 401/403 = key ditolak. 429 dihitung habis HANYA bila pesannya menyebut saldo/kuota
    (429 biasa = batas laju per menit, cukup ditunggu lalu diulang oleh retry transport)."""
    if status_code in (401, 402, 403):
        return True
    if status_code == 429:
        t = (teks or "").lower()
        return any(k in t for k in ("balance", "insufficient", "quota", "exceeded your"))
    return False


class KumpulanKunciJina:
    def __init__(self, kunci: list[str], cooldown_s: float | None = None):
        self._kunci = [k for k in (x.strip() for x in kunci) if k]
        self._cooldown = _COOLDOWN_BAWAAN_S if cooldown_s is None else cooldown_s
        self._aktif = 0
        self._habis_sampai: dict[int, float] = {}
        self._lock = threading.Lock()

    def jumlah(self) -> int:
        return len(self._kunci)

    def _hidup_nolock(self, idx: int) -> bool:
        sampai = self._habis_sampai.get(idx)
        if sampai is None:
            return True
        if time.monotonic() >= sampai:
            del self._habis_sampai[idx]  # masa tunggu lewat: boleh dicoba lagi
            return True
        return False

    def aktif(self) -> tuple[int, str]:
        """(index, key) yang sedang dipakai. Tetap di key yang sama selama masih hidup; bila sudah habis,
        maju ke key hidup berikutnya (melingkar). Raise `SemuaKunciJinaHabisError` bila semua habis."""
        with self._lock:
            n = len(self._kunci)
            for langkah in range(n):
                idx = (self._aktif + langkah) % n
                if self._hidup_nolock(idx):
                    self._aktif = idx
                    return idx, self._kunci[idx]
        raise SemuaKunciJinaHabisError(
            f"Semua {len(self._kunci)} API key Jina habis atau ditolak. Isi saldo di "
            "https://jina.ai/api-dashboard/key-manager atau tambahkan key baru ke JINA_API_KEYS."
        )

    def tandai_habis(self, idx: int, alasan: str) -> None:
        with self._lock:
            baru = self._hidup_nolock(idx)
            self._habis_sampai[idx] = time.monotonic() + self._cooldown
        if baru:
            logger.warning("API key Jina index %d ditandai habis (%s), pindah ke key berikutnya bila ada.",
                           idx, alasan)

    def jumlah_hidup(self) -> int:
        with self._lock:
            return sum(1 for i in range(len(self._kunci)) if self._hidup_nolock(i))


def muat_kunci_dari_env() -> list[str]:
    jamak = os.getenv("JINA_API_KEYS", "")
    kunci = [k.strip() for k in jamak.split(",") if k.strip()]
    if kunci:
        return kunci
    tunggal = os.getenv("JINA_API_KEY", "").strip()
    return [tunggal] if tunggal else []


_kumpulan: KumpulanKunciJina | None = None
_kunci_env_terakhir: tuple[str, str] | None = None
_kunci_lock = threading.Lock()


def get_kumpulan() -> KumpulanKunciJina:
    """Singleton per proses (embed dan rerank memakai akun yang sama). Dibangun ulang bila isi env berubah."""
    global _kumpulan, _kunci_env_terakhir
    sidik = (os.getenv("JINA_API_KEYS", ""), os.getenv("JINA_API_KEY", ""))
    with _kunci_lock:
        if _kumpulan is None or sidik != _kunci_env_terakhir:
            kunci = muat_kunci_dari_env()
            if not kunci:
                raise RuntimeError("JINA_API_KEYS/JINA_API_KEY tidak ditemukan di environment (provider jina).")
            cooldown = float(os.getenv("JINA_KEY_COOLDOWN_S", str(_COOLDOWN_BAWAAN_S)))
            _kumpulan = KumpulanKunciJina(kunci, cooldown)
            _kunci_env_terakhir = sidik
        return _kumpulan
