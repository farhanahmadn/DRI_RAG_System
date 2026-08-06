"""app/retrieval/cache.py — cache in-memory (TTL) untuk hasil retrieval+rerank final.

Ruang query di sistem ini kecil & repetitif: kategori indikator terbatas (~8: KDB/KLB/KDH/LP2B/
Kegiatan/Banjir/Resapan/Sempadan Sungai) x zona (puluhan) x dokumen/wilayah (3), diulang antar
permohonan berbeda yang kebetulan sezona/sekategori. Cache di sini memangkas panggilan
embed+rerank berulang (termasuk ke API eksternal Jina yang berbayar per-panggilan) untuk
kombinasi yang sudah pernah dihitung — bukan cache "benar selamanya": TTL membatasi umur entri
supaya perubahan regulasi (jarang, tapi ada) tetap ter-refresh dalam waktu wajar.

Per-proses (bukan distributed/Redis) — cukup utk beban 1 instance produksi & selaras skala sistem
ini; kalau nanti multi-instance jadi masalah nyata, ganti implementasi tanpa ubah pemanggil
(`get_or_compute` tetap kontraknya).
"""

from __future__ import annotations

import os
import time
from threading import Lock
from typing import Callable, TypeVar

T = TypeVar("T")


class TTLCache:
    """Cache key->value dgn TTL & batas ukuran (buang entri TERLAMA saat penuh — cukup sederhana
    utk ruang query kecil di sistem ini, bukan LRU sungguhan)."""

    def __init__(self, ttl_s: float, max_entries: int, enabled: bool = True) -> None:
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._enabled = enabled
        self._store: dict[tuple, tuple[float, object]] = {}
        self._lock = Lock()

    def get_or_compute(self, key: tuple, compute: Callable[[], T]) -> T:
        """Ambil dari cache kalau ada & belum kedaluwarsa; kalau tidak, panggil `compute()`, simpan,
        lalu kembalikan. `compute()` dipanggil DI LUAR lock (supaya panggilan lambat/API eksternal
        tak memblokir pembaca lain thread cache)."""
        if not self._enabled:
            return compute()
        now = time.monotonic()
        with self._lock:
            hit = self._store.get(key)
            if hit is not None and now - hit[0] < self._ttl_s:
                return hit[1]
        value = compute()
        with self._lock:
            if key not in self._store and len(self._store) >= self._max_entries:
                self._store.pop(next(iter(self._store)), None)  # buang entri terlama (insertion order)
            self._store[key] = (now, value)
        return value

    def clear(self) -> None:
        """Kosongkan cache (test helper / operasional manual)."""
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled, "n_entries": len(self._store),
                "ttl_s": self._ttl_s, "max_entries": self._max_entries,
            }


def from_env(prefix: str = "RETRIEVAL_CACHE") -> TTLCache:
    """Factory baca konfigurasi dari env (pola sama seperti LLM_TIMEOUT_S/LLM_MAX_RETRIES)."""
    enabled = os.getenv(f"{prefix}_ENABLED", "true").strip().lower() not in ("0", "false", "no")
    ttl_s = float(os.getenv(f"{prefix}_TTL_S", "3600"))
    max_entries = int(os.getenv(f"{prefix}_MAX_ENTRIES", "2000"))
    return TTLCache(ttl_s=ttl_s, max_entries=max_entries, enabled=enabled)
