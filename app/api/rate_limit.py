"""Rate limiting sederhana per-IP untuk /reasoning — in-memory, sliding window, tanpa dependency baru.

MVP single-instance (CLAUDE.md: "mulai ramping, tambah pengaman saat eval membuktikan perlu").
Kalau nanti multi-worker/multi-instance, state ini TIDAK dibagi antar proses — perlu diganti ke
limiter terpusat (mis. Redis) saat itu jadi masalah nyata, bukan sekarang.

/health SENGAJA tidak dibatasi (monitoring perlu akses bebas) — dependency ini hanya dipasang di
route /reasoning.
"""

import os
import threading
import time
from collections import defaultdict

from fastapi import HTTPException, Request

_LIMIT = int(os.getenv("RATE_LIMIT_PER_MENIT", "10"))
_WINDOW_S = 60.0

_lock = threading.Lock()
_riwayat: dict[str, list[float]] = defaultdict(list)


def reset_rate_limiter() -> None:
    """Utilitas test — bersihkan seluruh state limiter."""
    with _lock:
        _riwayat.clear()


def cek_rate_limit(request: Request) -> None:
    """Dependency FastAPI — raise 429 kalau IP klien melebihi _LIMIT permintaan per _WINDOW_S detik."""
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()

    with _lock:
        timestamps = _riwayat[ip]
        cutoff = now - _WINDOW_S
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)

        if len(timestamps) >= _LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"Terlalu banyak permintaan. Maksimum {_LIMIT} per {_WINDOW_S:.0f} detik.",
            )

        timestamps.append(now)
