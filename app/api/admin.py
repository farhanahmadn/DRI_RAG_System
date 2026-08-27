"""Endpoint admin — operasional (bukan reasoning), dilindungi token bersama (`ADMIN_TOKEN` env).

Kenapa endpoint ini perlu ada: cache retrieval (`app/retrieval/cache.py`) itu in-memory PER-PROSES,
dibuat sekali oleh `app/api/dependencies.py::get_retriever` (`@lru_cache`, satu instance retriever
untuk seluruh umur proses API). `scripts/ingest.py`/`scripts/reembed.py` SELALU dijalankan sebagai
proses CLI TERPISAH dari proses API yang sedang melayani request — dua proses OS punya memori
sendiri-sendiri, jadi cache milik proses API TIDAK BISA dikosongkan cuma dengan memanggil
`cache.clear()` di dalam skrip CLI (itu cuma bikin objek cache baru lalu clear di tempat, tak
menyentuh proses API sama sekali). Endpoint HTTP ini satu-satunya jalan sah menjangkau proses API
yang hidup dari proses CLI yang terpisah.

Fail-closed: kalau `ADMIN_TOKEN` tak dikonfigurasi di server, endpoint MENOLAK SEMUA permintaan
(403) — bukan default terbuka tanpa proteksi, mengingat domain ini publik.
"""

import os

from fastapi import Header, HTTPException

from app.retrieval.base import Retriever


def verifikasi_admin_token(x_admin_token: str | None = Header(default=None)) -> None:
    token_server = os.getenv("ADMIN_TOKEN")
    if not token_server:
        raise HTTPException(
            status_code=403,
            detail="ADMIN_TOKEN belum dikonfigurasi di server — endpoint admin dinonaktifkan.",
        )
    if x_admin_token != token_server:
        raise HTTPException(status_code=403, detail="Token admin tidak valid.")


def clear_retrieval_cache(retriever: Retriever) -> dict:
    """Kosongkan cache milik `retriever` yang diberikan. `retriever` HARUS datang dari pemanggil
    lewat Depends(get_retriever) (bukan dipanggil get_retriever() langsung di sini) — supaya
    app.dependency_overrides di test benar-benar tersambung (FastAPI cuma meng-intersep
    pemanggilan via Depends() di parameter route, bukan panggilan fungsi biasa di dalam modul).
    MockRetriever (mode test/dev tanpa DB) tak punya cache sama sekali — ditangani lewat getattr,
    bukan error."""
    clear = getattr(retriever, "clear_cache", None)
    if clear is None:
        return {"status": "ok", "cleared": False, "catatan": "Retriever aktif tak punya cache (mis. mode mock)."}
    clear()
    return {"status": "ok", "cleared": True}
