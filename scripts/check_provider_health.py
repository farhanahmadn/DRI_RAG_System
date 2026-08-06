"""Cek kesehatan provider embedding/rerank AKTIF (EMBEDDING_PROVIDER/RERANK_PROVIDER di .env) —
sanity call tunggal + stress-test concurrency kecil. Tool OPERASIONAL permanen (bukan sekali-pakai),
utk re-verifikasi manual kapan pun perlu: sebelum/sesudah upgrade tier provider, setelah
PROVIDER_MAX_CONCURRENT diubah, atau curiga ada masalah rate-limit/429 di produksi.

Ditulis setelah insiden konkret (Agustus 2026): observasi live 429 Voyage (tier gratis) & penambahan
semaphore `PROVIDER_MAX_CONCURRENT` setelah stress-test manual serupa — dipermanenkan di sini
supaya pengecekan yang sama tidak perlu ditulis ulang tiap kali dibutuhkan (lihat docs/STATUS_RAG.md
§ "Pembatas concurrency panggilan Jina").

Jalankan dari root repo:  python -m scripts.check_provider_health
Butuh: DATABASE_URL (RetrieverAsli) + API key provider aktif (mis. JINA_API_KEY).
"""

from __future__ import annotations

import os
import threading
import time

from dotenv import load_dotenv

load_dotenv()

from app.retrieval.base import RetrievalFilters  # noqa: E402
from app.retrieval.retriever import RetrieverAsli  # noqa: E402

_WILAYAH = os.getenv("DEMO_WILAYAH", os.getenv("RETRIEVER_WILAYAH", "Sleman Tengah"))
_QUERIES = ["kdb", "dampak tata guna lahan", "kegiatan"]


def _cek_sanity_tunggal(rt: RetrieverAsli) -> bool:
    print("=== 1. Sanity call tunggal ===")
    try:
        t0 = time.perf_counter()
        hasil = rt.search(_QUERIES[0], RetrievalFilters(dokumen=_WILAYAH), top_k=3)
        dur = time.perf_counter() - t0
        print(f"  OK — {len(hasil)} chunk, {dur:.2f}s")
        return True
    except Exception as exc:  # noqa: BLE001 — tool diagnostik, tangkap semua & laporkan
        print(f"  GAGAL: {type(exc).__name__}: {exc}")
        return False


def _cek_concurrency(rt: RetrieverAsli, n_paralel: int = 3) -> bool:
    print(f"\n=== 2. Stress-test concurrency ({n_paralel} query paralel) ===")

    hasil: list[tuple[str, bool, float, str | None]] = []
    lock = threading.Lock()

    def _satu(query: str) -> None:
        t0 = time.perf_counter()
        try:
            r = rt.search(query, RetrievalFilters(dokumen=_WILAYAH), top_k=3)
            with lock:
                hasil.append((query, True, time.perf_counter() - t0, f"{len(r)} chunk"))
        except Exception as exc:  # noqa: BLE001
            with lock:
                hasil.append((query, False, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"))

    threads = [threading.Thread(target=_satu, args=(q,)) for q in (_QUERIES * 2)[:n_paralel]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = True
    for query, sukses, dur, detail in hasil:
        tanda = "OK  " if sukses else "GAGAL"
        print(f"  [{tanda}] {query!r:<28} {dur:.2f}s  {detail}")
        ok = ok and sukses
    return ok


def main() -> None:
    embed_provider = os.getenv("EMBEDDING_PROVIDER", "local")
    rerank_provider = os.getenv("RERANK_PROVIDER", "local")
    max_concurrent = os.getenv("PROVIDER_MAX_CONCURRENT", "2")
    print(f"Provider: EMBEDDING={embed_provider!r} RERANK={rerank_provider!r} "
          f"PROVIDER_MAX_CONCURRENT={max_concurrent}  wilayah={_WILAYAH!r}\n")

    rt = RetrieverAsli(default_wilayah=_WILAYAH)
    ok1 = _cek_sanity_tunggal(rt)
    ok2 = _cek_concurrency(rt)

    print(f"\n=== RINGKASAN: {'SEHAT' if ok1 and ok2 else 'ADA MASALAH — lihat detail di atas'} ===")


if __name__ == "__main__":
    main()
