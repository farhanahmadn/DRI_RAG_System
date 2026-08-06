"""scripts/reembed.py — re-embed chunk yang SUDAH ter-ingest pakai provider kandidat A/B.

Beda dengan `app/ingest/ingest.py` (embed pertama kali dari `chunks_prosa.jsonl` mentah): script ini
membaca `teks_prefixed` yang SUDAH ada di tabel `chunks` (hasil ingest bge-m3/local) dan menghasilkan
vektor provider lain (Jina) TANPA menyentuh `chunks.embedding` (baseline) sama sekali — hasil
disimpan di tabel sisi `chunk_embeddings_ab` (chunk_id, embedding_provider) sehingga bge-m3 dan Jina
bisa hidup berdampingan untuk evaluasi perbandingan (Tahap 5), bukan jalur produksi.

Idempoten & resumable:
  - Upsert per baris (`ON CONFLICT (chunk_id, embedding_provider) DO UPDATE`) — aman dijalankan ulang.
  - Default LEWATI chunk yang sudah punya vektor provider ini (resume otomatis kalau proses berhenti
    di tengah jalan); pakai --force untuk re-embed ulang semuanya.
  - Commit per batch (bukan di akhir) — progres tersimpan permanen meski proses dihentikan (Ctrl+C).

PERINGATAN: jalankan dulu dgn --limit/--dokumen-id kecil (subset gold-set/dummy) sebelum menjalankan
tanpa batas ke seluruh korpus — setiap baris = 1 panggilan API berbayar ke provider eksternal.

Prasyarat: DATABASE_URL di .env ; JINA_API_KEY terisi ;
`pip install -e ".[rag-serverless]"` (tak butuh torch/transformers utk provider serverless).

CLI:
  python -m scripts.reembed --provider jina --dokumen-id perbup-sleman-80-2023 --limit 20 --dry-run
  python -m scripts.reembed --provider jina --dokumen-id perbup-sleman-80-2023 --limit 20
  python -m scripts.reembed --provider jina                        # SELURUH korpus — konfirmasi dulu!
"""

from __future__ import annotations

import argparse
import os
import time

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

_VALID_PROVIDERS = ("jina",)


def _connect():
    import psycopg
    from pgvector.psycopg import register_vector

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL belum diset di .env")
    conn = psycopg.connect(dsn)
    register_vector(conn)
    return conn


def _select_candidates(conn, dokumen_ids: list[str] | None, limit: int | None,
                       provider: str, force: bool) -> list[tuple[str, str]]:
    """Return [(chunk_id, teks_prefixed)] yang berembedding baseline & (kecuali --force) BELUM
    punya vektor provider ini di chunk_embeddings_ab."""
    conds = ["embedding IS NOT NULL"]
    params: list = []
    if dokumen_ids:
        conds.append("dokumen_id = ANY(%s)")
        params.append(dokumen_ids)
    if not force:
        conds.append(
            "id NOT IN (SELECT chunk_id FROM chunk_embeddings_ab WHERE embedding_provider = %s)"
        )
        params.append(provider)
    sql = f"SELECT id, teks_prefixed FROM chunks WHERE {' AND '.join(conds)} ORDER BY id"
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [(r[0], r[1]) for r in cur.fetchall()]


_UPSERT = """
INSERT INTO chunk_embeddings_ab (chunk_id, embedding_provider, embedding_model, embedding)
VALUES (%(chunk_id)s, %(provider)s, %(model)s, %(embedding)s)
ON CONFLICT (chunk_id, embedding_provider) DO UPDATE SET
    embedding_model = EXCLUDED.embedding_model,
    embedding = EXCLUDED.embedding,
    created_at = now()
"""


def run(provider: str, *, dokumen_ids: list[str] | None = None, limit: int | None = None,
        batch_size: int = 32, sleep_s: float = 0.5, force: bool = False, dry_run: bool = False,
        reindex: bool = False, encode_dense=None) -> dict:
    """Jalankan re-embed. `encode_dense` injectable (untuk test — hindari panggilan API sungguhan)."""
    if provider not in _VALID_PROVIDERS:
        raise SystemExit(f"--provider harus salah satu dari {_VALID_PROVIDERS}, dapat {provider!r}")

    os.environ["EMBEDDING_PROVIDER"] = provider  # embeddings.py baca provider dari env (SATU sumber)
    from app.retrieval.embeddings import current_model_name

    if encode_dense is None:
        from app.retrieval.embeddings import encode_dense  # impor lazy (butuh torch kalau provider=local)
    model_name = current_model_name(provider)

    conn = _connect()
    try:
        rows = _select_candidates(conn, dokumen_ids, limit, provider, force)
        print(f"[reembed] provider={provider} model={model_name} kandidat={len(rows)} "
              f"(force={force}, dokumen_ids={dokumen_ids or 'SEMUA'}, limit={limit or 'tanpa batas'})")
        if dry_run:
            print("[reembed] --dry-run: tidak memanggil API / menulis DB.")
            return {"provider": provider, "model": model_name, "n_kandidat": len(rows), "n_upsert": 0}

        n_upsert = 0
        for i in range(0, len(rows), batch_size):
            page = rows[i:i + batch_size]
            ids = [r[0] for r in page]
            texts = [r[1] for r in page]
            vecs = encode_dense(texts, batch_size=len(texts), input_type="document")
            with conn.cursor() as cur:
                for cid, vec in zip(ids, vecs):
                    cur.execute(_UPSERT, {
                        "chunk_id": cid, "provider": provider, "model": model_name, "embedding": vec,
                    })
            conn.commit()  # per-batch: progres tersimpan permanen (resumable kalau dihentikan)
            n_upsert += len(page)
            print(f"  [{provider}] {n_upsert}/{len(rows)} ter-upsert")
            if i + batch_size < len(rows) and sleep_s > 0:
                time.sleep(sleep_s)

        if reindex and n_upsert > 0:
            print("[reembed] REINDEX chunk_embeddings_ab_hnsw ...")
            with conn.cursor() as cur:
                cur.execute("REINDEX INDEX CONCURRENTLY chunk_embeddings_ab_hnsw")

        print(f"[reembed] OK: {n_upsert} vektor {provider} ter-upsert ke chunk_embeddings_ab.")
        return {"provider": provider, "model": model_name, "n_kandidat": len(rows), "n_upsert": n_upsert}
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Re-embed chunk yang sudah ter-ingest pakai provider A/B (Jina), "
                     "disimpan terpisah di chunk_embeddings_ab tanpa menimpa baseline bge-m3."
    )
    ap.add_argument("--provider", required=True, choices=_VALID_PROVIDERS)
    ap.add_argument("--dokumen-id", action="append", dest="dokumen_ids",
                     help="Filter dokumen_id (ulangi flag utk beberapa dokumen). Kosong = SEMUA dokumen.")
    ap.add_argument("--limit", type=int, default=None, help="Batasi jumlah chunk (subset dummy/gold-set).")
    ap.add_argument("--batch-size", type=int, default=32, help="Chunk per panggilan API (hormati rate limit).")
    ap.add_argument("--sleep-s", type=float, default=0.5, help="Jeda antar-batch (detik).")
    ap.add_argument("--force", action="store_true", help="Re-embed ulang meski sudah ada (default: skip).")
    ap.add_argument("--dry-run", action="store_true", help="Cuma tampilkan rencana, tanpa panggil API/tulis DB.")
    ap.add_argument("--reindex", action="store_true", help="REINDEX CONCURRENTLY HNSW di akhir.")
    args = ap.parse_args(argv)

    if args.dokumen_ids is None and args.limit is None and not args.dry_run:
        raise SystemExit(
            "Menjalankan tanpa --dokumen-id ATAU --limit berarti SELURUH korpus ke API eksternal "
            "berbayar. Jalankan dulu dgn subset (--limit N atau --dokumen-id X) sesuai gerbang "
            "konfirmasi Tahap 3. Kalau memang sudah disetujui menjalankan penuh, tambahkan "
            "--limit dgn angka besar (mis. --limit 100000) sebagai konfirmasi eksplisit di command line."
        )

    run(args.provider, dokumen_ids=args.dokumen_ids, limit=args.limit, batch_size=args.batch_size,
        sleep_s=args.sleep_s, force=args.force, dry_run=args.dry_run, reindex=args.reindex)


if __name__ == "__main__":
    main()
