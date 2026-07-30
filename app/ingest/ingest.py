"""app/ingest/ingest.py — embed prosa + isi Postgres (Langkah 7).

Muat registry `dokumen` + `chunks_prosa.jsonl` (dari chunk.py) ke Postgres. Chunk `to_embed=True`
di-embed bge-m3 (dense + sparse, via app/retrieval/embeddings.py — SATU sumber, sama dgn query);
chunk induk-berayat disimpan tanpa embedding (untuk get_parent). Idempoten (ON CONFLICT DO UPDATE),
jadi aman dijalankan ulang.

Tabel relasional (matriks_kegiatan/intensitas_zona) dimuat terpisah oleh loader relasional
(opsional; untuk demo Timur prosa-only tidak diperlukan). Filter daerah otomatis siap: tiap chunk
membawa dokumen_id/dokumen -> RetrievalFilters.dokumen (substring) membatasi ke satu wilayah.

Prasyarat: pip install -e ".[rag]"  ; DATABASE_URL di .env  ; DB sudah `docker compose up -d`.

CLI:
  python -m app.ingest.ingest --structured data/parsed/v1/structured --version v1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _connect():
    import psycopg
    from pgvector.psycopg import register_vector

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL belum diset di .env")
    conn = psycopg.connect(dsn)
    register_vector(conn)
    return conn


def _upsert_dokumen(conn, reg: list[dict]) -> None:
    with conn.cursor() as cur:
        for d in reg:
            cur.execute(
                """
                INSERT INTO dokumen(dokumen_id, nama, jenis, tanggal_berlaku, tanggal_dicabut, aliases)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (dokumen_id) DO UPDATE SET
                    nama=EXCLUDED.nama, jenis=EXCLUDED.jenis,
                    tanggal_berlaku=EXCLUDED.tanggal_berlaku, tanggal_dicabut=EXCLUDED.tanggal_dicabut,
                    aliases=EXCLUDED.aliases
                """,
                (d["dokumen_id"], d["nama"], d["jenis"], d.get("tanggal_berlaku"),
                 d.get("tanggal_dicabut"), d.get("aliases", [])),
            )
    conn.commit()


_INSERT_CHUNK = """
INSERT INTO chunks(
    id, level, parent_id, dokumen_id, dokumen, pasal, ayat, halaman,
    teks, teks_prefixed, istilah_kode, zona, jenis, tanggal_berlaku, tanggal_dicabut,
    embedding, sparse, source_version)
VALUES (%(id)s,%(level)s,%(parent_id)s,%(dokumen_id)s,%(dokumen)s,%(pasal)s,%(ayat)s,%(halaman)s,
    %(teks)s,%(teks_prefixed)s,%(istilah_kode)s,%(zona)s,%(jenis)s,%(tanggal_berlaku)s,%(tanggal_dicabut)s,
    %(embedding)s,%(sparse)s,%(source_version)s)
ON CONFLICT (id) DO UPDATE SET
    level=EXCLUDED.level, parent_id=EXCLUDED.parent_id, dokumen=EXCLUDED.dokumen,
    pasal=EXCLUDED.pasal, ayat=EXCLUDED.ayat, halaman=EXCLUDED.halaman,
    teks=EXCLUDED.teks, teks_prefixed=EXCLUDED.teks_prefixed, istilah_kode=EXCLUDED.istilah_kode,
    zona=EXCLUDED.zona, jenis=EXCLUDED.jenis, tanggal_berlaku=EXCLUDED.tanggal_berlaku,
    tanggal_dicabut=EXCLUDED.tanggal_dicabut, embedding=EXCLUDED.embedding, sparse=EXCLUDED.sparse,
    source_version=EXCLUDED.source_version
"""


def _load_chunks(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def ingest_chunks(conn, chunks: list[dict], version: str, batch_size: int = 32,
                  encode_dense=None) -> tuple[int, int]:
    """Embed to_embed=True lalu insert semua chunk. encode_dense injectable (untuk test)."""
    if encode_dense is None:
        from app.retrieval.embeddings import encode_dense  # impor lazy (butuh torch)

    # 0) re-ingest bersih: hapus dulu chunk dokumen yang ada di batch ini (hindari sisa id lama
    #    saat hasil chunking berubah). Dokumen lain (mis. wilayah lain) tidak tersentuh.
    dok_ids = sorted({c["dokumen_id"] for c in chunks})
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE dokumen_id = ANY(%s)", [dok_ids])
    conn.commit()

    # 1) embed hanya yang perlu, secara batch, atas teks_prefixed (dense 1024-dim)
    to_embed = [c for c in chunks if c.get("to_embed")]
    emb_by_id: dict[str, list] = {}
    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i:i + batch_size]
        vecs = encode_dense([c["teks_prefixed"] for c in batch])
        for c, v in zip(batch, vecs):
            emb_by_id[c["id"]] = v
        print(f"  embed {min(i + batch_size, len(to_embed))}/{len(to_embed)}")

    # 2) insert semua chunk (embedding NULL untuk induk-berayat)
    n = 0
    with conn.cursor() as cur:
        for c in chunks:
            dense = emb_by_id.get(c["id"])
            row = {
                "id": c["id"], "level": c["level"], "parent_id": c.get("parent_id"),
                "dokumen_id": c["dokumen_id"], "dokumen": c["dokumen"],
                "pasal": c.get("pasal"), "ayat": c.get("ayat"), "halaman": c.get("halaman"),
                "teks": c["teks"], "teks_prefixed": c["teks_prefixed"],
                "istilah_kode": c.get("istilah_kode"), "zona": c.get("zona"), "jenis": c.get("jenis"),
                "tanggal_berlaku": c.get("tanggal_berlaku"), "tanggal_dicabut": c.get("tanggal_dicabut"),
                "embedding": dense if dense else None,
                "sparse": None,  # jalur lexical pakai FTS Postgres; sparse bge-m3 tak dipakai
                "source_version": version,
            }
            cur.execute(_INSERT_CHUNK, row)
            n += 1
    conn.commit()
    return n, len(emb_by_id)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Embed prosa + isi Postgres (dokumen + chunks).")
    ap.add_argument("--structured", required=True, type=Path, help="Folder structured (dokumen.json, chunks_prosa.jsonl)")
    ap.add_argument("--version", default="v1", help="Tag source_version")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    reg = json.loads((args.structured / "dokumen.json").read_text(encoding="utf-8"))
    chunks = _load_chunks(args.structured / "chunks_prosa.jsonl")
    tabel_path = args.structured / "chunks_tabel.jsonl"
    if tabel_path.exists():
        tabel = _load_chunks(tabel_path)
        chunks += tabel
        print(f"[ingest] + chunks_tabel (Lampiran V.B/VI): {len(tabel)}")
    print(f"[ingest] dokumen={len(reg)}  chunks={len(chunks)}  to_embed={sum(1 for c in chunks if c.get('to_embed'))}")

    conn = _connect()
    try:
        _upsert_dokumen(conn, reg)
        n, e = ingest_chunks(conn, chunks, args.version, args.batch_size)
        print(f"[ingest] OK: {n} chunk tersimpan, {e} ter-embed.")
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            print(f"[ingest] chunks dgn embedding di DB: {cur.fetchone()[0]}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
