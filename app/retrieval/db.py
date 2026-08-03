"""app/retrieval/db.py — akses Postgres untuk retrieval (dense KNN + FTS + lookup + get_parent).

Membangun objek `Chunk` (base.py) dari baris tabel `chunks`. Filter versi/zona/dokumen/jenis
diterapkan konsisten (semantik sama dgn MockRetriever). Dipakai retriever.py.
"""

from __future__ import annotations

import os
from datetime import date

try:  # muat .env (DATABASE_URL) bila belum diset shell — supaya retriever jalan dari entrypoint mana pun
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from app.retrieval.base import Chunk, RetrievalFilters

_COLS = ("id, level, parent_id, teks, dokumen, pasal, ayat, halaman, skor, "
         "tanggal_berlaku, istilah_kode, zona, jenis, tanggal_dicabut")


def connect(dsn: str | None = None):
    import psycopg
    from pgvector.psycopg import register_vector

    dsn = dsn or os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL belum diset")
    conn = psycopg.connect(dsn)
    register_vector(conn)
    return conn


def _row_to_chunk(r: tuple, skor: float | None = None) -> Chunk:
    return Chunk(
        id=r[0], level=r[1], parent_id=r[2], teks=r[3], dokumen=r[4], pasal=r[5], ayat=r[6],
        halaman=r[7], skor=float(skor) if skor is not None else float(r[8] or 0.0),
        tanggal_berlaku=r[9], istilah_kode=r[10], zona=r[11], jenis=r[12], tanggal_dicabut=r[13],
    )


def _where(filters: RetrievalFilters | None) -> tuple[str, list]:
    """Klausa WHERE bersama. Semantik = MockRetriever: zona/jenis hanya menyaring bila keduanya terisi;
    NULL di chunk (ketentuan umum) selalu lolos. as_of pakai tanggal_berlaku & tanggal_dicabut."""
    if filters is None:
        return "", []
    conds, params = [], []
    if filters.as_of is not None:
        conds.append("(tanggal_berlaku IS NULL OR tanggal_berlaku <= %s)")
        params.append(filters.as_of)
        conds.append("(tanggal_dicabut IS NULL OR tanggal_dicabut > %s)")
        params.append(filters.as_of)
    if filters.zona is not None:
        conds.append("(zona IS NULL OR zona = %s)")
        params.append(filters.zona)
    if filters.dokumen is not None:
        conds.append("dokumen ILIKE %s")
        params.append(f"%{filters.dokumen}%")
    if filters.jenis is not None:
        conds.append("(jenis IS NULL OR jenis = %s)")
        params.append(filters.jenis)
    return (" AND " + " AND ".join(conds)) if conds else "", params


def _vec_literal(v: list[float]) -> str:
    """Format list -> literal vector pgvector '[..]' (di-cast ::vector di SQL; andal lintas versi klien)."""
    return "[" + ",".join(f"{float(x):.8f}" for x in v) + "]"


def dense_search(conn, query_vec: list[float], filters: RetrievalFilters | None, limit: int) -> list[tuple[str, float]]:
    """KNN cosine (pgvector HNSW). Return [(id, cosine_sim)] terurut menurun."""
    w, params = _where(filters)
    lit = _vec_literal(query_vec)
    sql = (f"SELECT id, 1 - (embedding <=> %s::vector) AS sim FROM chunks "
           f"WHERE embedding IS NOT NULL{w} ORDER BY embedding <=> %s::vector LIMIT %s")
    with conn.cursor() as cur:
        cur.execute(sql, [lit, *params, lit, limit])
        return [(r[0], float(r[1])) for r in cur.fetchall()]


def fts_search(conn, query_text: str, filters: RetrievalFilters | None, limit: int) -> list[tuple[str, float]]:
    """Lexical FTS Postgres (BM25-ish, ts_rank_cd). Return [(id, rank)] terurut menurun."""
    w, params = _where(filters)
    sql = (f"SELECT id, ts_rank_cd(fts, q) AS rank FROM chunks, plainto_tsquery('simple', %s) q "
           f"WHERE fts @@ q{w} ORDER BY rank DESC LIMIT %s")
    with conn.cursor() as cur:
        cur.execute(sql, [query_text, *params, limit])
        return [(r[0], float(r[1])) for r in cur.fetchall()]


def hydrate(conn, ids: list[str], skor_by_id: dict[str, float] | None = None) -> list[Chunk]:
    """Ambil Chunk lengkap utk daftar id, urut sesuai `ids`. skor diisi dari skor_by_id bila ada."""
    if not ids:
        return []
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_COLS} FROM chunks WHERE id = ANY(%s)", [ids])
        by_id = {r[0]: r for r in cur.fetchall()}
    out = []
    for i in ids:
        r = by_id.get(i)
        if r is not None:
            out.append(_row_to_chunk(r, (skor_by_id or {}).get(i)))
    return out


def get_parent(conn, chunk_id: str) -> Chunk | None:
    with conn.cursor() as cur:
        cur.execute("SELECT parent_id FROM chunks WHERE id = %s", [chunk_id])
        row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        cur.execute(f"SELECT {_COLS} FROM chunks WHERE id = %s", [row[0]])
        prow = cur.fetchone()
        return _row_to_chunk(prow) if prow else None


def lookup_reference(conn, dokumen_ids: list[str], pasal: str | None, ayat: str | None,
                     istilah_kode: str | None) -> list[Chunk]:
    """Lookup presisi utk get_by_reference: filter dokumen_id (+pasal/+ayat/+lampiran)."""
    conds = ["dokumen_id = ANY(%s)"]
    params: list = [dokumen_ids]
    if istilah_kode is not None:
        conds.append("istilah_kode ILIKE %s")
        params.append(f"%{istilah_kode}%")
    if pasal is not None:
        conds.append("pasal = %s")
        params.append(pasal)
    if ayat is not None:
        conds.append("ayat = %s")
        params.append(ayat)
    sql = f"SELECT {_COLS} FROM chunks WHERE {' AND '.join(conds)} ORDER BY level DESC, ayat"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [_row_to_chunk(r) for r in cur.fetchall()]


def match_dokumen(conn, ref_tokens: set[str]) -> list[str]:
    """Cari dokumen_id yang alias/nama-nya cocok token rujukan (untuk get_by_reference)."""
    with conn.cursor() as cur:
        cur.execute("SELECT dokumen_id, nama, aliases FROM dokumen")
        rows = cur.fetchall()
    hits = []
    for did, nama, aliases in rows:
        hay = (nama + " " + " ".join(aliases or [])).lower()
        haytok = set(_tok(hay))
        if ref_tokens & haytok:
            overlap = len(ref_tokens & haytok)
            hits.append((overlap, did))
    hits.sort(reverse=True)
    # ambil dokumen dgn overlap tertinggi (dan yang seri di atas ambang)
    if not hits:
        return []
    top = hits[0][0]
    return [did for ov, did in hits if ov == top]


_STOP = {"no", "nomor", "tahun", "tentang", "dan", "atau", "yang", "pasal", "ayat", "lampiran",
         "peraturan", "bupati", "rdtr", "kawasan", "tata", "ruang", "detail", "rencana"}


def _tok(s: str) -> list[str]:
    import re
    return [t for t in re.findall(r"\w+", s.lower()) if t not in _STOP and len(t) > 1]
