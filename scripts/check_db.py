"""scripts/check_db.py — health-check isi DB vektor (chunking + vektorisasi).

Cek cepat apakah ingest sudah benar TANPA memuat model (hanya psycopg, ringan):
  - jumlah chunk per dokumen & per level (pasal/ayat/tabel)
  - cakupan embedding (yang di-embed vs induk NULL) + dimensi 1024 + norma L2 (~1.0)
  - FTS (lexical) terisi
  - integritas parent-child (ayat -> pasal induk), id duplikat
  - tabel-chunk lampiran (V.B/VI) + tabel relasional (matriks/intensitas/dokumen)
  - statistik panjang teks (deteksi chunk raksasa) + sampel chunk tiap jenis

Jalankan:  python -m scripts.check_db
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _hr(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "-" * 70)


def main() -> None:
    import psycopg
    from pgvector.psycopg import register_vector

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL belum diset di .env")
    conn = psycopg.connect(dsn)
    register_vector(conn)
    cur = conn.cursor()

    def rows(sql, p=None):
        cur.execute(sql, p or [])
        return cur.fetchall()

    _hr("1. JUMLAH CHUNK per dokumen & level")
    print(f"{'dokumen_id':22s} {'level':6s} {'jumlah':>7s} {'ter-embed':>10s}")
    for did, lvl, n, emb in rows(
        "SELECT dokumen_id, level, count(*), count(embedding) FROM chunks GROUP BY dokumen_id, level ORDER BY dokumen_id, level"):
        print(f"{did:22s} {lvl:6s} {n:7d} {emb:10d}")
    (tot,) = rows("SELECT count(*) FROM chunks")[0]
    (emb_tot,) = rows("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")[0]
    print(f"\nTOTAL chunks={tot}  ter-embed={emb_tot}  (induk-berayat sengaja NULL, dipakai get_parent)")

    _hr("2. VEKTORISASI — dimensi & norma L2")
    dims = rows("SELECT DISTINCT vector_dims(embedding) FROM chunks WHERE embedding IS NOT NULL")
    print("dimensi vektor (harus {1024}):", sorted(d[0] for d in dims))
    # norma L2 via SQL: (v <#> v) = -(v·v) -> |v| = sqrt(-(v <#> v))
    print("norma L2 sampel (harus ~1.0 karena dinormalisasi):")
    for cid, norm in rows(
        "SELECT id, sqrt(-(embedding <#> embedding)) FROM chunks WHERE embedding IS NOT NULL ORDER BY id LIMIT 5"):
        print(f"   {cid:34s} |v|={float(norm):.4f}")
    (nzero,) = rows("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL AND vector_dims(embedding)<>1024")[0]
    print("chunk dgn dimensi != 1024:", nzero, "(harus 0)")

    _hr("3. FTS (lexical) & metadata")
    (fts_n,) = rows("SELECT count(*) FROM chunks WHERE length(fts::text) > 0")[0]
    print(f"chunk dgn FTS terisi: {fts_n}/{tot}")
    (hal_null,) = rows("SELECT count(*) FROM chunks WHERE halaman IS NULL AND level<>'tabel'")[0]
    print(f"chunk prosa tanpa halaman: {hal_null} (idealnya 0)")
    (sp,) = rows("SELECT count(*) FROM chunks WHERE sparse IS NOT NULL")[0]
    print(f"chunk dgn sparse terisi: {sp} (sengaja 0 — jalur lexical pakai FTS)")

    _hr("4. INTEGRITAS chunking")
    (orphan,) = rows(
        "SELECT count(*) FROM chunks c WHERE c.level='ayat' AND c.parent_id IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM chunks p WHERE p.id=c.parent_id)")[0]
    print(f"ayat 'yatim' (parent tak ada): {orphan} (harus 0)")
    dup = rows("SELECT id, count(*) FROM chunks GROUP BY id HAVING count(*)>1")
    print(f"id duplikat: {len(dup)} (harus 0)")
    lo, hi, avg, mx = rows(
        "SELECT min(length(teks)), max(length(teks)), round(avg(length(teks))), "
        "max(length(teks)) FILTER (WHERE embedding IS NOT NULL) FROM chunks")[0]
    print(f"panjang teks: min={lo} max={hi} avg={avg} | max(yang di-embed)={mx}")

    _hr("5. TABEL-CHUNK lampiran & tabel relasional")
    for ik, n in rows("SELECT istilah_kode, count(*) FROM chunks WHERE level='tabel' GROUP BY istilah_kode ORDER BY istilah_kode"):
        print(f"   {ik or '(none)':16s}: {n} chunk")
    for t in ("matriks_kegiatan", "intensitas_zona", "dokumen"):
        (c,) = rows(f"SELECT count(*) FROM {t}")[0]
        print(f"   tabel relasional {t:18s}: {c} baris")

    _hr("6. SAMPEL chunk tiap jenis (cek isi & metadata)")
    samples = {
        "PASAL (induk)": "SELECT id,pasal,ayat,halaman,dokumen_id,left(teks,80) FROM chunks WHERE level='pasal' ORDER BY id LIMIT 1",
        "AYAT (anak)": "SELECT id,pasal,ayat,halaman,dokumen_id,left(teks,80) FROM chunks WHERE level='ayat' ORDER BY id LIMIT 1",
        "DEFINISI KDB": "SELECT id,pasal,ayat,halaman,dokumen_id,left(teks,80) FROM chunks WHERE teks ILIKE '%%disingkat KDB adalah%%' AND level='ayat' ORDER BY length(teks) LIMIT 1",
        "TABEL V.B": "SELECT id,zona,istilah_kode,halaman,dokumen_id,left(teks,80) FROM chunks WHERE istilah_kode='Lampiran V.B' LIMIT 1",
        "TABEL VI": "SELECT id,zona,istilah_kode,halaman,dokumen_id,left(teks,80) FROM chunks WHERE istilah_kode='Lampiran VI' LIMIT 1",
    }
    for label, sql in samples.items():
        r = rows(sql)
        print(f"\n[{label}]")
        if r:
            print("  ", r[0])
        else:
            print("   (tak ada)")

    conn.close()
    print("\n" + "=" * 70 + "\nSelesai. Cek: dimensi=1024, norma~1.0, ayat_orphan=0, id_duplikat=0,\n"
          "FTS terisi, tabel-chunk V.B/VI ada, dan sampel teks masuk akal.")


if __name__ == "__main__":
    main()
