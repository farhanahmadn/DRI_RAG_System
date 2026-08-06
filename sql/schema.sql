-- =====================================================================
-- RDTR Sleman — RAG/Retrieval schema (wilayah tim RAG)
-- Postgres 16 + pgvector. DB dipakai bersama reasoning-side (tabel logs
-- milik reasoning terpisah — tidak ada bentrok nama dengan tabel di sini).
--
-- Prinsip yang tercermin di skema:
--  * Chunk (base.py) = sumber sitasi; `id` STABIL (dipakai jadi citation_id),
--    `halaman` akurat. Kolom mengikuti field Chunk 1:1 + kolom retrieval.
--  * Filter versi pakai DUA tanggal: tanggal_berlaku & tanggal_dicabut.
--  * zona = string tunggal (NULL = ketentuan umum, lolos semua filter zona).
--  * Matriks ITBX & intensitas KDB/KLB/KDH = tabel relasional OTORITATIF
--    (dipakai rule engine); chunk level='tabel' di-GENERATE dari baris ini.
--  * Hybrid = dense (pgvector HNSW cosine) + lexical (Postgres FTS/GIN) -> RRF.
--  * Embedding provider-agnostic (local bge-m3 / Jina, lihat app/retrieval/embeddings.py):
--    `chunks.embedding` = vektor BASELINE produksi, ditandai `chunks.embedding_provider`/
--    `embedding_model`; kandidat A/B (jina) hidup berdampingan di `chunk_embeddings_ab`
--    tanpa menimpa baseline, sampai salah satu dikunci jadi default (keputusan manusia, bukan skema).
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------
-- Registry dokumen: alias -> kanonik untuk get_by_reference (jalur UTAMA)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dokumen (
    dokumen_id      text PRIMARY KEY,           -- slug kanonik, mis. 'uu-41-2009'
    nama            text NOT NULL,              -- nama resmi panjang (= Chunk.dokumen)
    jenis           text NOT NULL,              -- UU/RDTR/Permen/RTRW/Metodologi (bebas string)
    tanggal_berlaku date,
    tanggal_dicabut date,
    aliases         text[] NOT NULL DEFAULT '{}' -- variasi rujukan: 'UU No. 41 Tahun 2009','UU 41/2009',...
);

-- Pencarian alias cepat (get_by_reference mencocokkan string rujukan -> dokumen_id)
CREATE INDEX IF NOT EXISTS dokumen_aliases_gin ON dokumen USING gin (aliases);

-- ---------------------------------------------------------------------
-- Chunks: prosa (pasal/ayat) + chunk tabel turunan. Yang di-embed.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id              text PRIMARY KEY,           -- STABIL & deterministik (citation_id)
    level           text NOT NULL CHECK (level IN ('pasal','ayat','tabel')),
    parent_id       text REFERENCES chunks(id) ON DELETE SET NULL,
    dokumen_id      text NOT NULL REFERENCES dokumen(dokumen_id),
    dokumen         text NOT NULL,              -- denormalized (= Chunk.dokumen; hemat join saat hydrate)
    pasal           text,
    ayat            text,
    halaman         integer,
    teks            text NOT NULL,              -- teks BERSIH untuk kutipan/sitasi
    teks_prefixed   text NOT NULL,              -- teks + prefiks kontekstual (INI yang di-embed)
    skor            double precision NOT NULL DEFAULT 0.0, -- placeholder; skor relevansi diisi saat query
    istilah_kode    text,                       -- mis. 'Lampiran VI'
    zona            text,                       -- NULL = ketentuan umum
    jenis           text,
    tanggal_berlaku date,
    tanggal_dicabut date,
    embedding       vector(1024),              -- vektor dense BASELINE produksi (lihat embedding_provider)
    embedding_provider text NOT NULL DEFAULT 'local', -- 'local'|'jina' — penghasil `embedding` di atas
    embedding_model text,                      -- mis. 'BAAI/bge-m3'|'jina-embeddings-v3'
    sparse          jsonb,                     -- bge-m3 lexical weights (disimpan; jalur query menyusul)
    fts             tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(teks, ''))) STORED,
    source_version  text,                      -- versi parse (data/parsed/vN)
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Migrasi utk DB yang sudah ada sebelum embedding_provider/embedding_model ditambahkan (idempoten;
-- reapply manual: `docker compose exec -T db psql -U rdtr -d rdtr < sql/schema.sql`, lihat sql/apply.md).
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_provider text NOT NULL DEFAULT 'local';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model text;
-- Backfill baris lama (di-embed sebelum kolom ini ada): satu-satunya provider yang pernah dipakai = local/bge-m3.
UPDATE chunks SET embedding_model = 'BAAI/bge-m3'
    WHERE embedding IS NOT NULL AND embedding_model IS NULL AND embedding_provider = 'local';

-- Index retrieval
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_embedding_provider
    ON chunks (embedding_provider);
CREATE INDEX IF NOT EXISTS chunks_fts_gin
    ON chunks USING gin (fts);
-- get_by_reference & get_parent: lookup pasal/ayat presisi
CREATE INDEX IF NOT EXISTS chunks_dok_pasal_ayat
    ON chunks (dokumen_id, pasal, ayat);
CREATE INDEX IF NOT EXISTS chunks_parent
    ON chunks (parent_id);
CREATE INDEX IF NOT EXISTS chunks_zona
    ON chunks (zona);
CREATE INDEX IF NOT EXISTS chunks_tanggal
    ON chunks (tanggal_berlaku, tanggal_dicabut);
CREATE INDEX IF NOT EXISTS chunks_level
    ON chunks (level);
CREATE INDEX IF NOT EXISTS chunks_istilah_kode
    ON chunks (istilah_kode);

-- ---------------------------------------------------------------------
-- Vektor kandidat A/B (Jina) — HIDUP BERDAMPINGAN dgn baseline `chunks.embedding` (local/
-- bge-m3), TIDAK MENIMPANYA. Satu baris per (chunk, provider kandidat). `scripts/reembed.py` yang
-- mengisi (upsert idempoten); dipakai evaluasi perbandingan (Tahap 5), BUKAN jalur produksi
-- RetrieverAsli.search() — provider produksi tetap baca `chunks.embedding` (ditandai
-- `chunks.embedding_provider`) sampai salah satu provider dikunci jadi default & di-promote ke sana.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunk_embeddings_ab (
    chunk_id           text NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    embedding_provider text NOT NULL,           -- 'jina' (kandidat; 'local' sudah ada di chunks)
    embedding_model    text NOT NULL,
    embedding          vector(1024) NOT NULL,
    created_at         timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, embedding_provider)
);
CREATE INDEX IF NOT EXISTS chunk_embeddings_ab_hnsw
    ON chunk_embeddings_ab USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunk_embeddings_ab_provider
    ON chunk_embeddings_ab (embedding_provider);

-- ---------------------------------------------------------------------
-- Tabel relasional OTORITATIF (rule engine). BUKAN di-embed.
-- Matriks ITBX ketentuan kegiatan per zona (mis. Lampiran V).
-- ---------------------------------------------------------------------
-- PK menyertakan dokumen_id: tiap RDTR wilayah (Barat/Tengah/Timur) punya matriksnya sendiri,
-- kode zona bisa sama antar dokumen. Divalidasi thd Tengah: 358 kegiatan x 32 zona = 11.456 baris.
CREATE TABLE IF NOT EXISTS matriks_kegiatan (
    dokumen_id      text NOT NULL REFERENCES dokumen(dokumen_id),
    zona            text NOT NULL,
    kode_kegiatan   text NOT NULL,              -- nomor kegiatan KBLI-based (mis. '001')
    nama_kegiatan   text,
    kategori        text,                       -- huruf golongan (A/B/C/...) di matriks
    izin            char(1) NOT NULL CHECK (izin IN ('I','T','B','X')),
    keterangan      text,                       -- syarat/batasan untuk T/B (Tabel Penjelasan, menyusul)
    lampiran        text,                       -- mis. 'Lampiran V.A'
    halaman         integer,
    tanggal_berlaku date,
    tanggal_dicabut date,
    PRIMARY KEY (dokumen_id, zona, kode_kegiatan)
);
CREATE INDEX IF NOT EXISTS matriks_kegiatan_zona ON matriks_kegiatan (dokumen_id, zona);
CREATE INDEX IF NOT EXISTS matriks_kegiatan_kode ON matriks_kegiatan (kode_kegiatan);

-- Intensitas pemanfaatan ruang: KDB/KLB/KDH per zona.
-- REVISI dari bake-off: nilai bergantung zona x kawasan resapan (bool) x hierarki jalan
-- (arteri/kolektor/lokal/lingkungan), bukan sekadar per-zona. Tengah: 31 zona x 2 x 4 = 248 baris.
CREATE TABLE IF NOT EXISTS intensitas_zona (
    dokumen_id      text NOT NULL REFERENCES dokumen(dokumen_id),
    zona            text NOT NULL,
    nama_zona       text,
    kawasan_resapan boolean NOT NULL,           -- true = Kawasan Resapan Air
    hierarki_jalan  text NOT NULL CHECK (hierarki_jalan IN ('arteri','kolektor','lokal','lingkungan')),
    kdb_maks        numeric,                    -- persen (mis. 70 = 70%); NULL bila '-'
    klb_maks        numeric,
    kdh_min         numeric,                    -- persen
    lampiran        text,                       -- mis. 'Lampiran VI'
    halaman         integer,
    tanggal_berlaku date,
    tanggal_dicabut date,
    PRIMARY KEY (dokumen_id, zona, kawasan_resapan, hierarki_jalan)
);
CREATE INDEX IF NOT EXISTS intensitas_zona_zona ON intensitas_zona (dokumen_id, zona);
