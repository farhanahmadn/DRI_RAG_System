# Status Handoff — Komponen RAG / Retrieval (RDTR Sleman Pre-Check)

Ringkasan kondisi sisi **RAG/retrieval** (parsing, chunking, embedding, retrieval). Untuk kontrak
SEAM lihat `docs/INTEGRASI_RETRIEVER.md`; untuk sisi reasoning lihat `docs/STATUS.md`.

## Cakupan saat ini

Retrieval **RDTR Kawasan Sleman Tengah** (Perbup 80/2023) — target demo — selesai end-to-end dan
LULUS gerbang uji. Ini dokumen paling lengkap (390 hlm: batang tubuh + lampiran), jadi cakupannya
paling kaya. RDTR **Timur** (Perbup 3/2021) juga sudah ter-ingest (prosa) dan bisa dipakai lewat
filter daerah. Barat = batang tubuh saja (belum di-ingest).

Tiga method Protocol `Retriever` (`app/retrieval/base.py`) terimplementasi di
`app/retrieval/retriever.py::RetrieverAsli`, patuh struktural (`isinstance(rt, Retriever)`=True) →
swap dari `MockRetriever` = satu baris tanpa mengubah kode reasoning.

## Hasil uji (gerbang "selesai") — Tengah

`python -m tests.test_retrieval` (butuh DB + model lokal):

| Metrik | Hasil | Ambang |
|---|---|---|
| Hit-Rate@5 | **100%** (22/22) | ≥ 80% |
| MRR | **0.936** | ≥ 0.60 |
| exact-match `get_by_reference` | **100%** | 100% |

Eval set: `tests/eval_set.jsonl` (22 query kategori indikator → chunk otoritatif Tengah; ground-truth
dari isi dokumen, termasuk tabel-chunk Lampiran V.B/VI). **Dev-seeded — perlu validasi ahli tata ruang.**

## Isi korpus Tengah (yang di-embed & bisa disitasi)

- **Prosa pasal** (Pasal 1–68) → chunk Pasal(induk)/Ayat(anak). Definisi Pasal 1 dipecah per butir
  (KDB/KLB/KDH/GSB dll. jadi chunk sendiri; toleran artefak OCR seperti nomor "11 7.").
- **Lampiran V.B** (Tabel Penjelasan) → 32 tabel-chunk: **syarat T (terbatas) & B (bersyarat) +
  sarana-prasarana minimal per zona**. Sitasi utama saat indikator Kegiatan = T/B.
- **Lampiran VI** (Intensitas) → 31 tabel-chunk: ringkasan **KDB/KLB/KDH per zona**. Sitasi KDB/KLB/KDH.
- **Matriks ITBX (Lampiran V.A)** → tabel RELASIONAL `matriks_kegiatan` (358 kegiatan × 32 zona =
  11.456 baris), BUKAN di-embed. Untuk lookup terstruktur; klasifikasi I/T/B/X citeable via prosa Pasal 43.
- Dikecualikan (diverifikasi tak dibutuhkan indikator): peta (LAMPIRAN I–III), Indikasi Program
  (LAMPIRAN IV), tabel ketinggian KKOP (hlm 373-387), tanda tangan/pengundangan.

## Pipeline & modul

| Tahap | Modul | Catatan |
|---|---|---|
| Parse PDF → markdown | `app/ingest/parse.py` | LlamaParse `agentic` (menang bake-off vs Docling). Page-aware. |
| Pilah matriks → relasional | `app/ingest/split.py` | Matriks ITBX + intensitas → JSONL relasional + registry `dokumen`. |
| Chunk struktural prosa | `app/ingest/chunk.py` | Pasal/Ayat; definisi Pasal 1 inline; berhenti sebelum lampiran/penutup; id stabil unik. |
| Tabel-chunk lampiran | `app/ingest/tabel_chunks.py` | Lampiran V.B (syarat T/B) + VI (ringkasan intensitas) → chunk citeable. |
| Embedding | `app/retrieval/embeddings.py` | **bge-m3 dense via `transformers` LANGSUNG** (CLS+normalize). BUKAN sentence-transformers/FlagEmbedding (crash datasets→pyarrow di Windows/py3.13). |
| Simpan | `app/ingest/ingest.py` | Embed + insert (prosa + tabel-chunk). Re-ingest bersih per dokumen. Idempoten. |
| DB akses | `app/retrieval/db.py` | Dense KNN (HNSW cosine) + FTS + lookup + get_parent. Muat `.env` sendiri. |
| Fusion | `app/retrieval/fusion.py` | Reciprocal Rank Fusion (k=60). |
| Rerank | `app/retrieval/rerank.py` | bge-reranker-v2-m3 via `transformers`. |
| Retriever | `app/retrieval/retriever.py` | 3 method Protocol + filter daerah/versi + query expansion. |

## Cara menjalankan (dari nol, wilayah Tengah)

```powershell
docker compose up -d
python -m app.ingest.parse        --backend llamaparse --input data\raw --out data\parsed\v1 --mode agentic
python -m app.ingest.split        --input data\parsed\v1 --out data\parsed\v1\structured
python -m app.ingest.chunk        --input data\parsed\v1 --out data\parsed\v1\structured --only tengah
python -m app.ingest.tabel_chunks --input data\parsed\v1 --structured data\parsed\v1\structured --only tengah
pip install -e ".[rag]"           # transformers+torch (hindari sentence-transformers)
python -m app.ingest.ingest       --structured data\parsed\v1\structured --version v1
python -m scripts.smoke_retrieval                 # lihat hasil retrieval
python -m tests.test_retrieval                    # gerbang: Hit-Rate/MRR/exact-match
```

`.env`: `DATABASE_URL`, `EMBEDDING_MODEL=BAAI/bge-m3`, `RERANKER_MODEL=BAAI/bge-reranker-v2-m3`,
`LLAMA_CLOUD_API_KEY` (parse). Ganti wilayah demo tanpa edit kode: `$env:DEMO_WILAYAH="Sleman Timur"`.

## Integrasi ke reasoning (SEAM)

Ganti isi `app/api/dependencies.py::get_retriever` (wilayah dev reasoning):

```python
from app.retrieval.retriever import RetrieverAsli
@lru_cache
def get_retriever() -> Retriever:
    return RetrieverAsli(default_wilayah="Sleman Tengah")
```

`default_wilayah` mengisi `RetrievalFilters.dokumen` otomatis → retrieval terbatas ke RDTR Tengah.
Nol perubahan lain di `app/reasoning/*`. Uji: `python -m scripts.test_integrasi`.

## Filter multi-daerah

Semua daerah satu tabel `chunks`, ditandai `dokumen_id`/`dokumen`. Daerah izin (dari lokasi/GIS
backend) mengalir ke `RetrievalFilters.dokumen` (substring, mis. "Sleman Tengah") → retrieval terbatas
ke daerah itu. Tak perlu DB terpisah / re-embed.

## Ditunda (sengaja, dengan alasan)

- **Wilayah Barat** — batang tubuh saja (tanpa lampiran matriks); belum di-ingest. Timur sudah
  ter-ingest (prosa) tapi lampiran matriksnya tak tersedia di sumber.
- **Matriks ITBX sebagai tabel-chunk sitasi** — tetap relasional (terlalu granular utk di-embed);
  klasifikasi I/T/B/X citeable via prosa Pasal 43 + syarat via Lampiran V.B.
- **bge-m3 sparse** — kolom `sparse` disediakan tapi jalur query pakai FTS Postgres.
- **Validasi ahli** utk `tests/eval_set.jsonl`.
- **Tabel ketinggian KKOP & Indikasi Program (Lampiran IV)** — belum dipakai 8 indikator; bisa
  ditambah (relasional/tabel-chunk) bila indikatornya muncul.
