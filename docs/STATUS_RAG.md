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

## Keputusan provider embedding/rerank: Jina (Agustus 2026)

**Keputusan: `EMBEDDING_PROVIDER=jina` + `RERANK_PROVIDER=jina` jadi default di `.env.example`**
(`jina-embeddings-v3` + `jina-reranker-v3`), menggantikan `local` (bge-m3 + bge-reranker-v2-m3).
Tetap bisa di-override via `.env` (baris `EMBEDDING_PROVIDER`/`RERANK_PROVIDER`) — TIDAK ada
perubahan kode utk berpindah provider, `local` dipertahankan penuh sebagai jalur rollback/self-host.

**Alasan** (VPS produksi tidak cukup memori utk `transformers`+`torch` lokal — motivasi migrasi
dari awal — lihat prinsip provider-agnostic di `app/retrieval/embeddings.py`/`rerank.py`):

1. **Retrieval gate eval (22 query, Sleman Tengah, `tests/test_retrieval.py`)** — dijalankan utk
   kedua stack atas korpus Tengah yang SAMA (440 chunk, full parity):

   | Metrik | bge-m3 (local) | Jina | Ambang gerbang |
   |---|---|---|---|
   | Hit-Rate@5 | 100.00% (22/22) | **100.00%** (22/22) | ≥ 80% |
   | MRR | 0.936 | **0.875** | ≥ 0.60 |
   | exact-match `get_by_reference` | 100% | **100%** | 100% |

   Keduanya lulus gerbang. Hit-Rate identik; MRR Jina sedikit di bawah bge-m3 tapi jauh di atas
   ambang — pada 4/22 query (`intensitas pemanfaatan ruang`, `Kegiatan`, `Banjir`, `rawan bencana
   banjir lahar`) chunk yang benar turun dari rank 1 ke rank 2, TETAP selalu masuk top-5.

2. **Smoke test nyata** (`scripts/test_integrasi.py`, fixture `l2_sample_amplop_6191.json`,
   `DEMO_WILAYAH="Sleman Tengah"`, panggilan Groq sungguhan) — 3 poin non-aman (itbx/intensitas/
   dampak) dibandingkan local vs Jina: **faithfulness & grounding konsisten** — tidak ada
   `low_confidence` di kedua stack, semua sitasi `terverifikasi=True` di kedua stack, status/arah
   rekomendasi selalu sama. Satu-satunya beda: poin `intensitas` mengutip chunk pendukung berbeda
   (local → Pasal 61 Ayat 4; Jina → Lampiran VI Zona KT) — KEDUANYA tetap relevan & terverifikasi,
   cuma pilihan pasal beda, bukan kegagalan. Poin `itbx` & `dampak`: sitasi identik persis di kedua
   stack. Pola ini selaras dgn temuan MRR di atas (retrieval indikator "intensitas"/"dampak" adalah
   satu-satunya area Jina sedikit kurang tajam dibanding bge-m3, tapi tidak sampai salah/gagal).

3. **Alasan operasional**: menghapus kebutuhan `transformers`+`torch` (~2-3GB) di VPS produksi
   (extra `rag-serverless` di `pyproject.toml`), tanpa DPA administratif terpisah dgn Jina — mitigasi
   privasi jadi TEKNIS (lihat `app/sanitize.py`, Tahap 6: data sensitif tidak pernah terbentuk jadi
   teks yang dikirim ke provider eksternal), bukan kontraktual.

**Kalau perlu rollback**: set `EMBEDDING_PROVIDER=local`/`RERANK_PROVIDER=local` di `.env` — tidak
ada migrasi/perubahan kode, baseline bge-m3 tetap utuh di `chunks.embedding` (tak pernah ditimpa
selama proses A/B, lihat § Skema `chunk_embeddings_ab` kalau ditambahkan).

**Belum diuji/diputuskan** (di luar 2 stack di atas): perbandingan lewat `eval/gold_set.jsonl` (6
kasus) — TIDAK BISA dijalankan lewat `RetrieverAsli` sama sekali, gap desain terpisah (lihat §
"Ditunda" di bawah), jadi keputusan ini murni berdasar retrieval gate eval + smoke test, bukan gold
set reasoning formal.

### Pembatas concurrency panggilan Jina (`PROVIDER_MAX_CONCURRENT`)

**Default: `PROVIDER_MAX_CONCURRENT=2`** (`.env.example`/`.env`) — semaphore di
`app/retrieval/_provider_http.py`, membatasi berapa banyak panggilan HTTP ke Jina (embed+rerank,
satu semaphore dibagi lintas keduanya) yang boleh IN-FLIGHT bersamaan. Panggilan ke-N+1 **antre**,
bukan gagal. Sengaja **terpisah total** dari `REASONING_MAX_WORKERS` (itu paralelisme thread
reasoning umum lintas SEMUA permohonan yang bersamaan lewat satu `ThreadPoolExecutor` global,
bukan cuma panggilan Jina).

Ditambahkan proaktif setelah observasi live (Agustus 2026): 1 permohonan penuh via `/reasoning`
(3 poin, DB+Groq+Jina sungguhan) DAN stress-test 3x paralel murni Jina (3 kategori sekaligus per
putaran) **sama sekali tidak memicu 429** — tapi cakupan uji itu terbatas ke maks 3 panggilan Jina
bersamaan dari SATU permohonan; beban produksi sungguhan (beberapa permohonan datang nyaris
bersamaan, semua berbagi `REASONING_MAX_WORKERS` yang sama) belum teruji. Semaphore ini jaring
pengaman murah dipasang sebelum sinyal 429 nyata muncul, bukan reaksi ke bug yang sudah terjadi.

**Batasan yang perlu diketahui — semaphore ini PER-PROSES, bukan terdistribusi.** Kalau nanti
deployment produksi menjalankan lebih dari satu worker/proses Python (mis. beberapa `uvicorn`
worker, atau beberapa instance di belakang load balancer), **setiap proses punya semaphore sendiri**
(state in-memory Python biasa, bukan Redis/DB) — batas efektif SEBENARNYA jadi
`PROVIDER_MAX_CONCURRENT × jumlah_proses`, bukan `PROVIDER_MAX_CONCURRENT` secara global. Kalau
concurrency lintas-proses jadi masalah nyata nanti (429 tetap muncul walau tiap proses sudah dibatasi
2), opsinya: (a) turunkan `PROVIDER_MAX_CONCURRENT` per proses sebanding jumlah worker, atau
(b) ganti semaphore in-memory dengan yang terdistribusi (mis. Redis semaphore) — belum diperlukan
sekarang (deployment saat ini 1 proses), dicatat di sini supaya tidak jadi kejutan nanti.

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
- **Gap ketahuan (migrasi embedding/rerank serverless, Agu 2026)**: `eval/gold_set.jsonl` (6 kasus)
  hardcode `MockRetriever()` di `eval/run_eval.py::main()`, dan `citation_ids_diharapkan` di gold set
  itu (mis. `rdtr-p1-a107`, `rdtr-lampiran-vi-c1`) cocok PERSIS dengan fixture ID di
  `app/retrieval/mock.py`, bukan skema ID chunk nyata (`rdtr-sleman-tengah-p1-a117` dst di DB). Gold
  set ini **tidak bisa dijalankan lewat `RetrieverAsli`** (local maupun provider lain) — `sitasi_grounded`
  akan gagal utk KEDUA provider karena mismatch ID, bukan sinyal kualitas retrieval. Jadi gold set 6
  kasus TIDAK BISA dipakai membandingkan provider embedding/rerank sampai `citation_ids_diharapkan`
  di-relabel ke ID chunk nyata (atau `jalankan_gold_set()` diberi param retriever eksplisit + gold set
  baru yang selaras korpus asli). Bukan blocker — regresi jangka panjang, layak dibahas kalau gold set
  mau dipakai lagi utk perbandingan retrieval, bukan cuma reasoning/LLM murni.
