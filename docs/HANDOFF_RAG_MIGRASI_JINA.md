# Handoff — Migrasi Embedding/Reranker ke Jina (Tahap 0–7)

Ringkasan alih-sesi utk migrasi provider embedding/rerank `bge-m3` lokal → API serverless Jina.
**Ini bukan pengganti `docs/STATUS_RAG.md`** — untuk detail penuh (alasan, angka lengkap, desain
skema), rujuk ke sana. Dokumen ini cukup dibaca sekali di awal sesi baru lalu dibuang dari konteks.

## 1. Status akhir

- **Provider default produksi: `EMBEDDING_PROVIDER=jina` / `RERANK_PROVIDER=jina`** (`.env.example`).
- `local` (bge-m3 + bge-reranker-v2-m3) **tetap ada penuh** sebagai fallback/rollback — bukan
  dihapus. Ganti provider = ganti env, nol perubahan kode.
- Commit terakhir migrasi ini: **`d580e83`** (`feat(retrieval): migrasi embedding+reranker ke
  provider serverless Jina, hidup berdampingan dgn bge-m3 lokal`) — sudah di-commit, **belum di-push**
  (butuh konfirmasi eksplisit terpisah sebelum push).

## 2. File kunci

| File | Fungsi |
|---|---|
| `app/retrieval/embeddings.py` | Dispatch embedding per `EMBEDDING_PROVIDER` (local/jina); satu sumber kebenaran embed query & chunk. |
| `app/retrieval/rerank.py` | Dispatch rerank per `RERANK_PROVIDER` (local/jina); kontrak `[(index, skor)]` sama di kedua provider. |
| `app/retrieval/_provider_http.py` | Transport HTTP bersama (Jina): retry+timeout+circuit-breaker+**semaphore `PROVIDER_MAX_CONCURRENT`** (default 2, dibagi lintas embed+rerank). |
| `app/sanitize.py` | Allowlist/scrub PII eksplisit sebelum teks masuk query retrieval (`retriever.search`/`get_by_reference`) atau prompt LLM (`build_user_prompt`). |
| `app/retrieval/cache.py` | Cache in-memory hasil retrieval+rerank final (kunci: kategori, zona, dokumen, +provider aktif). |
| `scripts/reembed.py` | Re-embed idempoten/resumable chunk yang sudah ter-ingest ke provider kandidat A/B (`chunk_embeddings_ab`), tanpa menyentuh baseline `chunks.embedding`. |
| `scripts/check_provider_health.py` | Tool operasional permanen: sanity call + stress-test concurrency provider aktif — jalankan manual kapan pun perlu re-verifikasi. |

## 3. Cakupan data — PENTING, jangan berasumsi semua wilayah tercakup

**`chunk_embeddings_ab` (vektor Jina) HANYA berisi Sleman Tengah** (440/440 chunk, parity penuh
dengan baseline bge-m3 di `chunks.embedding`). **Timur dan Barat BELUM di-re-embed ke Jina** — kalau
retrieval diminta utk wilayah itu dengan `EMBEDDING_PROVIDER=jina`, `dense_search_ab` akan kosong
(fallback ke lexical/FTS saja, HNSW dense-nya kosong utk wilayah itu). Perlu `scripts/reembed.py
--provider jina --dokumen-id rdtr-sleman-timur` (dan barat) sebelum wilayah itu dipakai produksi
dgn Jina.

## 4. Yang sudah diverifikasi

- Retrieval gate eval (22 query, Tengah): Hit-Rate@5 **100% kedua stack**; MRR Jina **0.875** vs
  bge-m3 **0.936** (ambang ≥0.60 — lulus, sedikit di bawah tapi tidak gagal).
- Live smoke test (`scripts/test_integrasi.py`, Groq+Jina nyata): faithfulness/grounding konsisten,
  0 `low_confidence` di kedua stack.
- Isolasi kegagalan PII: `PIIDetectedError` di satu poin (disimulasikan via `sanitize_fakta`/
  `ambil_chunks_pendukung`) **tidak menjatuhkan permohonan** — fail-safe per-poin (`template_low_confidence`),
  `low_confidence_keseluruhan`+`catatan_global` benar, endpoint tetap 200 (`tests/test_pii_failure_isolation.py`).
- Semaphore concurrency: peak in-flight terukur **persis** `PROVIDER_MAX_CONCURRENT` di bawah beban
  6 panggilan serentak (`tests/test_provider_http.py`), dan live: 1 permohonan penuh + stress-test
  3x paralel Jina murni → 0 error 429.

## 5. Keterbatasan diketahui (belum blocker, wajib diingat)

- `PROVIDER_MAX_CONCURRENT` **in-memory per-proses** — deployment multi-worker nanti membuat batas
  efektif = `PROVIDER_MAX_CONCURRENT × jumlah_proses`, bukan global.
- Gold set 6 kasus (`eval/gold_set.jsonl`) hardcode `MockRetriever` di `eval/run_eval.py` — **tidak
  bisa** dipakai membandingkan provider retrieval sampai `citation_ids_diharapkan` di-relabel ke ID
  chunk nyata. Backlog terpisah, lihat `STATUS_RAG.md` § "Ditunda".
- **Prioritas berikutnya (bukan backlog biasa):** latensi permohonan masih ~53 detik, disebabkan
  retry rate-limit di sisi **Groq (LLM)**, bukan Jina — di luar cakupan migrasi ini, belum diselidiki.
- Nama tabel `chunk_embeddings_ab` menyesatkan sekarang (sudah jadi tabel produksi aktif dipakai
  `dense_search_ab`, bukan cuma eksperimen A/B sementara) — rename opsional, tidak urgent.

## 6. Rujukan lengkap

`docs/STATUS_RAG.md` §§ **"Keputusan provider embedding/rerank: Jina (Agustus 2026)"** dan
**"Pembatas concurrency panggilan Jina (`PROVIDER_MAX_CONCURRENT`)"** — alasan lengkap, tabel angka,
detail desain skema `chunk_embeddings_ab`, dan catatan batasan per-proses.

---
*Catatan penulis dokumen ini: `context-transfer.md` yang disebut di prompt sebagai "sudah ada di
repo" tidak saya temukan di working tree saat ini (`find . -iname context-transfer.md` kosong) —
kemungkinan belum pernah dibuat, atau ada di tempat lain di luar repo ini. Tidak saya buat/timpa
sesuai instruksi ("jangan ubah context-transfer.md yang sudah ada"); flagging ini supaya tidak jadi
asumsi keliru di sesi berikutnya.*
