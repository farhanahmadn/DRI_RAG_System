# CLAUDE.md — Sistem AI Reasoning Pre-Check Izin Bangunan Sleman

> File ini otomatis dibaca Claude Code setiap sesi. Ia memberi konteks arsitektur, batas tanggung jawab, dan aturan main. **Jaga tetap ringkas & akurat.**

## Apa yang kita bangun
Komponen **AI Reasoning** untuk sistem pre-check risiko izin bangunan (B2G, Pemda Sleman). Sistem menerima **skor risiko rule-based** + jejak aturannya dari back-end, lalu menghasilkan untuk tiap sub-skor: **reasoning** (penjelasan), **sitasi terverifikasi** dari dokumen regulasi (RAG), dan **rekomendasi** — diakhiri **kesimpulan utama**. Output berupa **JSON** yang di-render tim web.

## Batas tanggung jawab (PENTING)
- **BUKAN tugas kita:** scoring rule-based, GIS/PostGIS (koordinat→fakta spasial), web/UI/PDF, alur persetujuan admin. Semua ada di tim lain.
- **Tugas kita:** RAG + reasoning + sitasi + rekomendasi + guardrail + API. Kita **mengonsumsi** jejak aturan, **menghasilkan** payload JSON.

## Pembagian kerja (2 dev)
- **Teman → seluruh RAG:** app/ingest/ (parse, chunk, embed, isi DB) + app/retrieval/retriever.py
  (implementasi asli: embed query bge-m3 → similarity pgvector → rerank). Retrieval MILIK teman.
- **Saya → jejak aturan JSON hingga output:** terima jejak aturan (input) → susun query → panggil
  retrieve() → kalkulator deterministik → prompt engineering → LLM (Groq) → guardrail → rakit output JSON → API.
  Modul saya: app/schemas.py (jejak & output), app/reasoning/* (llm_client, calculator, templates,
  generator, guardrail, assemble), app/api/*.

### SEAM (kontrak bersama — sepakati dulu, jarang diubah)
- `app/retrieval/base.py`: Protocol `Retriever.search(query: str, filters: dict) -> list[Chunk]`
  dan skema `Chunk` (dokumen, pasal, halaman, teks, skor). Saya & teman sama-sama patuh file ini.
- Saya pakai `app/retrieval/mock.py` (MockRetriever) sampai RAG teman siap. Integrasi = tukar mock→asli,
  kode reasoning TIDAK berubah.
- DB bersama (satu Postgres+pgvector): teman punya tabel vektor, saya punya tabel logs.

### Aturan main
- Saya TIDAK menyentuh app/ingest/ & app/retrieval/retriever.py. Teman TIDAK menyentuh app/reasoning/ & app/api/.
- Kerja di branch masing-masing, PR saat gabung.

## Prinsip arsitektur (jangan dilanggar)
1. **Faithful** — reasoning MENJELASKAN skor, tidak pernah mengontradiksi/menghitung ulang. Skor = ground truth.
2. **Grounded** — tiap klaim bersandar pada jejak + pasal. Sitasi harus nyata; jangan pernah mengarang nomor pasal.
3. **Deterministik di tempat presisi** — semua ANGKA (skor, target rekomendasi) berasal dari kode/back-end, **BUKAN dari LLM**. LLM hanya membungkus jadi kalimat.
4. **Neuro-simbolik** — kode untuk angka & fakta; LLM untuk bahasa. Karena itu model lokal menengah cukup.

## Kontrak Input (jejak aturan dari back-end)
Per indikator: `poin_id, kategori, bobot, skor, kontribusi, nilai_input, ambang, operator, formula, zona, referensi_hukum[], fakta_spasial{in_lp2b, banjir, resapan, jarak_sungai_m, nama_sungai, arah}`.
- **Target rekomendasi** (mis. `footprint_maks`, `selisih`) idealnya juga dikirim back-end. Jika tidak, hitung dengan satu fungsi deterministik dari nilai yang sudah ada — JANGAN minta LLM menghitung.
- Reasoning HANYA boleh menyebut fakta spasial yang ADA di `fakta_spasial`.

## Kontrak Output (JSON ke web)
`{ ringkasan{skor_total, level, kalimat}, poin[ {poin_id, kategori, status, kontribusi, reasoning_pendek, reasoning_panjang, sitasi[{citation_id, dokumen, pasal, halaman, kutipan, terverifikasi}], rekomendasi{tipe, target, saran, disclaimer}} ], kesimpulan{langkah_berdampak[], catatan_lokasi} }`
Taksonomi rekomendasi: `numerik` (hitung target) · `kegiatan` (I/T/B/X) · `lokasional` (banjir/resapan/sempadan/LP2B → jelaskan, tak bisa ditweak).

## Data
- `Indikator Scoring L2` (rubrik) → definisi indikator (dasar jejak aturan).
- `Relevansi Regulasi` → dasar hukum per indikator (seed sitasi).
- RDTR 390 hlm PDF + UU + Permen → **prosa ke RAG; matriks/tabel (I/T/B/X, KDB/KLB/KDH) ke tabel terstruktur** (bukan embedding).

## Tech stack
- **Bahasa:** Python 3.11+
- **Parsing (offline):** LlamaParse (atau Docling lokal) → simpan markdown di `data/parsed/` (di-version).
- **Chunking:** segmenter regex custom per Pasal/Ayat, pola parent-child (child=ayat embed, parent=pasal simpan) + metadata + prefiks kontekstual. Tabel utuh.
- **Embedding & reranker:** bge-m3 + bge-reranker-v2-m3. **Terpisah dari penyaji LLM** (Groq hanya untuk LLM generatif; embedding tetap self-host/layanan sendiri).
- **DB:** PostgreSQL + pgvector (+ FTS untuk BM25). Retrieval = hybrid (dense+BM25) + rerank + small-to-big + filter tanggal berlaku.
- **LLM serving:** provider-agnostic lewat antarmuka OpenAI-compatible. **Dev awal: Groq API** (cepat, tanpa GPU). **Produksi: self-host** Ollama/vLLM. Berpindah = ganti `base_url`+`api_key`+`model` di `.env`; JANGAN sebar detail provider ke seluruh kode — bungkus di satu modul client tipis.
  - Model dev (katalog Groq, mis. Llama terbaru) — model Indonesia-tuned (SEA-LION/Sahabat-AI) menyusul saat self-host. Kandidat dipilih via eval. Paksa JSON via structured outputs (didukung Groq & Ollama).
- **Residensi data (WAJIB diingat):** Groq = server AS, data keluar. Boleh untuk dev dengan data **uji/dummy**. **Sebelum produksi dengan PII warga**, WAJIB tinjau ulang: self-host, minimalkan PII, atau konfirmasi kebijakan Pemda.
- **API/orkestrasi:** FastAPI + kode Python custom. LlamaIndex OPSIONAL (boleh tulis retrieval sendiri).
- **Guardrail:** Python deterministik. **MVP: cek murah dulu** (konsistensi verdict + numerik). Verifikasi sitasi (entailment via model NLI kecil / panggilan LLM) **DITUNDA** sampai eval menunjukkan perlu.
- **Eval/observability:** Langfuse (self-host) + eval harness (promptfoo/custom).

## Konvensi kode
- Skema data pakai **Pydantic** (validasi jejak aturan & output JSON).
- Fungsi yang menghasilkan angka = **deterministik & ada unit test**. LLM tidak boleh menyentuh angka final.
- Indikator **skor 0 ("Aman") → template**, tanpa memanggil LLM.
- Guardrail gagal → regenerasi terarah (maks 1–2×) → fallback template + tanda `low_confidence`. JANGAN loop tak terbatas.
- **Log setiap input & output** (skor, reasoning, sitasi, rekomendasi) sejak awal — untuk pembelajaran yang ditunda.
- Retrieval diuji lebih dulu (pasal benar terambil?) sebelum menyentuh generation.
- Simpan rahasia (API key LlamaParse, Groq, dll) di `.env`, jangan commit.
- **Panggilan LLM lewat satu modul client OpenAI-compatible** (`app/reasoning/llm_client.py`) — provider (Groq/Ollama/vLLM) hanya config, bukan tersebar di kode.
- Data yang dikirim ke LLM saat dev = **uji/dummy**, bukan PII warga asli (residensi).

## Alur pengerjaan
Fase 0 fondasi → Fase 1 pipeline RAG → Fase 2 reasoning engine (1 indikator dulu: LP2B/KDB) → Fase 3 API + eval + hardening. Pembelajaran ditunda.

## Prinsip
**Faithful dulu, pintar kemudian · deterministik dulu, LLM sebagai pelapis · satu indikator dulu, baru melebar · mulai ramping, tambah pengaman saat eval membuktikan perlu.**
