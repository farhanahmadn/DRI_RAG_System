# CLAUDE.md — Sistem AI Reasoning Pre-Check Izin Bangunan Sleman

> File ini otomatis dibaca Claude Code setiap sesi. Ia memberi konteks arsitektur, batas tanggung jawab, dan aturan main. **Jaga tetap ringkas & akurat.**

## Apa yang kita bangun
Komponen **AI Reasoning** — peran **"L3 Advisory"** — untuk sistem pre-check risiko izin bangunan
(B2G, Pemda Sleman). Sistem menerima dari back-end (L2) hasil **gate_hukum** (2 tahap: ITBX →
Intensitas) dan **impact_assessment** (skor dampak tata guna lahan), lalu menghasilkan untuk tiap
poin: **reasoning** (penjelasan), **sitasi terverifikasi** dari dokumen regulasi (RAG), dan
**rekomendasi** — DAN men-turunkan **rekomendasi_sistem** kami sendiri secara deterministik (bukan
dari LLM) — diakhiri **kesimpulan utama**. Output berupa **JSON** yang di-render tim web.

## Batas tanggung jawab (PENTING)
- **BUKAN tugas kita:** menghitung **gate_hukum**/**impact_assessment** (itu L2/back-end — kita
  KONSUMSI, tidak menghitung ulang), GIS/PostGIS (koordinat→fakta spasial), web/UI/PDF, alur
  persetujuan admin. Semua ada di tim lain.
- **Tugas kita:** RAG + reasoning + sitasi + rekomendasi + **rekomendasi_sistem** (turunan kami
  sendiri, deterministik) + guardrail + API. Kita **mengonsumsi** gate_hukum & impact_assessment,
  **menghasilkan** payload JSON.

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

## Kontrak Input (`L2Assessment`, dari back-end/L2 — lihat `app/schemas.py`)
`{ lokasi{koordinat, geojson, rdtr_zone, luas_lahan_m2}, gate_hukum{final_gate_status, decisive_stage,
tahapan{itbx{status, lolos, kegiatan_diizinkan/terbatas/bersyarat[], keterangan_ketentuan[], reason,
dasar_hukum[]}, intensitas{status, lolos, parameter{kdb/klb/kdh:{usulan, ambang_maks/min, memenuhi}},
reason, dasar_hukum[]}}}, impact_assessment{dinilai, impact_score, impact_category, c_before/c_after,
...}, meta{data_confidence_keseluruhan, caveats[]} }`

**Gate hukum — 2 tahap, short-circuit** (hanya evaluasi tahap 2 kalau tahap 1 lolos):
1. **Tahap ITBX** — klasifikasi kegiatan di zona (I/T/B/TB/X). Hanya kalau hasilnya **'X'** →
   **Tidak Lolos**, tahap Intensitas TIDAK dievaluasi.
2. **Tahap Intensitas** (KDB/KLB/KDH, dst) — dievaluasi kalau ITBX bukan 'X'. Ada pelanggaran
   intensitas → **Lolos Bersyarat** (bukan Tidak Lolos).

**Impact assessment** — skor dampak berdasar kelas tata guna lahan. **Skor INVERS: TINGGI = dampak
RENDAH.** JANGAN dibalik saat menyusun reasoning/rekomendasi — cek dulu arah skornya tiap kali dipakai.

**Field-level structure**: lihat `tests/fixtures/l2_sample_lolos.json`, `l2_sample_amplop_6191.json`
(Lolos Bersyarat, ITBX 'T', dampak Tinggi — juga contoh payload ber-amplop {statusCode, message,
data}), `l2_sample_tidak_lolos.json` (contoh nyata dari back-end) — JANGAN menebak nama field baru
di luar yang ada di fixtures/`app/schemas.py`.

Reasoning HANYA boleh menyebut fakta yang ADA di payload gate_hukum/impact_assessment yang dikirim.

## Kontrak Output (`OutputL3`, JSON ke web — lihat `app/schemas.py`)
Dua-jalur: ringkasan **gate hukum** (kepatuhan) TERPISAH dari ringkasan **dampak** (mitigasi) — makna
beda, JANGAN dicampur jadi satu skor komposit seperti model lama.

`{ ringkasan_gate{final_gate_status, decisive_stage, kalimat}, ringkasan_dampak{impact_category,
impact_score, kalimat}, poin[ {poin_id, kategori, status, reasoning_pendek, reasoning_panjang,
sitasi[{citation_id, dokumen, pasal, halaman, kutipan, terverifikasi}], rekomendasi{tipe, target,
saran, disclaimer}, low_confidence} ], rekomendasi_sistem, kesimpulan{langkah_berdampak[],
catatan_lokasi}, catatan_global[], low_confidence_keseluruhan }`

`poin[]` selalu 3 entri tetap — `itbx`, `intensitas`, `dampak` (satu per tahap/skor yang dikonsumsi),
BUKAN daftar dinamis per-indikator seperti model lama.

Taksonomi rekomendasi (3 tipe):
- `kategorikal` — hasil tahap ITBX (lolos/tidak lolos kegiatan di zona).
- `numerik` — pelanggaran tahap Intensitas (KDB/KLB/KDH dst.), hitung target.
- `numerik-mitigasi` — dampak tata guna lahan (impact_assessment): mitigasi KDB↓ / KDH↑ / sumur
  resapan / kolam retensi.

**Caveat wajib:**
- Fallback ITBX saat kegiatan **tidak ditemukan** di tabel zona → `low_confidence=true` pada poin
  ITBX + caveat masuk `catatan_global` (→ `low_confidence_keseluruhan=true`), JANGAN diasumsikan
  otomatis sebagai 'X'. `ringkasan_gate.kalimat` juga TIDAK BOLEH overclaim kepatuhan/pelanggaran
  saat fallback ini terjadi — verdict (`final_gate_status`) tetap apa adanya, yang disesuaikan
  cuma narasi (lihat `app/reasoning/assemble.py::_rakit_kalimat_gate`).
- Kalau klasifikasi ITBX = 'X', **reason WAJIB membedakan** dua makna: (a) kegiatan **dilarang**
  eksplisit di zona tsb, vs (b) kegiatan **tidak terdaftar/tidak ditemukan** di tabel zona (ambiguitas
  data, bukan larangan tegas). Reasoning tidak boleh menyamakan keduanya.

## Data
- `Indikator Scoring L2` (rubrik) → dasar definisi status per tahap (ITBX/Intensitas) & kategori dampak.
- `Relevansi Regulasi` → dasar hukum per tahap/kategori (seed sitasi).
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
- Poin berstatus **aman** (ITBX 'I', Intensitas memenuhi syarat, dampak Rendah) → template, tanpa memanggil LLM.
- Guardrail gagal → regenerasi terarah (maks 1–2×) → fallback template + tanda `low_confidence`. JANGAN loop tak terbatas.
- **Log setiap input & output** (skor, reasoning, sitasi, rekomendasi) sejak awal — untuk pembelajaran yang ditunda.
- Retrieval diuji lebih dulu (pasal benar terambil?) sebelum menyentuh generation.
- Simpan rahasia (API key LlamaParse, Groq, dll) di `.env`, jangan commit.
- **Panggilan LLM lewat satu modul client OpenAI-compatible** (`app/reasoning/llm_client.py`) — provider (Groq/Ollama/vLLM) hanya config, bukan tersebar di kode.
- Data yang dikirim ke LLM saat dev = **uji/dummy**, bukan PII warga asli (residensi).

## Alur pengerjaan
Fase 0 fondasi → Fase 1 pipeline RAG → Fase 2 reasoning engine (1 poin dulu: Intensitas/KDB) → Fase 3 API + eval + hardening. Pembelajaran ditunda.

## Prinsip
**Faithful dulu, pintar kemudian · deterministik dulu, LLM sebagai pelapis · satu indikator dulu, baru melebar · mulai ramping, tambah pengaman saat eval membuktikan perlu.**
