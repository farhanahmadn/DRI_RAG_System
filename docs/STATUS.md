# Status Handoff — Komponen AI Reasoning (RDTR Sleman Pre-Check)

Ringkasan kondisi kerja per hari ini. Untuk kontrak integrasi, lihat `docs/INTEGRASI_BACKEND.md`
(tim rule engine) dan `docs/INTEGRASI_RETRIEVER.md` (tim RAG).

## Cakupan

**8 indikator, mencakup ketiga taksonomi rekomendasi**, semua end-to-end (jejak aturan → LLM →
guardrail → JSON siap-render), masing-masing sudah dibuktikan lewat minimal 1 panggilan Groq
nyata (bukan cuma mock):

| Indikator | Tipe rekomendasi | Pola khusus |
|---|---|---|
| LP2B | lokasional | boolean membership, jalur bersyarat kepentingan umum |
| KDB | numerik | magnitude maks (le), target dari kalkulator |
| KLB | numerik | klon pola KDB |
| KDH | numerik | magnitude minimum (ge) — arah "kurang", bukan "melebihi" |
| Kegiatan (I/T/B/X) | kegiatan | klasifikasi kategorikal 4-tingkat, saran dari data terstruktur |
| Banjir | lokasional | klasifikasi kategorikal 3-tingkat, sitasi metodologi (bukan pasal) |
| Resapan | lokasional | klon pola LP2B (boolean) |
| Sempadan Sungai | lokasional | magnitude minimum (ge), sitasi Permen PUPR — kasus arah verdict yang jadi perhatian khusus (fact-injected, terverifikasi benar) |

Semua indikator numerik/kegiatan/lokasional berbagi mesin yang sama (`calculator.py`,
`generator.py`, `guardrail.py`, `prompts.py`) — tidak ada cabang kode khusus per-indikator di luar
data mock & data terstruktur (`kegiatan_data.py`).

## Test & Eval

- **161 test pytest total** — 155 murni logic/mock (cepat, gratis, jalan tanpa `GROQ_API_KEY`) +
  6 live (butuh key nyata, di-skip otomatis kalau tidak ada, sudah terbukti lulus berulang kali
  sepanjang pengembangan tiap kali kuota Groq tersedia).
- **Eval harness** (`eval/`): gold set 15 kasus berlabel manual (`eval/gold_set.jsonl`) — semua 8
  indikator, ketiga tipe rekomendasi, kasus batas (tepat di ambang), kasus "Aman", kasus RAG-kosong.
  4 metrik struktural deterministik (`eval/metrics.py`): faithfulness (status + arah verdict),
  sitasi grounded, target numerik = kalkulator, JSON valid. **Terverifikasi 15/15 lulus** saat
  kuota tersedia (`python -m eval.run_eval`).
- **Bakeoff model** (`eval/bakeoff.py`) — **selesai, hasil bersih & dipercaya** (kuota tersedia,
  dikonfirmasi bukan artefak rate limit). Kandidat awal `gemma2-9b-it` ternyata sudah
  **decommissioned** di Groq (400 `model_decommissioned`, dicek via `client.models.list()`) —
  diganti `openai/gpt-oss-20b`. Tabel hasil:

  | Model | Overall | Grounded | Low_confidence | Avg Latensi |
  |---|---|---|---|---|
  | **llama-3.3-70b-versatile** | 100% | 100% | 0/15 | 2.21s |
  | llama-3.1-8b-instant | 33% | 33% | 10/15 | 39.06s |
  | openai/gpt-oss-20b | 100% | 100% | 0/15 | 8.70s |

  `llama-3.1-8b-instant` gagal GENUINE (bukan rate limit) — diagnosis langsung ke JSON mentahnya:
  model ini membeo STRUKTUR skema (menaruh jawaban di dalam `"value"` per-field) alih-alih
  menghasilkan instance JSON yang sesuai, sehingga `generator.py` gagal parse dan guardrail
  fallback ke `low_confidence` secara benar di 10/15 kasus.

  **Keputusan model: `llama-3.3-70b-versatile`** (sudah default `.env.example`/`.env`, tak perlu
  diubah). Alasan: seri dengan `openai/gpt-oss-20b` di keempat metrik struktural (100%), tapi ~4x
  lebih cepat (2.21s vs 8.70s) — penting utk precheck yang harus responsif, apalagi latensi
  bertumpuk saat guardrail retry (maks 2x) atau banyak indikator. Sesuai juga ekspektasi arsitektur
  CLAUDE.md ("model dev: katalog Groq, mis. Llama terbaru"). **Catatan**: kualitas Bahasa Indonesia
  (LLM-as-judge) belum diukur sama sekali (sengaja ditunda, lihat di bawah) — bukan faktor
  pembeda keputusan ini; kalau nanti diaktifkan, keputusan model bisa ditinjau ulang.

## Hardening (Fase 3.3)

- **Paralel** — indikator dalam 1 request diproses konkuren (`ThreadPoolExecutor`), urutan output
  tetap terjaga; terbukti lewat test timing (bukan sekadar klaim).
- **Timeout & retry transport** — panggilan Groq: timeout 30s, retry bawaan SDK 2x (env-configurable).
- **Rate limiting** — `/reasoning` dibatasi 10/menit per-IP; **diverifikasi lewat HTTP nyata**
  (request ke-11 & ke-12 dari 12 beruntun terbukti dapat `429`).
- **Observability (opsional)** — integrasi Langfuse v4 (API SDK diverifikasi nyata via
  `pip install`, bukan ditebak), no-op total tanpa konfigurasi. **Belum diverifikasi terhadap
  server Langfuse sungguhan** — tidak ada instance tersedia di lingkungan dev ini.

## Ketangguhan Guardrail — Bukti Nyata, Bukan Klaim

Sepanjang pengembangan, kuota harian Groq berulang kali habis di tengah pekerjaan (>150 panggilan
LLM gagal terkumulatif akibat `RateLimitError` 429, tersebar di berbagai sesi test/eval/bakeoff).
**Nol crash.** Setiap kegagalan — baik transient maupun kuota habis total — selalu berujung
fallback `template_low_confidence` (status/kontribusi tetap benar dari jejak, `sitasi=[]`,
`low_confidence=True`) via `guardrail.py`, dan lapis pertahanan tambahan di `assemble.py`
memastikan satu indikator gagal tidak pernah menjatuhkan seluruh batch. Ini bukan skenario
hipotetis yang "seharusnya bekerja" — ini pola yang benar-benar terjadi berulang kali dan
tertangani dengan benar setiap kali.

## Item Tersisa / Ditunda (sengaja, dengan alasan)

- **Integrasi RAG asli** — `MockRetriever` masih dipakai (data dummy, `app/retrieval/mock.py`).
  Titik sambung SUDAH disiapkan (`app/api/dependencies.py::get_retriever`) — lihat
  `docs/INTEGRASI_RETRIEVER.md`. Menunggu implementasi dari tim RAG.
- **Integrasi back-end nyata** — jejak aturan yang dipakai sejauh ini adalah data uji/gold-set,
  bukan output rule engine sungguhan. Kontrak sudah didokumentasikan
  (`docs/INTEGRASI_BACKEND.md`), menunggu wiring nyata.
- **Validasi ahli tata ruang** — gold set (`eval/gold_set.jsonl`) diseed developer (saya), BELUM
  divalidasi oleh ahli domain RDTR/tata ruang. CLAUDE.md eksplisit menyebut ini langkah berikutnya.
- **LLM-as-judge utk "Kejelasan Bahasa Indonesia"** — DITUNDA sesuai CLAUDE.md ("mulai ramping,
  tambah pengaman saat eval membuktikan perlu") — metrik struktural (4 metrik saat ini) belum
  terbukti tidak cukup.
- **Verifikasi entailment sitasi/verdict via NLI** — stub sudah ada di `guardrail.py`
  (`verifikasi_entailment_sitasi`), sengaja belum diaktifkan, sama alasan seperti di atas.
- **Observability Langfuse live** — kode selesai & no-op aman, tapi belum pernah terhubung ke
  server Langfuse sungguhan (tidak tersedia di lingkungan dev ini).
- **Tinjauan residensi data sebelum produksi** — Groq (LLM provider dev) server-nya di AS. WAJIB
  ditinjau ulang (self-host/minimalkan PII/konfirmasi kebijakan Pemda) sebelum menyentuh data
  warga asli — CLAUDE.md § Residensi data, belum dilakukan karena masih tahap dev dengan data uji.
- **Retriever asli multi-worker** — rate limiter endpoint saat ini in-memory single-instance;
  kalau nanti deploy multi-worker/multi-instance, perlu diganti limiter terpusat (mis. Redis).
