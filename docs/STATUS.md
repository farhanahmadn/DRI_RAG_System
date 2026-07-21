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
- **Bakeoff model** (`eval/bakeoff.py`) — kode siap, PACING antar-panggilan sudah ditambah supaya
  tahan rate-limit RPM, tapi **belum menghasilkan angka yang bisa dipercaya** — dua kali percobaan
  sama-sama kena kuota harian (TPD) Groq yang sudah habis dari pemakaian sepanjang sesi
  pengembangan (bukan RPM — jeda tidak membantu, terkonfirmasi via `RateLimitError` langsung).

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

- **Bakeoff model dengan kuota segar** — perlu sesi baru di luar jam pemakaian berat hari ini
  (lihat di atas). Kode & tabel siap pakai begitu kuota reset.
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
