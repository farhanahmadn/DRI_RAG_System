# Alternatif penyedia LLM, embedding, dan rerank untuk RAG pemda (status Oktober 2026)

Catatan metode: harga diambil dari halaman resmi lewat fetch pada 16 Okt 2026 kecuali ditandai. Fetch memakai ringkasan model kecil, jadi angka perlu diverifikasi ulang sebelum anggaran final. Tanda [3P] = pihak ketiga, bukan resmi.

## LLM: OpenRouter dan penyedia berbayar langsung

### Takeaway
gpt-oss-20b/120b tersedia di banyak penyedia berbayar (Groq, DeepInfra, OpenRouter, Together untuk 120b), jadi prompt dan guardrail gpt-oss-20b bisa dipertahankan. Biaya LLM untuk beban ini hanya ~$1 sampai $6 per bulan di jalur gpt-oss; biaya bukan pembeda, keandalan dan tata kelola data yang menentukan. Groq berbayar mendukung strict json_schema (constrained decoding) untuk gpt-oss-20b/120b, sehingga cadangan harus menjaga sifat "strict" ini.

### Cited Findings
- Groq (resmi): gpt-oss-20b $0.075 in / $0.30 out per 1M token; gpt-oss-120b $0.15 / $0.60 — [Groq models](https://console.groq.com/docs/models)
- Groq (resmi): strict json_schema (`strict: true`, constrained decoding) hanya untuk openai/gpt-oss-20b, gpt-oss-120b, dan qwen/qwen3.8-27b; mode best-effort (default) bisa menghasilkan JSON rusak atau error 400; gpt-oss-safeguard-20b hanya best-effort — [Groq structured outputs](https://console.groq.com/docs/structured-outputs)
- DeepInfra (resmi): gpt-oss-20b $0.03 in / $0.14 out (standar); tier Flex 0.8x ($0.024/$0.112); Priority 1.5x; konteks 131.072; ditandai mendukung JSON dan function calling. Dukungan *strict* json_schema tidak terkonfirmasi di halaman — [DeepInfra](https://deepinfra.com/openai/gpt-oss-20b)
- Together AI (resmi): gpt-oss-120B serverless $0.15 in / $0.60 out; gpt-oss-20B tidak tercantum di tabel serverless; tabel embedding dan rerank pada halaman yang diambil kosong ("No matching models") — [Together pricing](https://www.together.ai/pricing)
- Fireworks (resmi): halaman harga tidak mencantumkan harga serverless gpt-oss (merujuk ke dokumentasi); embedding $0.008/1M (<150M param), $0.016 (150-350M), Qwen3 8B $0.1 (penawaran terbatas); batas laju tidak terkonfirmasi — [Fireworks pricing](https://fireworks.ai/pricing)
- Cerebras (resmi): halaman memuat kredit $5 sekali pakai (kedaluwarsa 30 hari), tier dev "tidak untuk produksi"; harga gpt-oss-120b dan batas laju tidak terbaca dari halaman, tidak terkonfirmasi — [Cerebras pricing](https://www.cerebras.ai/pricing)
- OpenRouter (resmi): halaman model gpt-oss-20b menampilkan harga terendah $0.018 in / $0.09 out per 1M, konteks 131K, structured outputs didukung (level model), rilis 5 Agu 2025; daftar per-provider tidak terbaca — [OpenRouter gpt-oss-20b](https://openrouter.ai/openai/gpt-oss-20b/providers)
- OpenRouter routing (resmi): `order` mematikan load balancing default; `allow_fallbacks` default true; `require_parameters` default false, bila true hanya provider yang mendukung semua parameter; walau false, `response_format` hanya "soft preference" (model tidak dibuang dari kandidat); `data_collection: "deny"`; `zdr: true` membatasi ke endpoint Zero Data Retention (setelan akun/guardrail digabung OR; request tak bisa mematikannya) — [OpenRouter provider selection](https://openrouter.ai/docs/guides/routing/provider-selection)
- OpenRouter ZDR (resmi): ZDR berlaku hanya untuk routing inferensi; endpoint yang kebijakannya tak terverifikasi dianggap menyimpan dan melatih data; prompt tidak disimpan kecuali opt-in logging; BYOK mengikuti aturan ZDR yang sama; daftar endpoint ZDR via `/api/v1/endpoints/zdr` — [OpenRouter ZDR](https://openrouter.ai/docs/guides/features/zdr)
- OpenAI (resmi): GPT-5 mini $0.25 in / $2.00 out; GPT-5 nano $0.05 / $0.40; batch/flex setengah harga; gpt-oss tidak tercantum di pricing OpenAI — [OpenAI pricing](https://developers.openai.com/api/docs/pricing)
- Google Gemini (resmi, tier berbayar standar): 2.5 Flash-Lite $0.10 in / $0.40 out; 3.1 Flash-Lite $0.25/$1.50; 2.5 Flash $0.30/$2.50; 3.5 Flash-Lite $0.30/$2.50; seri Flash 3.6/3.8 $0.75/$3.75 sampai 31 Des 2026 lalu naik jadi $1.50/$7.50 mulai 1 Jan 2027. Output termasuk thinking token. Tier gratis: konten dipakai untuk meningkatkan produk Google; tier berbayar: tidak — [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)
- Anthropic: tidak diteliti (tidak ada fetch); tidak terkonfirmasi.

### Inferences
- Jalur paling sedikit perubahan: Groq berbayar (kunci resmi, tier Developer) sebagai primer dan DeepInfra atau OpenRouter (dengan `provider.order`, `require_parameters: true`, `zdr`/`data_collection: deny`) sebagai cadangan, semuanya model gpt-oss-20b yang sama.
- Karena `response_format` di OpenRouter hanya soft preference tanpa `require_parameters: true`, request json_schema strict bisa jatuh ke provider yang mengabaikan schema; wajib setel `require_parameters: true` dan tetap validasi schema di sisi klien.
- GPT-5 nano / Gemini Flash-Lite adalah cadangan lintas-model yang sah tetapi mengubah perilaku, jadi guardrail tuning gpt-oss tidak berlaku otomatis; perlu evaluasi ulang.

### Gaps
- Batas laju resmi Groq berbayar, DeepInfra, Together, Fireworks, Cerebras tidak terambil.
- Daftar provider OpenRouter untuk gpt-oss-20b (harga, kuantisasi, uptime, mana yang ZDR) tidak terambil; biaya platform OpenRouter (fee kredit/BYOK) tidak terkonfirmasi.
- Harga Cerebras, harga Fireworks gpt-oss, harga Anthropic, ketersediaan gpt-oss-20b di Together: tidak terkonfirmasi.
- Dukungan strict json_schema di DeepInfra, Together, Fireworks, Cerebras: tidak terkonfirmasi.

## Embedding dan rerank pengganti Jina

### Takeaway
Untuk korpus 2.000-10.000 chunk, self-host bge-m3 (1024 dimensi, MIT, 8192 token) plus bge-reranker-v2-m3 (Apache-2.0, 0,6B) di CPU cukup dan menghapus ketergantungan API pihak ketiga; pilihan API: Voyage (murah, 200M token gratis) atau Cohere. Mengganti model embedding wajib embed ulang seluruh korpus.

### Cited Findings
- bge-m3: dimensi 1024, panjang maks 8192 token, lisensi MIT; angka MIRACL per bahasa (termasuk Indonesia) hanya berupa gambar, tidak terbaca — [HF bge-m3](https://huggingface.co/BAAI/bge-m3)
- bge-reranker-v2-m3: cross-encoder, 0,6B parameter, Apache-2.0, ditandai multilingual dan "ringan, mudah di-deploy"; skor mentah contoh -8,19 (tak relevan) sampai 5,26 (relevan); benchmark hanya gambar — [HF bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- Voyage (resmi): voyage-4 $0.06, voyage-4-lite $0.02, voyage-4-large $0.12 per 1M token (masing-masing 200M token gratis); voyage-3.5 $0.06; voyage-3-large $0.18; rerank-2.5 $0.05 (200M gratis), rerank-2.5-lite $0.02; dimensi tidak tercantum di halaman — [Voyage pricing](https://docs.voyageai.com/docs/pricing)
- Cohere (resmi): produksi bayar sesuai pemakaian, butuh aplikasi "Go to Production"; trial key tidak boleh untuk komersial/produksi; trial 1.000 panggilan API/bulan; limit embed 2.000 input/mnt, rerank 10 req/mnt (trial) vs 1.000 req/mnt (produksi). Harga per token/search tidak tampil di halaman pricing yang diambil — [Cohere pricing](https://cohere.com/pricing), [rate limits](https://docs.cohere.com/docs/rate-limits)
- Cohere [3P]: Embed v4 sekitar $0.12/1M token teks (Azure/marketplace), Rerank 3.5 sekitar $2 per 1.000 search (1 search = 1 query + hingga 100 dokumen), ada daftar yang menyebut $1/1.000; tidak resmi — [ringkasan pencarian](https://futureagi.com/llm-cost-calculator/cohere/embed-v4-0)
- OpenAI (resmi): text-embedding-3-large $0.13, -small $0.02 per 1M (3-large bisa dipotong ke 1024 dimensi lewat parameter dimensions; tidak diverifikasi di sumber ini) — [OpenAI pricing](https://developers.openai.com/api/docs/pricing)
- Google (resmi): gemini-embedding-2 teks $0.20/1M standar, $0.10 batch; tier gratis ada — [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)
- Together/Fireworks: embedding murah ($0.008-$0.1/1M) di Fireworks; ketersediaan bge-m3 di sana tidak terkonfirmasi — [Fireworks pricing](https://fireworks.ai/pricing)

### Inferences
- Ganti dari Jina ke model lain (termasuk bge-m3 bila Jina bukan bge-m3) mengubah ruang vektor: semua chunk harus di-embed ulang; untuk 10.000 chunk x ~400 token = 4M token, biaya API ~$0.08-$0.52, waktu menit. Menjaga dimensi 1024 tetap mengubah ruang vektor.
- Pada CPU 4 vCPU/8 GB, bge-m3 (~568M parameter, perkiraan, tidak ada sumber hardware resmi) dan reranker 0,6B layak untuk beban rendah dengan latensi ratusan ms sampai beberapa detik per batch; perlu benchmark sendiri. Ini perkiraan, bukan fakta resmi.
- Self-host memenuhi kebutuhan kedaulatan data untuk query dan chunk peraturan, dengan biaya operasional (patching, monitoring) pada VPS yang sudah ada.

### Gaps
- Skor MIRACL/MTEB bahasa Indonesia untuk bge-m3, Voyage, Cohere, OpenAI tidak terbaca (hanya gambar atau tidak diakses); bandingkan kualitas dengan set uji sendiri.
- Kebutuhan hardware resmi (RAM/CPU) untuk bge-m3, TEI, Infinity tidak ditemukan; harga resmi Cohere Embed v4 dan Rerank 3.5 tidak terkonfirmasi; dimensi Voyage-4 dan dukungan Matryoshka tidak terkonfirmasi.

## Perbandingan biaya bulanan

### Takeaway
Seluruh opsi gpt-oss berada di ~$1-$6/bulan untuk LLM; opsi proprietary $3-$17. Embedding dan rerank hampir nol (gratis di Voyage atau self-host).

### Cited Findings
Asumsi (milik saya): 1.000 permohonan x 5 panggilan = 5.000 panggilan; input 15M token, output 5M token (termasuk reasoning). Embedding: 3.000 query x ~50 token = 0,15M token. Rerank: 3.000 panggilan x 20 dokumen x ~400 token = ~24M token (asumsi), atau 3.000 search unit. Perhitungan dari harga resmi di atas, tanpa pajak, fee platform, cache.

| Opsi LLM | Hitungan | Per bulan |
|---|---|---|
| OpenRouter gpt-oss-20b termurah | 15x0,018 + 5x0,09 | ~$0,72 (harga terendah; provider belum tentu strict/ZDR; fee platform tak terkonfirmasi) |
| DeepInfra gpt-oss-20b | 15x0,03 + 5x0,14 | ~$1,15 |
| Groq gpt-oss-20b | 15x0,075 + 5x0,30 | ~$2,63 |
| GPT-5 nano | 15x0,05 + 5x0,40 | ~$2,75 |
| Gemini 2.5 Flash-Lite | 15x0,10 + 5x0,40 | ~$3,50 |
| Groq / Together gpt-oss-120b | 15x0,15 + 5x0,60 | ~$5,25 |
| GPT-5 mini | 15x0,25 + 5x2,00 | ~$13,75 |
| Gemini 2.5 Flash | 15x0,30 + 5x2,50 | ~$17,00 |

| Embedding / rerank | Per bulan |
|---|---|
| Embedding query (Voyage-4 / OpenAI small / bge-m3 lokal) | <$0,01 / <$0,01 / $0 |
| Rerank Voyage rerank-2.5 (24M token) | $1,20 (gratis bila dalam 200M token gratis) |
| Rerank Cohere 3.5 (3.000 search x ~$2/1.000 [3P]) | ~$6 |
| Rerank bge-reranker-v2-m3 lokal | $0 tambahan (VPS sudah ada) |

### Inferences
- Total realistis: ~$3-$8/bulan dengan Groq 20b + Voyage atau self-host; selisih antar penyedia lebih kecil daripada biaya satu jam kerja perbaikan insiden.

### Gaps
- Panjang rerank sebenarnya tidak diketahui; jumlah dokumen per panggilan dan token reasoning belum terukur.

## Tata kelola data pemda

### Takeaway
UU PDP Pasal 56 membolehkan transfer ke luar negeri dengan syarat berjenjang; PP 71/2019 membedakan lingkup publik dan privat, dan data pemerintah memiliki aturan penempatan lebih ketat. Dengan PII disanitasi risiko turun, tetapi kepastian hukum perlu konfirmasi ke biro hukum dan kominfo daerah.

### Cited Findings
- Pasal 56 UU PDP (UU 27/2022): transfer ke luar Indonesia dimungkinkan bila negara penerima memiliki pelindungan setara/lebih tinggi, atau ada pelindungan memadai dan mengikat, atau persetujuan subjek data — [Hukumonline](https://www.hukumonline.com/berita/a/pasal-56-uu-pdp-dinilai-belum-memadai-melindungi-subjek-data-pribadi-lt68f98c9417fa9) (ringkasan sekunder)
- Pasal 56 diuji di MK (perkara 137/PUU-XXIII/2025); putusan 19 Januari 2026 dilaporkan, amar putusan tidak terkonfirmasi — [Hukumonline](https://www.hukumonline.com/berita/a/transfer-data-lintas-negara-dan-ilusi-pelindungan-data-pribadi-pasca-putusan-mk-lt6981468330d9c)
- PP 71/2019 Pasal 21(1): PSE lingkup privat boleh mengelola/menyimpan di dalam atau luar Indonesia; lingkup publik (instansi penyelenggara negara) diatur terpisah — [Selular.ID](https://selular.id/2019/10/johnny-plate-bahas-isi-pasal-21-pp-71-2019/); teks resmi: [JDIH Kemenkeu](https://jdih.kemenkeu.go.id/dok/pp-71-2019/view)
- Rencana revisi PP 71/2019 diumumkan 2023; apakah sudah terbit tidak terkonfirmasi — [Bisnis.com](https://teknologi.bisnis.com/read/20231124/101/1717649/pp-no712019-tentang-penyimpanan-data-pribadi-bakal-direvisi-pada-2024)
- Opsi ZDR / tanpa pelatihan: OpenRouter (`zdr`, `data_collection: deny`), Gemini tier berbayar tidak melatih dari konten — sumber di bagian LLM. Opsi region/data residency Indonesia untuk Groq, DeepInfra, Together: tidak terkonfirmasi.

### Inferences
- Karena pengirim adalah instansi pemerintah daerah (lingkup publik), pengiriman teks permohonan ke API luar negeri sebaiknya dikonsultasikan ke Diskominfo; menyanitasi PII dan mengirim hanya potongan teks yang diperlukan mengurangi risiko.
- Menjadikan embedding dan rerank self-host menghindari pengiriman query dan korpus ke luar negeri; sisa paparan hanya panggilan LLM.

### Gaps
- Isi resmi PP 71/2019 Pasal 20-21 untuk lingkup publik, amar putusan MK, regulasi turunan UU PDP dan pembentukan lembaga pengawas tidak diverifikasi dari sumber primer.

## Rekomendasi arsitektur dan failover

### Takeaway
Paling stabil dan murah untuk beban kecil: LLM gpt-oss-20b di Groq berbayar (kunci resmi) dengan cadangan penyedia berbayar kedua (DeepInfra atau OpenRouter dengan `require_parameters`+`zdr`), embedding dan rerank self-host bge-m3 / bge-reranker-v2-m3 (Voyage sebagai cadangan API).

### Cited Findings
- Dasar fitur: strict json_schema pada Groq gpt-oss; kontrol routing OpenRouter (`order`, `allow_fallbacks`, `require_parameters`, `zdr`) — lihat bagian LLM.
- Cohere trial dilarang untuk produksi/komersial — [Cohere pricing](https://cohere.com/pricing). Tier gratis Gemini memakai konten untuk perbaikan produk — [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing).

### Inferences
- Failover sah: dua penyedia berbayar dengan kunci atas nama instansi (Groq + DeepInfra/OpenRouter), model sama, validasi schema di klien, retry dengan backoff, circuit breaker, dan anggaran/ batas pengeluaran per kunci. Tidak merotasi akun gratis atau kunci banyak akun untuk menghindari kuota (melanggar ToS umum penyedia; saya tidak merekomendasikannya).
- Embedding/rerank dengan model sama (bge-m3 melalui Voyage tidak sama) harus tetap konsisten; cadangan embedding dengan model berbeda butuh indeks vektor terpisah, jadi lebih mudah menjaga self-host dan snapshot indeks.
- Uji regresi: jalankan set evaluasi 4 payload yang ada terhadap penyedia cadangan sebelum diaktifkan.

### Gaps
- SLA/uptime historis tiap penyedia tidak ditemukan; kompatibilitas guardrail pada GPT-5 nano/Gemini belum diuji.
