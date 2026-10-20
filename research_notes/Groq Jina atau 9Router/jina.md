# Jina AI API (embeddings v3 dan reranker v3) untuk RAG produksi, status Oktober 2026

## 1. Skema harga, free tier, saldo, arti HTTP 403

### Takeaway
Harga berbasis token, satu saldo token per kunci API yang dipakai bersama semua produk (Embeddings, Reranker, Reader, dll). Harga daftar API resmi: USD 0,05 per 1 juta token untuk jina-embeddings-v3 dan jina-reranker-v3 (USD 0,02 untuk varian nano v5). Kunci baru mendapat 10 juta token gratis sekali (bukan bulanan). Arti kode 403 AUTHZ_INSUFFICIENT_BALANCE tidak terkonfirmasi dari dokumen resmi.

### Cited Findings
- Penagihan berdasarkan total token yang diproses; permintaan gagal tidak ditagih; bayar via Stripe, ada auto top-up; pembelian self-serve, volume/pengadaan lewat Elastic Sales. Halaman resmi tidak mencantumkan nominal dolar. — [jina.ai/embeddings](https://jina.ai/embeddings/)
- Pengguna baru mendapat kunci API otomatis dengan token gratis yang berlaku untuk semua model; jumlahnya tidak disebut di halaman itu; sisanya dibeli di tab "Buy tokens". — [jina.ai/embeddings](https://jina.ai/embeddings/)
- Reranker: token dibagi antar API pada kunci yang sama; paket token dapat dibeli. — [jina.ai/reranker](https://jina.ai/reranker/)
- Daftar model resmi (endpoint models) memuat harga `prompt` per token: jina-embeddings-v3 dan jina-reranker-v3 = 0,00000005 USD/token (= USD 0,05 per 1 juta); jina-reranker-v3.5, v4, v5-text-small, v5-omni-small juga 0,05; v5-text-nano dan v5-omni-nano = 0,02 per juta. Total 30 model terdaftar. Tidak ada field deprecasi atau lisensi di daftar itu. — [api.jina.ai/v1/models](https://api.jina.ai/v1/models)
- Hibah 10 juta token sekali untuk kunci baru, tidak reset bulanan, dibagi lintas API (Embeddings, Reranker, Reader, Search, Classifier, Segmenter); tidak ada promo per September 2026; harga jina-embeddings-v4 kontradiktif (FAQ: gratis, models API: 0,05). Sumber pihak ketiga, diperbarui 1 Sep 2026. — [markaicode](https://markaicode.com/pricing/jina-ai-pricing/)
- Klaim "free tier hanya non-komersial (CC-BY-NC)" hanya muncul di ringkasan mesin pencari, tidak ditemukan di halaman resmi: tidak terkonfirmasi. — [markaicode](https://markaicode.com/pricing/jina-ai-pricing/)
- Tentang 403: tidak ditemukan dokumentasi resmi untuk AUTHZ_INSUFFICIENT_BALANCE. Profil pihak ketiga menyebut tier ditentukan saldo token/status tagihan dan kode kuota 402 (bukan 403). Jina menyatakan top-up harus memakai kunci API yang benar. — [apis.io rate limits](https://apis.io/rate-limits/jina-ai/jina-ai-rate-limits/)

### Inferences
- Kunci menyimpan saldo token; 403 "saldo tidak cukup" hampir pasti berarti saldo kunci itu habis (atau top-up masuk ke kunci lain). Ini inferensi dari nama kode, bukan dokumen resmi.
- Hibah 10M sekali pakai dan dibagi lintas API, sehingga kunci gratis akan habis lalu gagal; produksi butuh top-up.

### Gaps
- Nominal paket top-up tidak terpublikasi di halaman yang bisa diambil (ada di dasbor).
- Arti resmi 403 AUTHZ_INSUFFICIENT_BALANCE dan apakah token kedaluwarsa: tidak terkonfirmasi.
- Cara reranker menghitung token (query + dokumen): tidak terkonfirmasi.

## 2. Batas laju, panjang input, batch

### Takeaway
Batas dikaitkan dengan tier kunci (Free/Paid/Premium); Free 100 RPM dan 100 ribu TPM. Tidak ada batas keras jumlah item per permintaan, hanya TPM dan konteks model.

### Cited Findings
- Embeddings dan Reranker per kunci: Free 100 RPM / 100.000 TPM; Paid 500 RPM / 2.000.000 TPM; Premium 5.000 RPM / 50.000.000 TPM. Tambahan batas IP 10.000 permintaan per 60 detik. Melebihi batas memberi HTTP 429; disarankan exponential backoff. — [jina.ai/embeddings](https://jina.ai/embeddings/); [jina.ai/reranker](https://jina.ai/reranker/)
- Input melebihi konteks model memberi error kecuali `truncate: true`; tidak ada batas keras jumlah item per permintaan. jina-embeddings-v3: konteks 8.192 token. — [jina.ai/embeddings](https://jina.ai/embeddings/)
- jina-reranker-v3: konteks 131.072 token dengan auto-truncation; tidak ada batas keras jumlah dokumen per permintaan (di-batch internal menurut token). — [jina.ai/reranker](https://jina.ai/reranker/)
- Syarat naik tier tidak dinyatakan di halaman resmi; pihak ketiga menyebut tier mengikuti saldo/status tagihan. — [apis.io](https://apis.io/rate-limits/jina-ai/jina-ai-rate-limits/)

### Inferences
- Beban kecil (sekitar 18 ribu token per permohonan) di bawah 100 ribu TPM Free, tetapi 6 atau lebih permohonan serentak per menit bisa menembus TPM Free.

### Gaps
- Ambang saldo untuk naik ke Paid/Premium: tidak terkonfirmasi.
- Parameter `task` (retrieval.query / retrieval.passage), `dimensions`, dan keluaran 1024 dimensi untuk v3: halaman Hugging Face v3 gagal diambil (ukuran terlalu besar), tidak diverifikasi di sesi ini.

## 3. Perubahan terbaru, deprecasi, kompatibilitas vektor

### Takeaway
Elastic menyelesaikan akuisisi Jina AI pada 9 Okt 2025. API tetap berjalan; model baru ada: embeddings v5 (text/omni, small/nano) dan reranker v3.5 (pengganti drop-in v3). v3 masih tersedia dengan harga sama; tidak ditemukan pengumuman deprecasi.

### Cited Findings
- Elastic menyelesaikan akuisisi Jina AI 9 Okt 2025; model tetap dirilis di Hugging Face dan ditawarkan lewat Elastic Inference Service; v5-text berupa dua model kecil 0,2B dan 0,6B (dari ringkasan pencarian; fetch langsung ke BusinessWire diblokir 403). — [BusinessWire](https://www.businesswire.com/news/home/20251009619654/en/Elastic-Completes-Acquisition-of-Jina-AI-a-Leader-in-Frontier-Models-for-Multimodal-and-Multilingual-Search)
- Model embedding tercantum: v5-omni-small (~1,74B, 1024 dim, 32K), v5-omni-nano (~1,04B, 768 dim, 8K), v5-text-small (677M, 32K), v5-text-nano (239M, 8K), v4 (3,8B, 32K; "gratis tapi dibatasi, tidak cocok untuk produksi"), jina-clip-v2, v3, colbert-v2. — [jina.ai/embeddings](https://jina.ai/embeddings/)
- jina-reranker-v3.5 adalah flagship saat ini dan pengganti drop-in v3; v3 masih tersedia. — [jina.ai/reranker](https://jina.ai/reranker/)
- Tidak ada repricing publik hingga ulasan Juni 2026 (sumber pihak ketiga). — [markaicode](https://markaicode.com/pricing/jina-ai-pricing/)

### Inferences
- Vektor dari model berbeda (v3 vs v5, dimensi berbeda seperti 768 pada nano) umumnya tidak kompatibel; ganti model berarti re-embed seluruh korpus. Inferensi umum, tidak ada dokumen Jina yang dikonfirmasi.
- v3 kemungkinan menjadi model lama, tetapi belum ada tanggal sunset.

### Gaps
- Tanggal deprecasi v3 dan dokumen Jina soal kompatibilitas vektor antar model: tidak ditemukan.

## 4. Kebijakan, lisensi, self-host, pihak ketiga

### Takeaway
ToS: data input/output tidak dipakai melatih model; retensi minimal. Bobot terbuka CC BY-NC 4.0; produksi komersial self-host perlu lisensi "Jina On-Prem" dari Elastic. API terhosting dicakup ToS layanan.

### Cited Findings
- Jina tidak memakai data permintaan, input, prompt, atau konten unggahan untuk melatih model (ToS 10.4); input/output disimpan hanya sejauh diperlukan (4.6); metadata operasional boleh disimpan dan dipakai secara teragregasi/anonim; log server hingga sekitar 7 hari. — [jina.ai/legal](https://jina.ai/legal/)
- Data pribadi mengikuti Elastic DPA; Jina AI GmbH berbasis di Berlin; lokasi pemrosesan inferensi tidak dinyatakan. Halaman menyatakan ketentuan ditulis sebelum akuisisi dan mungkin belum mencerminkan praktik kini. — [jina.ai/legal](https://jina.ai/legal/)
- Pelanggan bertanggung jawab atas semua aktivitas di bawah kunci dan akunnya; tidak ada ketentuan spesifik soal banyak akun/kunci; dilarang memakai Output membangun layanan pesaing dan scraping otomatis. — [jina.ai/legal](https://jina.ai/legal/)
- Lisensi self-host: v1/v2 Apache-2.0; CC BY-NC 4.0 butuh lisensi komersial (Elastic menjual "Jina On-Prem" sejak 10 Agu 2026, tahunan); v4 Qwen Research License tanpa opsi komersial. — [jina.ai/embeddings](https://jina.ai/embeddings/)
- jina-reranker-v3: CC-BY-NC 4.0; produksi komersial butuh lisensi komersial; API terhosting dicakup ketentuan layanan. — [jina.ai/reranker](https://jina.ai/reranker/)
- Jalur lain: Elastic Inference Service dan marketplace AWS/Azure/GCP. — [jina.ai/embeddings](https://jina.ai/embeddings/)

### Inferences
- API berbayar menghindari masalah CC-BY-NC; self-host untuk produksi komersial butuh lisensi. Memakai banyak kunci gratis untuk menghindari batas berisiko terhadap ToS (tidak ada larangan eksplisit, tetapi izin juga tidak terkonfirmasi).

### Gaps
- Lisensi pasti jina-embeddings-v3 di Hugging Face tidak diverifikasi (umum diketahui CC BY-NC 4.0, perlu dicek). Lokasi pemrosesan inferensi dan penyedia pihak ketiga selain Elastic/marketplace: tidak terkonfirmasi. Apakah free tier boleh komersial: tidak terkonfirmasi.

## 5. Perkiraan biaya bulanan

### Takeaway
Dengan harga daftar USD 0,05 per juta token, beban sekitar 18,1 juta token per bulan berbiaya sekitar USD 0,9 per bulan (bulan pertama sekitar USD 0,4 setelah hibah 10 juta).

### Cited Findings
- Harga 0,05 USD per juta token untuk v3 embeddings dan reranker v3; hibah 10 juta sekali. — [api.jina.ai/v1/models](https://api.jina.ai/v1/models); [markaicode](https://markaicode.com/pricing/jina-ai-pricing/)

### Inferences (perhitungan dengan asumsi saya)
- Asumsi: query pendek ~20 token; reranker menghitung query + 20 dokumen x 300 token = ~6.020 token per panggilan (cara hitung reranker tidak terkonfirmasi).
- Per permohonan: embedding 3 x 20 = 60; rerank 3 x 6.020 = 18.060; total ~18.120 token.
- 1.000 permohonan: ~18,12 juta token x 0,05 = sekitar USD 0,91/bulan. Bulan pertama: ~8,12 juta berbayar, sekitar USD 0,41.
- Batas atas bila reranker menghitung ganda: sekitar USD 1,8/bulan. Biaya nyaris tak signifikan; faktor penentu adalah ketersediaan saldo/top-up dan lisensi, bukan harga.

### Gaps
- Ukuran paket top-up minimum (bisa melebihi kebutuhan bulanan) tidak terkonfirmasi.
