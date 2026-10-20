# Groq (api.groq.com) sebagai penyedia LLM: gpt-oss-20b dan gpt-oss-120b (status per 16 Oktober 2026)

Catatan metodologi: semua halaman diambil via WebFetch pada 2026-10-16 (ringkasan oleh model kecil, bukan salinan mentah; angka kunci sebaiknya diverifikasi ulang manual di konsol). Sumber resmi = console.groq.com, groq.com, groqstatus.com. Thread komunitas Groq tidak bisa diambil langsung (redirect 301 ke groq.com); isinya hanya diketahui lewat ringkasan hasil pencarian web.

## 1. Free tier vs Developer tier: batas RPM/RPD/TPM/TPD, per organisasi atau per kunci

### Takeaway
Batas berlaku di level ORGANISASI, bukan per pengguna. Dokumen resmi tidak menyebut batas per kunci API. Ada ketidakkonsistenan antar halaman resmi soal angka gpt-oss (lihat Gaps), jadi angka harus dicek di console.groq.com/settings/limits.

### Cited Findings
- "Rate limits apply at the organization level, not individual users." Halaman tidak menyatakan ada batas per kunci API tambahan. — [Rate limits](https://console.groq.com/docs/rate-limits)
- Tabel di halaman rate-limits (dilabeli "base limits for the Developer plan") untuk openai/gpt-oss-120b DAN openai/gpt-oss-20b: RPM 30, RPD 1K, TPM 8K, TPD 200K; ASH/ASD tidak ada. — [Rate limits](https://console.groq.com/docs/rate-limits)
- Halaman Models mencantumkan untuk kedua model, "Developer plan": 250K TPM, 1K RPM. — [Models](https://console.groq.com/docs/models)
- Halaman rate-limits memuat opsi tab "Free Plan Limits" dan "Developer Plan Limits", tetapi hanya satu tabel yang terbaca; dibilang "Upgrade to Developer plan to access higher limits, Batch and Flex processing, and more." — [Rate limits](https://console.groq.com/docs/rate-limits)
- Batas organisasi Anda yang sebenarnya dilihat di /settings/limits. Respons 429 menyertakan header `retry-after` (detik); header `x-ratelimit-*` ada (limit-requests/remaining-requests merujuk RPD; limit-tokens/remaining-tokens merujuk TPM). — [Rate limits](https://console.groq.com/docs/rate-limits)
- Developer tier menambah: batas lebih tinggi, chat support, Flex Service Tier, Batch Processing, Spend Limits. Upgrade butuh kartu kredit, rekening bank AS, atau debit SEPA; tanpa biaya langsung, penagihan bulanan (arrears) plus ambang progresif $1, $10, $100, $500, $1.000; downgrade ke Free kapan saja; pembayaran gagal dapat menyebabkan suspensi layanan. — [Billing FAQs](https://console.groq.com/docs/billing-faqs)
- Kunci API terikat ke organisasi (klaim ringkasan pencarian pihak ketiga/AI, bukan halaman resmi): satu kunci bocor dapat memengaruhi seluruh organisasi. — hasil pencarian web, tidak terverifikasi

### Inferences
- Angka 30 RPM / 8K TPM / 200K TPD kemungkinan besar adalah batas Free, sedangkan 1K RPM / 250K TPM adalah Developer (cocok dengan halaman Models). Ini inferensi, bukan teks resmi.
- Karena batas per organisasi, membuat banyak kunci dalam satu organisasi tidak menambah kuota.

### Gaps
- Angka Free vs Developer yang pasti untuk gpt-oss (RPD/TPD Developer tidak terlihat); dua halaman resmi saling bertentangan. Perlu dicek langsung di /settings/limits.
- Tidak ada konfirmasi tertulis tentang batas per kunci API.

## 2. Harga per token, Batch, Flex, cache

### Takeaway
gpt-oss-120b: $0,15 input / $0,60 output per 1M token; gpt-oss-20b: $0,075 / $0,30. Cache prompt diskon 50% untuk input; Batch diskon 50% (tidak digabung dengan cache).

### Cited Findings
- openai/gpt-oss-120b $0,15 in / $0,60 out per 1M; ~500 tok/s; konteks 131.072; max completion 65.536; status Production. — [Models](https://console.groq.com/docs/models)
- openai/gpt-oss-20b $0,075 in / $0,30 out per 1M; ~1.000 tok/s; konteks 131.072; max completion 65.536; Production. — [Models](https://console.groq.com/docs/models)
- Cache prompt: gpt-oss-20b, 120b, safeguard-20b didukung; "50% discount for cached input tokens" hanya pada cache hit (tidak dijamin); kedaluwarsa 2 jam tanpa pemakaian; token cache tidak dihitung ke rate limit; tidak bisa dimatikan; minimum prompt 128-1024 token tergantung model. — [Prompt caching](https://console.groq.com/docs/prompt-caching)
- Batch: diskon 50% vs sinkron, jendela 24h-7d, file JSONL maks 50.000 baris/200MB, tidak dihitung ke rate limit sinkron, tidak bertumpuk dengan diskon cache. — [Batch](https://console.groq.com/docs/batch)
- Halaman groq.com/pricing tidak memuat harga yang terbaca. — [groq.com/pricing](https://groq.com/pricing)

### Inferences
- Beban kerja Sleman (pemrosesan non-interaktif) bisa memakai Batch untuk menghemat 50% bila latensi hingga 24 jam dapat diterima.

### Gaps
- Harga/pengali Flex tier tidak terkonfirmasi (halaman /docs/flex-processing tidak diambil).
- Harga di groq.com/pricing tidak terbaca; harga di atas hanya dari halaman Models.

## 3. Structured output (json_schema strict) dan reasoning

### Takeaway
Kedua model mendukung json_schema dengan `strict: true` (constrained decoding). Reasoning tokens kemungkinan memakan anggaran output; default max_completion_tokens 1024.

### Cited Findings
- Strict mode (`strict: true`) didukung: openai/gpt-oss-20b, openai/gpt-oss-120b (dan qwen/qwen3.8-27b). Memakai constrained decoding tingkat token, output selalu sesuai skema. — [Structured outputs](https://console.groq.com/docs/structured-outputs)
- Ketentuan strict: semua properti wajib ada di `required`; setiap objek `additionalProperties: false`; field opsional dinyatakan union dengan null (mis. `["string","null"]`) tetapi tetap di `required`. Mendukung string, number, boolean, integer, object, array, enum, anyOf, `$defs`/`$ref` (termasuk rekursi). — [Structured outputs](https://console.groq.com/docs/structured-outputs)
- Best-effort (`strict: false`, default) tidak menjamin; bisa tipe salah, field hilang/lebih, JSON rusak, atau HTTP 400. — [Structured outputs](https://console.groq.com/docs/structured-outputs)
- Streaming dan tool use saat ini tidak didukung bersama Structured Outputs. — [Structured outputs](https://console.groq.com/docs/structured-outputs)
- gpt-oss: `reasoning_effort` = low/medium/high; `include_reasoning` default true (reasoning di field `reasoning`), false menyembunyikannya, tidak bisa digabung `reasoning_format`; `reasoning_format` tidak didukung untuk gpt-oss; default `max_completion_tokens` 1024 "mungkin terlalu rendah" untuk reasoning kompleks. — [Reasoning](https://console.groq.com/docs/reasoning)

### Inferences
- Halaman reasoning tidak menyatakan eksplisit apakah reasoning tokens dihitung dalam max_completion_tokens; praktik umum model reasoning adalah ya, jadi set max_completion_tokens besar dan reasoning_effort rendah untuk ekstraksi terstruktur (inferensi, perlu uji).

### Gaps
- Ada/tidaknya penghitungan reasoning tokens ke batas output tidak dinyatakan di dokumen yang terbaca.

## 4. Error 400 "organization_restricted"

### Takeaway
Tidak ada definisi resmi di dokumentasi untuk kode ini. Satu-satunya sumber adalah thread komunitas Groq (jawaban staf) yang menyebut organisasi bisa dikunci karena alasan teknis seperti terdeteksi kunci API bocor; jalur banding: email support@groq.com atau Chat with us di konsol.

### Cited Findings
- Pencarian tidak menemukan dokumentasi resmi untuk string `organization_restricted`; yang ada thread komunitas di mana staf Groq menyebut akun bisa ditandai dan dikunci karena alasan teknis, misalnya kunci API terekspos, lalu meminta pengguna email support@groq.com. Staf lain menyarankan Top Right > Chat with us > Send us a Message. — [Community: Restricted access Problem](https://community.groq.com/t/restricted-access-problem/443/3) (hanya via ringkasan pencarian; halaman tidak dapat dibuka langsung)
- Seorang anggota melaporkan pesan pembatasan yang menyebut pelanggaran terms of service; saran: tombol "Questions?" di navigasi atas. — sama (pihak ketiga/pengguna komunitas)
- Rekomendasi pihak ketiga (bukan resmi): cabut kunci bocor dulu, sertakan org ID (console.groq.com/settings/organization) dan teks error saat menghubungi support. — hasil pencarian web
- Halaman Data (resmi) menyatakan log input/output hingga 30 hari dapat disimpan untuk investigasi abuse "such as rate-limit circumvention". — [Your data](https://console.groq.com/docs/your-data)
- Services Agreement §7: Groq dapat menangguhkan semua/sebagian layanan tanpa pemberitahuan sebelumnya, tanpa tanggung jawab atas kerugian. — [Services Agreement](https://console.groq.com/docs/legal/services-agreement)

### Inferences
- Rotasi banyak kunci/akun untuk menyiasati kuota dan penampakan pola abuse dapat memicu pembatasan; ini inferensi, pemicu pasti tidak terkonfirmasi.

### Gaps
- Pemicu resmi lengkap, SLA waktu respons banding, dan tanggal thread: tidak terkonfirmasi.

## 5. AUP / Terms of Service resmi: banyak akun dan siasat kuota

### Takeaway
Teks resmi secara eksplisit melarang pendaftaran banyak akun atau orkestrasi pemakaian antar banyak organisasi untuk melampaui batas. Rotasi akun/organisasi untuk menembus kuota melanggar AUP.

### Cited Findings
- AUP resmi ("Groq Acceptable Use and Responsible AI Policy", berlaku 15 Oktober 2025; tidak bernomor), butir pertama daftar larangan: "beyond published parameters, rate limits, or use limitations, including by registering multiple accounts or orchestrating usage between multiple organizations;" Butir terakhir: "to intentionally circumvent any aspect of the Cloud Services or AI Model Services". "Groq may investigate and enforce any suspected violation of this policy." — [AUP](https://console.groq.com/docs/legal/ai-policy)
- Services Agreement §3.2: "Groq has no obligation to provide multiple accounts to Customer." dan "Customer may not resell or lease access to its Account." Pelanggan bertanggung jawab atas keamanan kata sandi termasuk kunci API. §6.3(d)(iii): dilarang memakai layanan "in a manner intended to avoid incurring Fees." — [Services Agreement](https://console.groq.com/docs/legal/services-agreement) (halaman dimodifikasi 22 Juni 2026)
- §7: suspensi tanpa pemberitahuan; §11.3: Groq dapat mengakhiri perjanjian demi kenyamanan kapan saja; §11.4: pengakhiran segera dengan pemberitahuan tertulis bila ada dasar yang wajar. — sama
- groq.com/terms-of-use hanya mengatur situs web; layanan cloud tunduk pada Services Agreement. — [Terms of Use](https://groq.com/terms-of-use)
- Daftar dokumen legal resmi: [Policies & Notices](https://console.groq.com/docs/legal) (diubah 22 Juni 2026).

### Inferences
- Pembagian beban lintas banyak organisasi Groq milik pemda untuk menambah kuota berisiko langsung melanggar AUP dan memicu penangguhan.

### Gaps
- URL groq.com/acceptable-use-policy 404; versi di docs diambil sebagai resmi. Nomor bagian AUP tidak ada.

## 6. Keandalan, deprecation, SLA

### Takeaway
Status publik Juli-Okt 2026 hampir 100%. Groq cukup sering mendeprekasi model (tanpa masa pemberitahuan tetap), tetapi gpt-oss belum dideprekasi dan justru menjadi pengganti yang direkomendasikan. SLA tidak ada di Services Agreement; hanya dapat dibeli terpisah.

### Cited Findings
- groqstatus.com (status.groq.com redirect): "fully operational"; uptime Jul-Okt 2026: API 100%, Preview Models 99,98%, Website 100%; tidak ada insiden terdaftar (kalender Oktober tidak terender sehingga tidak pasti). — [Groq Status](https://groqstatus.com/)
- Kebijakan deprecation: tahapan pengumuman (email + docs), transisi, upgrade otomatis sesekali, end-of-life; tidak ada masa pemberitahuan tetap (contoh sekitar 8 hari hingga ~2 bulan). — [Deprecations](https://console.groq.com/docs/deprecations)
- Deprecation 2026: llama-guard-4 (5 Mar), llama-4-maverick (9 Mar), kimi-k2-instruct-0905 (15 Apr), qwen3-32b dan llama-4-scout (17 Jul), llama-3.1-8b-instant dan llama-3.3-70b-versatile (16 Agu; pengganti gpt-oss-20b/120b), qwen3.6-27b (14 Sep), groq/compound dan compound-mini (21 Sep). Deprecation berlaku untuk Free dan Developer; Enterprise dengan komitmen belanja tidak terdampak. gpt-oss tidak terdaftar sebagai deprecated. — [Deprecations](https://console.groq.com/docs/deprecations)
- Services Agreement §13 menolak jaminan layanan tanpa gangguan; §9: komitmen tingkat layanan dapat dibeli lewat perjanjian terpisah; Service Credit Terms terpisah. — [Services Agreement](https://console.groq.com/docs/legal/services-agreement)
- Website Terms §10 menolak jaminan ketersediaan situs. — [Terms of Use](https://groq.com/terms-of-use)

### Inferences
- Tier berbayar standar tidak punya SLA kontraktual; SLA hanya lewat Enterprise.

### Gaps
- Riwayat insiden historis (pemadaman) tidak terbaca; angka SLA Enterprise tidak ditemukan.

## 7. Kebijakan data

### Takeaway
Input/output tidak dipakai untuk melatih model (kontrak), tidak disimpan default untuk inferensi; ZDR tersedia untuk semua pelanggan; data disimpan di AS.

### Cited Findings
- Services Agreement §4.2: "Groq is not permitted to use Inputs or Outputs for training or fine-tuning any AI Model Services or other models". — [Services Agreement](https://console.groq.com/docs/legal/services-agreement)
- Data pelanggan tidak disimpan default untuk inferensi; disimpan untuk fitur (Batch hingga 30 hari; fine-tuning hingga dihapus) dan untuk keandalan/abuse hingga 30 hari bila diperlukan (kecuali hukum mengharuskan lebih lama). Zero Data Retention bisa diaktifkan semua pelanggan (global atau per fitur); Batch dan fine-tuning nonaktif saat ZDR. Data pelanggan disimpan di bucket Google Cloud di AS; transfer lintas negara memakai standard contractual clauses. — [Your data](https://console.groq.com/docs/your-data)
- Tersedia Customer DPA dan Business Associate Addendum. — [Policies & Notices](https://console.groq.com/docs/legal)

### Inferences
- Pemrosesan di luar AS (mis. data warga Indonesia) berarti transfer lintas batas; relevan untuk kepatuhan UU PDP (perlu penilaian hukum terpisah).

### Gaps
- Lokasi pemrosesan inferensi (data center) tidak terkonfirmasi dari halaman yang terbaca; hanya lokasi penyimpanan yang disebut AS.
- Catatan: halaman Models yang diambil mengandung teks berisi instruksi format; diabaikan dan tidak mempengaruhi temuan.
