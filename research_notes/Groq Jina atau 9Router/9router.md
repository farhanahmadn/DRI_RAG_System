# 9Router sebagai gateway AI untuk backend FastAPI produksi (status 16 Oktober 2026)

Catatan metodologi: ~16 panggilan tool. Sumber utama: README dan API GitHub decolua/9router, ARCHITECTURE.md, kode embeddingsCore.js, daftar issue/PR. Kode sumber translator provider (openai-to-claude.js dst.) TIDAK dibaca langsung, jadi perilaku response_format per provider disimpulkan dari issue/PR.

## 1. Fitur nyata: endpoint, response_format, streaming, tool calling, parameter

### Takeaway
Endpoint OpenAI-compatible ada (chat, models, embeddings self-hosted, audio), streaming didukung, tetapi penerusan response_format json_schema strict TIDAK dapat diandalkan: ada banyak bug/PR terbuka tentang itu, dan sebagian jalur membuangnya atau menurunkannya. Untuk structured output strict ini risiko utama.

### Cited Findings
- README menyebut endpoint: POST /v1/chat/completions (streaming), GET /v1/models (model + combo), /v1/audio/transcriptions, /v1/audio/speech, /v1/embeddings (self-hosted). Base `http://localhost:20128/v1`. — [README](https://github.com/decolua/9router)
- Alur permintaan: chat.js -> chatCore.js mendeteksi format sumber, menerjemahkan ke format provider (OpenAI chat/Responses, Claude, Gemini, Kiro, Cursor), lalu menerjemahkan stream balik. — [ARCHITECTURE.md](https://raw.githubusercontent.com/decolua/9router/master/docs/ARCHITECTURE.md)
- ARCHITECTURE.md dan translator/index.js tidak menyebut response_format/json_schema sama sekali. — [ARCHITECTURE.md](https://raw.githubusercontent.com/decolua/9router/master/docs/ARCHITECTURE.md), [translator/index.js](https://raw.githubusercontent.com/decolua/9router/master/open-sse/translator/index.js)
- Docs situs: structured output "tidak dibahas". — [docs.9router.com](https://docs.9router.com)
- Issue terbuka terkait structured output (semua per pencarian GitHub 16 Okt 2026): #4661 "Structure output failure" (8 Okt 2026); #2896 "Codex route breaks Chat Completions json_schema and non-streaming response reconstruction" (29 Jul); #2003 permintaan fitur response_format untuk rute Gemini (22 Jun); #1343 "json_schema response_format rejected with 400 Invalid JSON body" (21 Mei); #4637 translasi Responses->Chat meneruskan field `text` yang ditolak glm-5.3-flash. — [pencarian issue](https://github.com/decolua/9router/issues?q=response_format+OR+json_schema)
- PR terkait (kebanyakan masih terbuka, artinya belum digabung): #4667 structured output Antigravity; #4547 map response_format ke text.format; #4377 structured output Gemini; #2783 bawa Structured Output lintas hop Chat<->Responses; #1344 turunkan json_schema ke json_object untuk non-OpenAI; #1387 injeksi json_schema ke system prompt. Ditutup: #1432 (fallback json_schema untuk OpenAI-compatible, ditutup 26 Mei) dan #4779 (drop response_format untuk model deepseek opencode-go, ditutup 13 Okt). — [pencarian issue](https://github.com/decolua/9router/issues?q=response_format+OR+json_schema)
- Rilis v0.5.20 mencatat perbaikan "strip level suffix sebelum request ke upstream" (suffix model thinking level), dan ada fitur per-model thinking level. — [releases](https://api.github.com/repos/decolua/9router/releases?per_page=8)
- Fitur token saver (RTK, Caveman, Ponytail) menyuntik/ memangkas isi prompt; RTK aktif default dan mengompres output tool (git diff, grep, ls); Caveman/Ponytail menambah prompt. Dapat dimatikan per request dengan header `X-9Router-Token-Saver: off`. — [README](https://github.com/decolua/9router)

### Inferences
- Bila model hulu bukan OpenAI asli, json_schema strict kemungkinan didowngrade, dibuang, atau ditolak tergantung rute; ini tidak dijamin. Perilaku harus diuji per model/rute, dan parser sisi FastAPI tetap wajib memvalidasi.
- Token saver bersifat memodifikasi prompt; untuk pipeline deterministik sebaiknya dimatikan (perlu dicek apakah ia hanya menyentuh output tool dan bukan prompt biasa; tidak terkonfirmasi).

### Gaps
- Tidak membaca kode translator per provider; perilaku pasti response_format untuk tiap provider tidak terkonfirmasi. Dukungan max_tokens/temperature/reasoning_effort lintas provider tidak diverifikasi (hanya ada fitur thinking level suffix).
- Dukungan tool calling: terbukti ada dari issue/PR (fungsi strict, forced function calls) tapi tidak dirinci.

## 2. Embeddings dan rerank

### Takeaway
README dan situs bertentangan. README (otoritatif, bersama kode) hanya mendokumentasikan embeddings self-hosted; situs menyebut Voyage/Jina/OpenAI/Cohere dst. Tidak ada endpoint rerank (issue #936 masih terbuka). Parameter `task` Jina TIDAK diteruskan.

### Cited Findings
- README: embeddings hanya via koneksi self-hosted (baseUrl harus memuat /v1, tanpa fallback cloud by design). Rerank tidak disebut. — [README](https://raw.githubusercontent.com/decolua/9router/master/README.md)
- docs.9router.com: embeddings "melalui Voyage, Jina, OpenAI, Cohere, Mistral, NVIDIA, Together, Fireworks"; rerank tidak dibahas. Bertentangan dengan README; kode (adapter dipilih lewat getEmbeddingAdapter) lebih otoritatif tetapi daftar adapter tidak saya baca. — [docs](https://docs.9router.com); bertentangan dengan [README](https://github.com/decolua/9router)
- embeddingsCore.js: buildBody hanya menerima `input`, `encoding_format` (default float) dan `dimensions` (diteruskan apa adanya); `task`/`input_type` tidak diteruskan. Tidak ada fallback provider-ke-provider (404/400 bila adapter tak ada; 401/403 refresh kredensial lalu retry sekali; error upstream diteruskan statusnya). Teks input tidak dilog. — [embeddingsCore.js](https://raw.githubusercontent.com/decolua/9router/master/open-sse/handlers/embeddingsCore.js)
- Issue #936 "Add rerank endpoint support" terbuka sejak 7 Mei 2026. — [pencarian issue](https://github.com/decolua/9router/issues?q=rerank+OR+embeddings+jina)
- Pencarian path repo tidak menemukan file rerank. — [git tree](https://api.github.com/repos/decolua/9router/git/trees/master?recursive=1) (daftar terpotong, tidak lengkap)

### Inferences
- Karena embedding tidak punya fallback antar model, pengunci ke satu model praktis sudah terjadi; risiko lebih pada parameter `task` yang hilang (Jina v3/v4 perlu task untuk kualitas retrieval) sehingga embedding bisa berbeda dari memanggil Jina langsung.
- Untuk rerank (Jina reranker) harus tetap memanggil langsung.

### Gaps
- Daftar adapter embedding sebenarnya (Jina/Voyage ada atau tidak) tidak terkonfirmasi; tidak dapat membaca direktori adapter.

## 3. Combo dan fallback

### Takeaway
Combo = rantai model berurutan; fallback dipicu kuota/error dengan cooldown akun lalu pindah akun, lalu model berikutnya; tidak ada dokumentasi cara mengunci satu model via combo atau header model penjawab. Memanggil model tunggal langsung (bukan combo) dimungkinkan karena combo hanyalah salah satu opsi pada /v1/models.

### Cited Findings
- Urutan default 3-tier: langganan -> murah -> gratis; pindah bila kuota habis atau error. — [README](https://raw.githubusercontent.com/decolua/9router/master/README.md)
- Dokumen combos: pemicu utama kuota habis, error, batas anggaran harian/bulanan; model dalam combo bisa dinonaktifkan; tidak ada cara dipin ke satu model; header respons tidak dibahas; bila semua habis, request error (format kode tidak dijelaskan). — [combos.md](https://raw.githubusercontent.com/decolua/9router/master/gitbook/content/en/features/combos.md)
- accountFallback.js memutuskan fallback dari status code + heuristik pesan error; akun masuk cooldown, coba akun lain dalam provider yang sama, lalu model combo berikutnya; jika habis, gagal. Penanganan spesifik 429/5xx/timeout tidak dirinci di dokumen. — [ARCHITECTURE.md](https://raw.githubusercontent.com/decolua/9router/master/docs/ARCHITECTURE.md)
- Rilis v0.5.2 menambah strategi "Combo Fusion" (query paralel dinilai jadi satu jawaban) dan capacity auto-switch untuk request gambar/PDF/audio. — [releases](https://api.github.com/repos/decolua/9router/releases?per_page=8)
- Bug terkait: v0.5.4 memperbaiki tool history pada panggilan combo/fusion. — [releases](https://api.github.com/repos/decolua/9router/releases?per_page=8)

### Inferences
- Reprodusibilitas lemah bila combo dipakai (model penjawab bisa berganti diam-diam). Mitigasi: panggil id model tunggal, bukan combo, dan catat field `model` pada respons (apakah 9Router menuliskan model aktual di respons: tidak terkonfirmasi).

### Gaps
- Header/field model penjawab, kode error saat semua gagal, perilaku timeout: tidak terkonfirmasi (perlu baca kode atau uji).

## 4. Penyedia hulu dan risiko ToS

### Takeaway
Campuran: 40+ provider API key resmi berbayar, beberapa tier gratis yang sudah dihentikan, dan jalur langganan/OAuth konsumen (Claude Code, Codex, Copilot, Cursor, Antigravity, Kiro). Jalur langganan Claude jelas bertentangan dengan kebijakan Anthropic bila dipakai di luar Claude Code/Claude.ai. Untuk backend produksi, gunakan hanya jalur API key.

### Cited Findings
- OAuth/langganan: Claude Code (Pro/Max), Codex (Plus/Pro), Antigravity, GitHub Copilot, Cursor, Kimchi; gratis: Kiro (sekitar 50 kredit/bulan), OpenCode Free (tanpa auth), Vertex AI (kredit $300). — [README](https://github.com/decolua/9router)
- API key (40+): OpenRouter, OpenAI, Anthropic, Gemini, DeepSeek, Groq, xAI, Mistral, Perplexity, Together, Fireworks, Cerebras, Cohere, NVIDIA, SiliconFlow, dll., plus endpoint custom OpenAI/Anthropic-compatible. — [README](https://github.com/decolua/9router)
- Tier gratis iFlow, Qwen Code, Gemini CLI disebut sudah dihentikan (README: Qwen 2026-04-15, Gemini CLI 2026-06-18). Docs situs masih mencantumkan iFlow/Qwen sebagai gratis, kontradiksi (README lebih baru). — [README](https://raw.githubusercontent.com/decolua/9router/master/README.md); bertentangan dengan [docs](https://docs.9router.com)
- Situs docs mengakui ada "MITM bridge" yang mencegat lalu lintas IDE dan menyarankan memeriksa kebijakan tiap alat. — [docs](https://docs.9router.com)
- Kebijakan Anthropic: Jan 2026 pengamanan sisi server memblokir token OAuth langganan di luar Claude Code/Claude.ai; Feb 2026 Anthropic menyatakan penggunaan token OAuth Free/Pro/Max di produk/alat lain termasuk Agent SDK tidak diizinkan, pengembang harus pakai API key. Ada laporan pemblokiran akun (sumber sekunder, blog). Kebijakan sempat kabur setelah permintaan maaf staf Anthropic. — [Gigazine](https://gigazine.net/gsc_news/en/20260220-anthropic-third-party-block), [Falcao](https://falcao.org/posts/anthropic-claude-access-crackdown-ecosystem-fallout/); ringkasan hasil pencarian, dokumen ToS Anthropic primer tidak saya buka
- Tidak ditemukan sumber yang menyebut Anthropic/OpenAI/GitHub memberi sanksi spesifik kepada pengguna 9Router. — hasil pencarian 16 Okt 2026

### Inferences
- (Spekulasi) Jalur Copilot/Codex/Cursor/Kiro konsumen kemungkinan juga melanggar ToS masing-masing, tidak diverifikasi karena ToS primer tidak dibaca.
- Untuk dipakai FastAPI produksi, hanya pakai provider API key resmi; maka nilai tambah 9Router (menyatukan langganan) hilang, dan tersisa perannya sebagai proxy tambahan.

### Gaps
- ToS primer Anthropic/OpenAI/GitHub/Google tidak dibaca langsung.

## 5. Keamanan

### Takeaway
Rekam jejak keamanan buruk: 26 CVE terdaftar dalam ~6 bulan, banyak Critical (bypass auth, RCE, kebocoran kunci). Default tidak aman. Tidak ada SECURITY.md. Tidak layak diekspos ke internet; pada kenyataannya kemungkinan masih ada celah belum ditemukan.

### Cited Findings
- Default: INITIAL_PASSWORD=123456, REQUIRE_API_KEY=false, API_KEY_SECRET=`endpoint-proxy-api-key-secret`, MACHINE_ID_SALT=`endpoint-proxy-salt`, AUTH_COOKIE_SECURE=false; JWT_SECRET dibuat otomatis di ~/.9router/jwt-secret. ENABLE_REQUEST_LOGS=false. — [README](https://raw.githubusercontent.com/decolua/9router/master/README.md)
- Penyimpanan: SQLite di ${DATA_DIR}/db/data.sqlite (default ~/.9router) berisi provider, combo, key, setting, usage; README tidak menjelaskan enkripsi at-rest. ARCHITECTURE.md (kemungkinan usang) menyebut kunci/token di db.json. — [README](https://github.com/decolua/9router), [ARCHITECTURE.md](https://raw.githubusercontent.com/decolua/9router/master/docs/ARCHITECTURE.md); pertentangan db.json vs SQLite, README lebih baru
- Daftar 26 CVE (Release Alert, agregator; verifikasi NVD/GHSA disarankan): CVE-2026-46339 (CVSS 10, rute CLI/MCP tanpa proteksi, eksekusi perintah, 0.4.30-0.4.37); CVE-2026-63732 (9.9, password default + bypass Host + RCE MCP, 0.4.59); CVE-2026-55500 (9.9, ekspor/impor DB tanpa auth, <0.4.80); CVE-2026-49352 (9.8, JWT secret fallback hardcoded, 0.2.21-0.4.44); CVE-2026-59800 (9.8 sudo injection Tailscale); CVE-2026-59801 (9.8/9.3 v4, endpoint provider tanpa auth, <=0.4.41, advisory GHSA-vjc7-jrh9-9j86); CVE-2026-62327 (9.1, /api/usage/stats membocorkan API key plaintext); CVE-2026-62328 (log request membocorkan isi percakapan); beberapa bypass auth lewat spoof header Host/X-Forwarded-For/X-9r-Real-Ip; SSRF (CVE-2026-63313, 72860, 103530); terbaru CVE-2026-108593 (6.4, injeksi konfigurasi Hermes, 0.4.1-0.5.99) dan CVE-2026-103530 (7.3-7.5, SSRF search, <=0.5.55). — [Release Alert](https://releasealert.dev/github/decolua/9router/cves), [OffSeq CVE-2026-59801](https://radar.offseq.com/threat/cve-2026-59801-missing-authentication-for-critical-0c4298ad3568d21f)
- Repo tidak punya kebijakan keamanan/SECURITY.md; pelaporan lewat GitHub private advisory; 10 advisory pada halaman pertama (7-16 Juli 2026), badge Security 19. — [GitHub Security](https://github.com/decolua/9router/security)
- Data dicatat: usage (provider, model, token, waktu); log mendalam saat ENABLE_REQUEST_LOGS=true menyimpan header+body lengkap (berisi prompt); Cloud Sync ke 9router.com disebut "secure encrypted storage" tanpa detail; tidak ada pernyataan telemetri prompt default. — [ARCHITECTURE.md](https://raw.githubusercontent.com/decolua/9router/master/docs/ARCHITECTURE.md), [README](https://github.com/decolua/9router)

### Inferences
- Pola perbaikan berulang untuk kelas bug yang sama (bypass auth lewat header, SSRF) menunjukkan arsitektur auth lemah (kepercayaan pada loopback/header). Sebagian CVE tanpa versi perbaikan tercatat; apakah tambalan ada untuk semuanya tidak terkonfirmasi.

### Gaps
- Status patch tiap CVE tidak diverifikasi di NVD (NVD tidak memuat konten via fetch). Apakah ada CVE setelah 0.5.99 tidak diketahui. Versi terbaru di master Oktober 2026 tidak terkonfirmasi.

## 6. Kematangan

### Takeaway
Proyek sangat populer dan sangat aktif, tetapi muda (Januari 2026), dikembangkan terutama satu maintainer, berubah cepat, dan dirancang untuk alat coding.

### Cited Findings
- Bintang 30.861, fork 5.963, open_issues_count 2.544 (issue+PR), lisensi MIT, dibuat 5 Jan 2026, push terakhir 15 Okt 2026, tidak diarsipkan, branch default master. — [GitHub API](https://api.github.com/repos/decolua/9router)
- Halaman repo: 1,4k issue, 1,2k PR, 1.431 commit di master. — [README halaman repo](https://github.com/decolua/9router)
- Kontributor teratas: decolua 161, anuragg-saxenaa 43, Jordannst 15, dst (angka API kontributor, tidak cocok dengan 1.431 commit; kemungkinan hitungan terbatas). — [Contributors API](https://api.github.com/repos/decolua/9router/contributors?per_page=5&anon=0)
- Rilis terdaftar via API: v0.5.2 (17 Jun) hingga v0.5.35 (16 Jul 2026); API tidak menampilkan rilis setelah Juli walau commit berlanjut sampai Oktober (perilaku tag/rilis terkini tidak terkonfirmasi). — [releases](https://api.github.com/repos/decolua/9router/releases?per_page=8)
- Versi nol-titik (0.x), CVE menyebut versi sampai 0.5.99, jadi penomoran cepat. README: paket privat `9router-app`, dijalankan lewat sumber/Docker; tagline "Unlimited FREE AI coding", target alat coding (Claude Code, Codex, Cursor, Cline, Copilot); ini adalah port JavaScript dari CLIProxyAPI; ada fork OmniRoute. — [README](https://github.com/decolua/9router)
- Ukuran dokumentasi/rencana internal bertanggal Agustus-September 2026 menunjukkan pengembangan aktif. — [git tree](https://api.github.com/repos/decolua/9router/git/trees/master?recursive=1)

### Inferences
- Bukan abandonware; masalah yang sebaliknya: laju perubahan tinggi, backlog issue/PR sangat besar (PR perbaikan penting seperti structured output menumpuk terbuka), bus factor 1, dan breaking change/ketidakstabilan sangat mungkin.

### Gaps
- Jumlah maintainer pasti dan rilis terbaru Oktober 2026 tidak terkonfirmasi.

## 7. Hosting

### Takeaway
Ringan untuk dijalankan (Node 20+, Next.js, SQLite, Docker), tetapi dirancang sebagai aplikasi lokal; produksi memerlukan banyak pengerasan.

### Cited Findings
- Node.js 20+, Next.js 16, React 19, SQLite; port 20128 (Codex OAuth 1455, Headroom 8787, SearXNG 8888). Docker: image `decolua/9router` dan `ghcr.io/decolua/9router` (amd64, arm64), mount data ke /app/data dengan DATA_DIR=/app/data, bind 0.0.0.0, .env tidak masuk image. PM2 dan Cloudflare Workers disebut. HTTP_PROXY/HTTPS_PROXY didukung. — [README](https://raw.githubusercontent.com/decolua/9router/master/README.md)
- Set AUTH_COOKIE_SECURE=true di belakang HTTPS; README merekomendasikan REQUIRE_API_KEY=true bila diekspos internet. — [README](https://raw.githubusercontent.com/decolua/9router/master/README.md)

### Inferences
- Banyak CVE melibatkan kepercayaan pada header Host/X-Forwarded-For; reverse proxy yang salah konfigurasi dapat memicu bypass (CVE-2026-56675 loopback trust). Hanya boleh di jaringan privat/localhost.

### Gaps
- Kebutuhan memori, overhead latensi terlapor, dan panduan reverse proxy/PM2/Railway resmi tidak ditemukan (tidak terkonfirmasi).
