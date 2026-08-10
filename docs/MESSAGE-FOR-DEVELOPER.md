# Message for DRISleman-ai Developer — 7 Aug 2026

## ✅ Vector DB SUDAH terisi

Di VPS1 (user `ml`):
- 484 chunks ter-ingest, 440 ter-embed (jina API, no torch, hemat RAM)
- 3 dokumen + tabel lampiran V.B/VI
- tabel `chunks` (baseline + dense 1024-dim), `chunk_embeddings_ab` siap buat A/B Jina, `dokumen`, `matriks_kegiatan`, `intensitas_zona`

PM2 `4005-drisman-ai` online, RAM 64MB, disk VPS 94.7%. Ingest tidak restart app.

## GitHub Action auto-deploy — jawaban: **deploy key SSH, BUKAN PAT**

**Soal PAT**: Boleh saja pakai PAT GitHub, tapi untuk clone/update pakai PAT lebih lemah daripada deploy key karena:
- PAT = akses **user/org** (boros permission, hangus kalau rotate)
- Deploy key = akses **repo spesifik saja** (least-privilege, hanya bisa push repo itu)
- PAT bisa clone repo lain yang sama org-nya (blast radius lebih besar)

Deploy key perlu:
```bash
# Di VPS user ml:
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "ci-drisman-ai"
cat ~/.ssh/id_ed25519.pub   # -> Deploy Keys di GitHub repo
```

Private key (`~/.ssh/id_ed25519`) → simpan di GitHub Secret `ML_SSH_KEY` di repo.

Workflow (sudah disiapkan):
- Trigger: push ke `main`
- rsync source ke VPS `~/project-2026-07-DRISleman-ai/` (exclude `.env` biar credential aman)
- `pip install -e ".[rag]"` + `pm2 restart 4005-drisman-ai --update-env`

**Kekurangan PAT vs SSH key**: kalau pakai PAT untuk `git fetch/pull` di CI, perlu:
```yaml
- uses: actions/checkout@v4
  with:
    token: ${{ secrets.GH_PAT }}
```
PAT juga perlu scope `repo` (read cukup). Tapi tetap pakai deploy key lebih aman.

## A→B Vector DB & Reasoning flow — penjelasan

Ini alur berpikir retrieval + reasoning, biar jelas per langkah:

```
[L2 payload dari back-end]
     |
     v
[app/api/main.py:36 POST /reasoning] -> [jalankan_precheck() in app/reasoning/assemble.py]
     |
     v
[1. RETRIEVAL — ambil chunks relevan dari pgvector]
   |
   v
   a) Embed QUERY via encode_dense([query_text], input_type="query")     ← jina API
      -> vektor 1024-dim (normalised L2 ≈ 1.0)
   |
   v
   b) ANN search via HNSW index `chunks_embedding_hnsw` di tabel `chunks`
      SQL: SELECT id, teks, ... FROM chunks
           WHERE dokumen_id ILIKE '%Sleman%'  -- filter wilayah
           ORDER BY embedding <=> $1 LIMIT 50
      -> top-50 chunk dari prospektif peraturan (pasal/ayat/tabel V.B/VI)
   |
   v
   c) Re-rank via jina-reranker (jina-reranker-v3)
      input: (query, chunk.teks) pairs
      -> skor ulang, sort descending
      -> top-K = 8 chunk final (lihat app/retrieval/retriever.py)
     |
     v
[2. PROMPT ASSEMBLY — bangun prompt dari chunk + L2 facts]
   - allowlist field `poin.fakta` (filter fakta yang masuk ke LLM — lihat app/sanitize.py)
   - system prompt di app/reasoning/prompts.py (instruksi: "berdasar chunk di bawah, jawab…")
     |
     v
[3. LLM CALL — Groq Llama 3.3 70B Versatile]
   - app/reasoning/llm_client.py: timeout/retry/transport hardening
   - response: reasoning_pendek (≤250 char) + reasoning_panjang + rekomendasi + dasar_hukum + sitasi
     |
     v
[4. GUARDRAIL — validasi struktural sebelum kembalikan]
   app/reasoning/guardrail.py:
     - Cek #1-#5: field wajib ada, sitasi sesuai teks chunk, status konsisten, dsb
     - kalau gagal → repair/retry/degrade ke fallback deterministik
     |
     v
[OutputL3 JSON] -> dikirim balik ke back-end
```

**Kenapa A→B (A: lokal bge-m3 vs B: Jina API):**
- `chunks.embedding` (kolom utama) = baseline **A** (bge-m3 / 1024-dim) — kompatibel tanpa ubah model
- `chunk_embeddings_ab` = tabel terpisah, tempat simpan **B** (Jina API, serverless, dimensi sama 1024) → A/B compare per query
- Jalur produksi selalu pakai A (`chunks.embedding`); B hanya dipakai eksperimen

## Test end-to-end setelah deploy

Cek cepat di VPS (user `ml`):
```bash
curl -s http://localhost:4005/health
# {"status":"ok"}

# /reasoning pakai fixture sample_lolos (28KB):
curl -s -X POST http://localhost:4005/reasoning \
  -H 'Content-Type: application/json' \
  --data-binary @~/project-2026-07-DRISleman-ai/tests/fixtures/l2_sample_lolos.json \
  | python3 -m json.tool | head -60
```

Rate limit: 10/menit per IP. Kalau IP-nya VPS sendiri, gampang kena 429 → kirim header `X-Forwarded-For: 127.0.0.1` tidak ngefek (retriever pakai `request.client.host`).

## Langkah selanjutnya (urutan)

1. ☐ Generate deploy key SSH di VPS ml → add ke GitHub Deploy Keys
2. ☐ Push 3 secrets ke GitHub repo (`VPS1_HOST`, `ML_USER`, `ML_SSH_KEY`)
3. ☐ Buat file `.github/workflows/deploy.yml` di repo (sudah disiapkan draft-nya)
4. ☐ Test /reasoning pakai fixture (di atas)
5. ☐ Setup NPM + Cloudflare untuk expose publik (subdomain `<SUB>.urbansolv.co.id`)
