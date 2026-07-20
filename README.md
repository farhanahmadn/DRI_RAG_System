# RDTR Sleman — AI Reasoning (bagian saya)

Komponen AI Reasoning untuk pre-check risiko izin bangunan (Sleman): menerima jejak aturan
rule-based, menghasilkan reasoning, sitasi terverifikasi, rekomendasi, dan kesimpulan sebagai
JSON. Lihat `CLAUDE.md` untuk konteks arsitektur lengkap dan pembagian kerja.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -e ".[dev]"

cp .env.example .env          # lalu isi GROQ_API_KEY dengan key asli
```

## Run

API belum ada (`app/api/` masih placeholder — menyusul di task implementasi berikutnya).
Setelah `app/api/main.py` dibuat:

```bash
uvicorn app.api.main:app --reload
```

## Test

```bash
pytest
```

## Struktur

- `app/schemas.py` — skema Pydantic (jejak aturan & output JSON)
- `app/reasoning/` — llm_client, calculator, templates, generator, guardrail, assemble
- `app/retrieval/` — kontrak `Retriever`/`Chunk` (`base.py`) + `MockRetriever` (`mock.py`) dipakai sampai RAG asli (milik teman) siap
- `app/api/` — endpoint FastAPI
- `app/ingest/` — **milik teman** (parsing/chunking/embedding), tidak disentuh dari sisi ini
- `tests/`, `eval/` — unit test & eval harness
