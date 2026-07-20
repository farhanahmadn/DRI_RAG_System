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

```bash
uvicorn app.api.main:app --reload
```

- `GET /health` — liveness check (`{"status": "ok"}`).
- `POST /reasoning` — jalankan precheck (lihat contoh di bawah).
- `/docs` — Swagger UI (OpenAPI) bawaan FastAPI, bisa langsung coba endpoint dari browser.
- `/redoc` — dokumentasi API alternatif.

Retriever memakai `MockRetriever` (data dummy) secara default lewat dependency injection
(`app/api/dependencies.py::get_retriever`) — swap ke RAG asli (punya teman) nanti tidak perlu
mengubah endpoint.

## Contoh Request

```bash
curl -X POST http://127.0.0.1:8000/reasoning \
  -H "Content-Type: application/json" \
  -d '{
    "skor_total": 20.0,
    "level": "Tinggi",
    "zona": "LP2B",
    "indikator": [
      {
        "poin_id": "LP2B-01",
        "kategori": "Lokasional LP2B",
        "bobot": 20.0,
        "skor": 100.0,
        "kontribusi": 20.0,
        "nilai_input": "dalam_lp2b",
        "ambang": "tidak_dalam_lp2b",
        "operator": "==",
        "formula": "in_lp2b == True",
        "zona": "LP2B",
        "referensi_hukum": ["UU No. 41 Tahun 2009 Pasal 44"],
        "fakta_spasial": {"in_lp2b": true, "banjir": false, "resapan": false}
      },
      {
        "poin_id": "KDB-01",
        "kategori": "KDB",
        "bobot": 10.0,
        "skor": 0.0,
        "kontribusi": 0.0,
        "nilai_input": 0.4,
        "ambang": 0.6,
        "operator": "<=",
        "formula": "kdb_aktual <= kdb_maks"
      }
    ]
  }'
```

Respons: JSON `OutputPreCheck` (`ringkasan`, `poin[]`, `kesimpulan`) — lihat `app/schemas.py` untuk
bentuk lengkap.

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
