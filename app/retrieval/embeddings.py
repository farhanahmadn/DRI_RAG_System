"""app/retrieval/embeddings.py — SATU sumber kebenaran embedding, provider dipilih via `EMBEDDING_PROVIDER`.

WAJIB: query & chunk di-embed model & fungsi yang SAMA (CLAUDE.md § Embedding). Dipakai ingest.py
(chunk, `input_type="document"`) DAN retriever.py (query, `input_type="query"` lewat
`encode_dense_one`).

Provider ("EMBEDDING_PROVIDER" di .env, default "local"; ganti provider = ganti env, kode di sini
dan pemanggilnya TIDAK berubah — pola sama seperti `app/reasoning/llm_client.py`):
  - "local" — bge-m3 via `transformers` LANGSUNG (AutoTokenizer+AutoModel), BUKAN
    sentence-transformers/FlagEmbedding. Alasan (dibuktikan via faulthandler di Windows/Python 3.13):
    sentence-transformers memuat `datasets` -> `pyarrow` saat import, dan pyarrow crash (access
    violation) di lingkungan ini. Dense bge-m3 = CLS token last_hidden_state, dinormalisasi L2 ->
    cosine (metode dense resmi bge-m3, berbasis XLM-RoBERTa). Butuh `pip install -e ".[rag]"`
    (transformers+torch) — berat, tidak cocok VPS produksi ber-memori terbatas; dipertahankan
    sebagai opsi rollback / self-host GPU nanti (bukan dihapus).
  - "jina" — Jina AI `/v1/embeddings` (model `JINA_EMBEDDING_MODEL`, default "jina-embeddings-v3"),
    `task=retrieval.passage|retrieval.query` sesuai sisi (index/search).

Kedua provider dikonfigurasi 1024-dim (`EMBEDDING_DIM` di .env) sehingga vektornya bisa hidup
berdampingan di kolom `chunks.embedding` yang sama untuk A/B, ditandai `embedding_provider`/
`embedding_model` (lihat sql/schema.sql & scripts/reembed.py) — tidak saling menimpa baseline bge-m3
yang sudah lulus gerbang retrieval.

(Catatan: Voyage AI sempat diimplementasikan sebagai kandidat A/B kedua lalu DIHAPUS — lihat riwayat
git — karena tak jadi dibandingkan. Menambahkannya lagi = tambah cabang provider baru di sini +
rerank.py, pola sama seperti "jina".)

PII: teks yang lewat modul ini HARUS SUDAH bersih dari field permohonan (nama pemohon, NIK,
koordinat presisi, application_number, dst) — sumbernya `teks_prefixed` (regulasi, dari
app/ingest/*) saat index, atau nama kategori indikator pendek (mis. "KDB") saat query
(app/retrieval/retriever.py). Modul ini TIDAK melakukan sanitasi sendiri; lihat
tests/test_pii_provider_safety.py untuk jaminan di titik pemanggilan.
"""

from __future__ import annotations

import os
from functools import lru_cache

# Windows: cegah abort konflik OpenMP (torch + lib native lain). Set sebelum torch dimuat.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
_MAX_LEN = int(os.getenv("EMBED_MAX_LEN", "1024"))  # chunk pasal/ayat pendek; 1024 aman & cepat (provider local)

_EMBED_TIMEOUT_S = float(os.getenv("EMBED_TIMEOUT_S", "30"))
_EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "2"))

_JINA_URL = "https://api.jina.ai/v1/embeddings"

_VALID_PROVIDERS = ("local", "jina")


def _provider() -> str:
    return os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()


def current_model_name(provider: str | None = None) -> str:
    """Nama model utk `provider` (default: EMBEDDING_PROVIDER aktif) — dipakai untuk tagging
    `chunks.embedding_model`/`chunk_embeddings_ab.embedding_model` (scripts/reembed.py, ingest.py)."""
    provider = provider or _provider()
    if provider == "local":
        return os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    if provider == "jina":
        return os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
    raise RuntimeError(
        f"EMBEDDING_PROVIDER tidak dikenal: {provider!r} (pilihan: {', '.join(_VALID_PROVIDERS)})."
    )


# --------------------------------------------------------------------- local (bge-m3)
@lru_cache(maxsize=1)
def _load_local():
    import torch
    from transformers import AutoModel, AutoTokenizer  # lazy; TIDAK menarik datasets/pyarrow

    name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tok, model, device


def _encode_dense_local(texts: list[str], batch_size: int) -> list[list[float]]:
    import torch

    tok, model, device = _load_local()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=_MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state          # [B, T, 1024]
        cls = hidden[:, 0]                                     # CLS token = dense bge-m3
        cls = torch.nn.functional.normalize(cls, p=2, dim=1)  # L2 -> cosine
        out.extend(cls.cpu().tolist())
    return [[float(x) for x in v] for v in out]


# --------------------------------------------------------------------- jina
def _encode_dense_jina(texts: list[str], input_type: str) -> list[list[float]]:
    from app.retrieval import _provider_http as http

    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise RuntimeError("JINA_API_KEY tidak ditemukan di environment (EMBEDDING_PROVIDER=jina).")
    model = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
    task = "retrieval.passage" if input_type == "document" else "retrieval.query"
    body = {"model": model, "task": task, "input": texts}
    data = http.post_json(
        "jina-embed", _JINA_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json_body=body, timeout_s=_EMBED_TIMEOUT_S, max_retries=_EMBED_MAX_RETRIES,
    )
    rows = sorted(data["data"], key=lambda r: r["index"])
    return [r["embedding"] for r in rows]


# --------------------------------------------------------------------- API publik
def encode_dense(texts: list[str], batch_size: int = 16, input_type: str = "document") -> list[list[float]]:
    """Vektor dense (list of list), dinormalisasi L2 (cosine). Untuk chunk (index) & query (search).

    `input_type`: "document" saat index (default; dipakai ingest.py), "query" saat search (dipakai
    `encode_dense_one`). Provider "local" (bge-m3) tidak membedakan keduanya (simetris); "jina"
    memakainya untuk instruction-tuned embedding asimetris.
    """
    if not texts:
        return []
    provider = _provider()
    if provider == "local":
        vecs = _encode_dense_local(texts, batch_size)
    elif provider == "jina":
        vecs = []
        for i in range(0, len(texts), batch_size):
            vecs.extend(_encode_dense_jina(texts[i:i + batch_size], input_type))
    else:
        raise RuntimeError(
            f"EMBEDDING_PROVIDER tidak dikenal: {provider!r} (pilihan: {', '.join(_VALID_PROVIDERS)})."
        )

    for v in vecs:
        if len(v) != DIM:
            raise RuntimeError(
                f"Provider embedding '{provider}' mengembalikan dim={len(v)}, diharapkan {DIM} "
                f"(EMBEDDING_DIM di .env) — cek model/konfigurasi provider."
            )
    return vecs


def encode_dense_one(text: str, input_type: str = "query") -> list[float]:
    """Vektor dense untuk satu query (default `input_type='query'`)."""
    return encode_dense([text], input_type=input_type)[0]
