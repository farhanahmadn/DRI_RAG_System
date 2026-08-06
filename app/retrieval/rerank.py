"""app/retrieval/rerank.py — reranker, provider dipilih via `RERANK_PROVIDER`.

Provider ("RERANK_PROVIDER" di .env, default "local"; ganti provider = ganti env, kode di sini dan
pemanggilnya (retriever.py) TIDAK berubah — pola sama seperti `app/reasoning/llm_client.py`):
  - "local" — bge-reranker-v2-m3 (cross-encoder) via `transformers` LANGSUNG
    (AutoModelForSequenceClassification), bukan sentence-transformers/FlagEmbedding (menghindari
    rantai datasets->pyarrow yang crash di Windows). Berat (transformers+torch) — dipertahankan
    sebagai opsi rollback / self-host GPU nanti (bukan dihapus).
  - "jina" — Jina AI `/v1/rerank` (model `JINA_RERANK_MODEL`, default "jina-reranker-v3").

Kontrak dipertahankan sama di kedua provider: `[(index_passage, skor)]` terurut menurun, index
merujuk posisi di `passages` — dipanggil retriever.py atas kandidat hasil fusion apa adanya.

(Catatan: Voyage AI sempat diimplementasikan sebagai kandidat A/B kedua lalu DIHAPUS — lihat
riwayat git — karena tak jadi dibandingkan. Menambahkannya lagi = tambah cabang provider baru di
sini + embeddings.py, pola sama seperti "jina".)

PII: `passages`/`query` yang lewat modul ini HARUS SUDAH bersih dari field permohonan (nama
pemohon, NIK, koordinat presisi, application_number, dst) — lihat tests/test_pii_provider_safety.py.
"""

from __future__ import annotations

import os
from functools import lru_cache

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_MAX_LEN = int(os.getenv("RERANK_MAX_LEN", "512"))
_RERANK_TIMEOUT_S = float(os.getenv("RERANK_TIMEOUT_S", "30"))
_RERANK_MAX_RETRIES = int(os.getenv("RERANK_MAX_RETRIES", "2"))

_JINA_URL = "https://api.jina.ai/v1/rerank"

_VALID_PROVIDERS = ("local", "jina")


def _provider() -> str:
    return os.getenv("RERANK_PROVIDER", "local").strip().lower()


# --------------------------------------------------------------------- local (bge-reranker-v2-m3)
@lru_cache(maxsize=1)
def _load_local():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tok, model, device


def _rerank_local(query: str, passages: list[str], batch_size: int) -> list[tuple[int, float]]:
    import torch

    tok, model, device = _load_local()
    scores: list[float] = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        pairs = [[query, p] for p in batch]
        enc = tok(pairs, padding=True, truncation=True, max_length=_MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc, return_dict=True).logits.view(-1).float()
        scores.extend(logits.cpu().tolist())
    return list(enumerate(scores))


# --------------------------------------------------------------------- jina
def _rerank_jina(query: str, passages: list[str]) -> list[tuple[int, float]]:
    from app.retrieval import _provider_http as http

    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise RuntimeError("JINA_API_KEY tidak ditemukan di environment (RERANK_PROVIDER=jina).")
    model = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v3")
    body = {"model": model, "query": query, "documents": passages}
    data = http.post_json(
        "jina-rerank", _JINA_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json_body=body, timeout_s=_RERANK_TIMEOUT_S, max_retries=_RERANK_MAX_RETRIES,
    )
    return [(int(r["index"]), float(r["relevance_score"])) for r in data["results"]]


# --------------------------------------------------------------------- API publik
def rerank(query: str, passages: list[str], top_k: int | None = None,
           batch_size: int = 16) -> list[tuple[int, float]]:
    """Return [(index_passage, skor)] terurut menurun. index merujuk posisi di `passages`."""
    if not passages:
        return []
    provider = _provider()
    if provider == "local":
        scored = _rerank_local(query, passages, batch_size)
    elif provider == "jina":
        scored = _rerank_jina(query, passages)
    else:
        raise RuntimeError(
            f"RERANK_PROVIDER tidak dikenal: {provider!r} (pilihan: {', '.join(_VALID_PROVIDERS)})."
        )

    order = sorted(scored, key=lambda t: t[1], reverse=True)
    if top_k is not None:
        order = order[:top_k]
    return [(j, float(s)) for j, s in order]
