"""app/retrieval/rerank.py — reranker bge-reranker-v2-m3 (cross-encoder) via transformers LANGSUNG.

Sama seperti embeddings.py: pakai transformers (AutoModelForSequenceClassification) alih-alih
sentence-transformers/FlagEmbedding (menghindari rantai datasets->pyarrow yang crash di Windows).
Cross-encoder menilai relevansi pasangan (query, passage) -> satu skor logit; makin tinggi makin relevan.
Dipanggil retriever.py atas kandidat hasil fusion untuk memilih top-k final.
"""

from __future__ import annotations

import os
from functools import lru_cache

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_MAX_LEN = int(os.getenv("RERANK_MAX_LEN", "512"))


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tok, model, device


def rerank(query: str, passages: list[str], top_k: int | None = None,
           batch_size: int = 16) -> list[tuple[int, float]]:
    """Return [(index_passage, skor)] terurut menurun. index merujuk posisi di `passages`."""
    if not passages:
        return []
    import torch

    tok, model, device = _load()
    scores: list[float] = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        pairs = [[query, p] for p in batch]
        enc = tok(pairs, padding=True, truncation=True, max_length=_MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc, return_dict=True).logits.view(-1).float()
        scores.extend(logits.cpu().tolist())
    order = sorted(range(len(passages)), key=lambda j: scores[j], reverse=True)
    if top_k is not None:
        order = order[:top_k]
    return [(j, float(scores[j])) for j in order]
