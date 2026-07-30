"""app/retrieval/embeddings.py — SATU sumber kebenaran embedding bge-m3 (dense).

WAJIB: query & chunk di-embed model & fungsi yang SAMA (CLAUDE.md § Embedding). Dipakai ingest.py
(chunk) DAN retriever.py (query).

Backend: `transformers` LANGSUNG (AutoTokenizer + AutoModel), BUKAN sentence-transformers/FlagEmbedding.
Alasan (dibuktikan via faulthandler di Windows/Python 3.13): sentence-transformers memuat `datasets`
-> `pyarrow` saat import, dan pyarrow crash (access violation) di lingkungan ini. Kita hanya inferensi,
jadi seluruh rantai training itu tak diperlukan. Dense bge-m3 = CLS token last_hidden_state,
dinormalisasi L2 -> cosine (metode dense resmi bge-m3, berbasis XLM-RoBERTa). Konsisten query<->chunk
karena keduanya lewat fungsi ini.
"""

from __future__ import annotations

import os
from functools import lru_cache

# Windows: cegah abort konflik OpenMP (torch + lib native lain). Set sebelum torch dimuat.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DIM = 1024
_MAX_LEN = int(os.getenv("EMBED_MAX_LEN", "1024"))  # chunk pasal/ayat pendek; 1024 aman & cepat


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModel, AutoTokenizer  # lazy; TIDAK menarik datasets/pyarrow

    name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tok, model, device


def encode_dense(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Vektor dense 1024-dim (list of list), dinormalisasi L2 (cosine). Untuk chunk & query."""
    import torch

    tok, model, device = _load()
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


def encode_dense_one(text: str) -> list[float]:
    """Vektor dense untuk satu query."""
    return encode_dense([text])[0]
