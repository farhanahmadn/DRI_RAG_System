"""app/retrieval/fusion.py — Reciprocal Rank Fusion (RRF).

Gabung beberapa daftar peringkat (mis. dense + FTS) tanpa perlu menormalkan skala skor yang berbeda.
RRF: skor(id) = sum_r 1/(k + rank_r(id)), rank mulai 1. k=60 default (nilai standar literatur).
"""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """rankings = daftar list id (terurut relevansi menurun). Return [(id, skor_rrf)] terurut menurun."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
