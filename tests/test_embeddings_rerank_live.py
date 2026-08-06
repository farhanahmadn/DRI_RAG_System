"""tests/test_embeddings_rerank_live.py — smoke test PROVIDER SERVERLESS sungguhan (Jina), di bawah
`@pytest.mark.live` (butuh JINA_API_KEY, tak jalan di `pytest -q -m "not live"` / CI rutin — pola
sama seperti tests/test_generator.py § Live).

Menguji `app/retrieval/embeddings.py`/`rerank.py` end-to-end thd API Jina sungguhan (bukan mock) —
dim vektor, arah relevansi rerank, dan konsistensi query<->document (WAJIB sama model/fungsi, lihat
docstring embeddings.py). TIDAK menyentuh DB (fokus provider HTTP call, bukan retrieval penuh —
untuk itu lihat tests/test_retrieval.py yang butuh DATABASE_URL+korpus ter-ingest).
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

_ALASAN_SKIP = "JINA_API_KEY tidak diset — skip smoke test provider Jina sungguhan."

pytestmark = [pytest.mark.live, pytest.mark.skipif(not os.getenv("JINA_API_KEY"), reason=_ALASAN_SKIP)]


@pytest.fixture(autouse=True)
def _paksa_provider_jina(monkeypatch):
    # Independen dari EMBEDDING_PROVIDER/RERANK_PROVIDER ambien (apa pun default saat ini di .env) —
    # test ini KHUSUS menguji jalur Jina, bukan "provider aktif sekarang".
    monkeypatch.setenv("EMBEDDING_PROVIDER", "jina")
    monkeypatch.setenv("RERANK_PROVIDER", "jina")


def test_encode_dense_jina_dim_dan_konsistensi():
    from app.retrieval.embeddings import DIM, encode_dense, encode_dense_one

    vecs = encode_dense(
        ["Koefisien Dasar Bangunan adalah angka persentase luas lantai dasar terhadap luas lahan.",
         "Koefisien Lantai Bangunan adalah angka perbandingan luas lantai bangunan."],
        input_type="document",
    )
    assert len(vecs) == 2
    assert all(len(v) == DIM == 1024 for v in vecs)
    assert vecs[0] != vecs[1]  # teks beda -> vektor beda (bukan placeholder/nol)

    qvec = encode_dense_one("KDB Koefisien Dasar Bangunan")
    assert len(qvec) == DIM


def test_encode_dense_jina_query_lebih_dekat_ke_dokumen_relevan():
    """Sanity check kualitas embedding: query KDB harus lebih mirip (cosine) ke passage KDB
    dibanding passage KLB — bukan cuma "API merespons", tapi hasilnya masuk akal secara semantik."""
    import math

    from app.retrieval.embeddings import encode_dense, encode_dense_one

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb)

    passages = [
        "Koefisien Dasar Bangunan (KDB) adalah angka persentase perbandingan luas lantai dasar "
        "bangunan dengan luas lahan/perpetakan.",
        "Koefisien Lantai Bangunan (KLB) adalah angka perbandingan jumlah luas lantai bangunan "
        "dengan luas lahan/perpetakan.",
    ]
    doc_vecs = encode_dense(passages, input_type="document")
    qvec = encode_dense_one("KDB Koefisien Dasar Bangunan")

    sim_kdb = cosine(qvec, doc_vecs[0])
    sim_klb = cosine(qvec, doc_vecs[1])
    assert sim_kdb > sim_klb, f"query KDB harus lebih dekat ke passage KDB (sim={sim_kdb:.4f}) drpd KLB (sim={sim_klb:.4f})"


def test_rerank_jina_urutkan_passage_relevan_lebih_tinggi():
    from app.retrieval.rerank import rerank

    query = "KDB Koefisien Dasar Bangunan"
    passages = [
        "Koefisien Lantai Bangunan (KLB) adalah angka perbandingan jumlah luas lantai bangunan.",
        "Koefisien Dasar Bangunan (KDB) adalah angka persentase perbandingan luas lantai dasar "
        "bangunan dengan luas lahan/perpetakan.",
    ]
    scored = rerank(query, passages, top_k=2)

    assert len(scored) == 2
    top_idx, top_score = scored[0]
    assert top_idx == 1, "passage KDB (index 1) seharusnya rank teratas utk query KDB"
    assert top_score >= scored[1][1]


def test_rerank_jina_top_k_membatasi_hasil():
    from app.retrieval.rerank import rerank

    passages = [f"Pasal {i} tentang ketentuan zonasi." for i in range(5)]
    scored = rerank("ketentuan zonasi", passages, top_k=2)
    assert len(scored) == 2
