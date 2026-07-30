"""tests/test_retrieval.py — GERBANG retrieval: Hit-Rate@k + MRR + exact-match get_by_reference.

Prinsip rencana: "Uji retrieval DULU sebelum apa pun" & "uji sebelum deklarasi selesai". Menguji
retriever ASLI (bge-m3 + FTS + RRF + reranker) atas DB Timur — jadi butuh DATABASE_URL + model
lokal. Otomatis SKIP kalau DB tak tersedia (mirip live-test reasoning yg skip tanpa GROQ_API_KEY),
supaya tak membebani `pytest` cepat/CI.

Jalankan sebagai laporan metrik:  python -m tests.test_retrieval
Atau via pytest (butuh DB + model):  pytest tests/test_retrieval.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

try:  # pytest opsional: laporan CLI (python -m tests.test_retrieval) tak membutuhkannya
    import pytest
except ImportError:
    pytest = None

_EVAL = Path(__file__).parent / "eval_set.jsonl"
_TOP_K = 5
_HIT_RATE_MIN = 0.80   # ambang gerbang (dev-seeded; validasi ahli menyusul)
_MRR_MIN = 0.60

# get_by_reference: rujukan -> pasal yang WAJIB terambil (exact-match sitasi)
_REF_CASES = [
    ("RDTR Sleman Tengah Pasal 43", "43"),
    ("RDTR Kawasan Sleman Tengah Pasal 49", "49"),
    ("Perbup Sleman 80/2023 Pasal 1", "1"),
]

_db_ready = bool(os.getenv("DATABASE_URL"))
if pytest is not None:
    pytestmark = pytest.mark.skipif(not _db_ready, reason="DATABASE_URL tak diset — butuh DB+model retrieval")


def _load_eval() -> list[dict]:
    return [json.loads(l) for l in _EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]


def _retriever():
    from app.retrieval.retriever import RetrieverAsli

    return RetrieverAsli(default_wilayah=os.getenv("DEMO_WILAYAH", "Sleman Tengah"))


def evaluate() -> dict:
    """Hitung Hit-Rate@k & MRR atas eval set. Return dict metrik + detail per query."""
    from app.retrieval.base import RetrievalFilters

    rt = _retriever()
    evalset = _load_eval()
    hits, rrs, detail = 0, 0.0, []
    for row in evalset:
        got = [c.id for c in rt.search(row["query"], RetrievalFilters(), top_k=_TOP_K)]
        rel = set(row["relevan"])
        rank = next((i + 1 for i, cid in enumerate(got) if cid in rel), None)
        hit = rank is not None
        hits += 1 if hit else 0
        rrs += (1.0 / rank) if rank else 0.0
        detail.append({"query": row["query"], "hit": hit, "rank": rank, "top": got})
    n = len(evalset)
    return {"n": n, "hit_rate": hits / n, "mrr": rrs / n, "detail": detail}


def exact_match_reference() -> dict:
    rt = _retriever()
    ok, total, detail = 0, len(_REF_CASES), []
    for ref, expect_pasal in _REF_CASES:
        chunks = rt.get_by_reference([ref])
        good = len(chunks) > 0 and all(c.pasal == expect_pasal for c in chunks)
        ok += 1 if good else 0
        detail.append({"ref": ref, "expect_pasal": expect_pasal, "n": len(chunks), "ok": good})
    return {"rate": ok / total, "detail": detail}


# --------------------------------------------------------------------- pytest
def test_hit_rate_and_mrr():
    m = evaluate()
    assert m["hit_rate"] >= _HIT_RATE_MIN, f"Hit-Rate@{_TOP_K}={m['hit_rate']:.2f} < {_HIT_RATE_MIN}"
    assert m["mrr"] >= _MRR_MIN, f"MRR={m['mrr']:.2f} < {_MRR_MIN}"


def test_get_by_reference_exact():
    r = exact_match_reference()
    assert r["rate"] == 1.0, f"exact-match get_by_reference={r['rate']:.2f}: {r['detail']}"


def test_get_parent_small_to_big():
    rt = _retriever()
    ref = rt.get_by_reference(["RDTR Sleman Timur Pasal 44"])
    ayat = next((c for c in ref if c.level == "ayat"), None)
    assert ayat is not None, "tak ada chunk ayat utk Pasal 44"
    parent = rt.get_parent(ayat.id)
    assert parent is not None and parent.level == "pasal" and parent.id == ayat.parent_id


# ----------------------------------------------------------------- report CLI
def _main() -> None:
    m = evaluate()
    print(f"\n=== Retrieval Eval ({os.getenv('DEMO_WILAYAH', 'Sleman Tengah')}) — n={m['n']} ===")
    print(f"Hit-Rate@{_TOP_K}: {m['hit_rate']:.2%}   MRR: {m['mrr']:.3f}")
    print("-" * 74)
    for d in m["detail"]:
        mark = "OK " if d["hit"] else "!! "
        print(f"  {mark} rank={d['rank'] or '-':<3} {d['query'][:48]}")
    r = exact_match_reference()
    print("-" * 74)
    print(f"exact-match get_by_reference: {r['rate']:.0%}")
    for d in r["detail"]:
        print(f"  {'OK ' if d['ok'] else '!! '} {d['ref']} -> {d['n']} chunk (pasal {d['expect_pasal']})")
    gate = m["hit_rate"] >= _HIT_RATE_MIN and m["mrr"] >= _MRR_MIN and r["rate"] == 1.0
    print("\nGERBANG:", "LULUS ✅" if gate else "BELUM ❌",
          f"(syarat: Hit-Rate≥{_HIT_RATE_MIN:.0%}, MRR≥{_MRR_MIN:.2f}, exact-match=100%)")


if __name__ == "__main__":
    _main()
