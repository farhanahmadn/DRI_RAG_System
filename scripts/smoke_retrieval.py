"""Smoke test retrieval NYATA (bge-m3 + bge-reranker-v2-m3 asli) atas DB Timur.

Jalankan dari root repo:  python -m scripts.smoke_retrieval
Catatan: pemanggilan pertama mengunduh reranker bge-reranker-v2-m3 (~2,3 GB, sekali).
"""

import os

from app.retrieval.base import RetrievalFilters
from app.retrieval.retriever import RetrieverAsli

WILAYAH = os.getenv("DEMO_WILAYAH", "Sleman Tengah")  # ganti daerah tanpa ubah kode
QUERIES = ["Lokasional LP2B", "Sempadan Sungai", "Resapan", "KDB", "KLB", "Kegiatan", "Banjir"]


def main() -> None:
    rt = RetrieverAsli(default_wilayah=WILAYAH)  # filter otomatis ke RDTR daerah demo
    print(f"(demo wilayah: {WILAYAH})")

    for q in QUERIES:
        print("=" * 78)
        print(f"QUERY: {q}")
        hits = rt.search(q, RetrievalFilters(), top_k=3)
        if not hits:
            print("  (tidak ada hasil)")
        for c in hits:
            teks = c.teks.replace("\n", " ")[:100]
            print(f"  [{c.skor:6.2f}] {c.id}  (Pasal {c.pasal}, hal {c.halaman})")
            print(f"           {teks}...")

    print("\n" + "=" * 78)
    ref_str = f"RDTR {WILAYAH} Pasal 49"
    ref = rt.get_by_reference([ref_str])
    print(f"get_by_reference({ref_str!r}):", [c.id for c in ref][:6])
    if ref:
        ayat = next((c for c in ref if c.level == "ayat"), None)
        if ayat:
            par = rt.get_parent(ayat.id)
            print(f"get_parent({ayat.id}) -> {par.id if par else None} (small-to-big)")


if __name__ == "__main__":
    main()
