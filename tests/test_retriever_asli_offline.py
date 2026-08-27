"""tests/test_retriever_asli_offline.py — bagian RetrieverAsli yang TIDAK butuh koneksi DB
sungguhan (murni in-memory: cache). Beda dari tests/test_retrieval.py (live, butuh DB+model
retrieval sungguhan) — constructor RetrieverAsli tidak connect DB secara eager, jadi aman
diinstansiasi di sini tanpa DATABASE_URL."""

from app.retrieval.retriever import RetrieverAsli


class TestClearCache:
    def test_mengosongkan_entri_yang_ada(self):
        r = RetrieverAsli()
        r._cache._store["dummy-key"] = (0.0, ["dummy-chunk"])
        assert r._cache.stats()["n_entries"] == 1

        r.clear_cache()

        assert r._cache.stats()["n_entries"] == 0

    def test_aman_dipanggil_saat_cache_masih_kosong(self):
        r = RetrieverAsli()
        r.clear_cache()  # tidak boleh raise walau belum pernah diisi
        assert r._cache.stats()["n_entries"] == 0

    def test_tak_memengaruhi_instance_retriever_lain(self):
        # Cache per-instance (bukan singleton modul) — clear_cache() satu instance TIDAK BOLEH
        # menyentuh instance RetrieverAsli lain (mis. candidate_k/rerank_pool berbeda).
        r1 = RetrieverAsli()
        r2 = RetrieverAsli()
        r1._cache._store["k1"] = (0.0, ["a"])
        r2._cache._store["k2"] = (0.0, ["b"])

        r1.clear_cache()

        assert r1._cache.stats()["n_entries"] == 0
        assert r2._cache.stats()["n_entries"] == 1
