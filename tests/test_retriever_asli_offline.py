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


class TestPecahSeriWilayahGetByReference:
    """Rujukan generik back-end ("RDTR Sleman ...") kehilangan seluruh token pembeda di `db._tok`
    (stopword membuang "rdtr"/"kawasan"/"peraturan"/dst), menyisakan "sleman" yang cocok ke SEMUA
    dokumen RDTR wilayah dgn skor seri — terukur 2026-09-07: `get_by_reference` mengembalikan 1172
    chunk lintas Tengah+Timur. `search()` sudah berfilter wilayah, `get_by_reference()` belum."""

    @staticmethod
    def _patch_match_dokumen(monkeypatch, hasil_utk_wilayah):
        from app.retrieval import db

        monkeypatch.setattr(db, "match_dokumen", lambda conn, tokens: list(hasil_utk_wilayah))

    def test_rujukan_ambigu_disaring_ke_wilayah_default(self, monkeypatch):
        self._patch_match_dokumen(monkeypatch, ["rdtr-sleman-tengah"])
        r = RetrieverAsli(default_wilayah="Sleman Tengah")

        hasil = r._pecah_seri_wilayah(
            None, ["rdtr-sleman-barat", "rdtr-sleman-tengah", "rdtr-sleman-timur"]
        )

        assert hasil == ["rdtr-sleman-tengah"]

    def test_rujukan_spesifik_satu_dokumen_tidak_disentuh(self, monkeypatch):
        # Mis. "UU No. 41 Tahun 2009" — dokumen lintas-wilayah, harus tetap bisa dirujuk apa adanya.
        self._patch_match_dokumen(monkeypatch, ["rdtr-sleman-tengah"])
        r = RetrieverAsli(default_wilayah="Sleman Tengah")

        assert r._pecah_seri_wilayah(None, ["uu-41-2009"]) == ["uu-41-2009"]

    def test_wilayah_default_tak_termasuk_yang_cocok_kembalikan_utuh(self, monkeypatch):
        # Jangan mengosongkan hasil berdasar asumsi — kalau wilayah default tak ada di kandidat,
        # biarkan pemanggil memutuskan, bukan diam-diam menghapus semuanya.
        self._patch_match_dokumen(monkeypatch, ["rdtr-sleman-tengah"])
        r = RetrieverAsli(default_wilayah="Sleman Tengah")

        hasil = r._pecah_seri_wilayah(None, ["uu-41-2009", "permen-pupr-28-2015"])

        assert hasil == ["uu-41-2009", "permen-pupr-28-2015"]

    def test_tanpa_default_wilayah_tidak_disentuh(self, monkeypatch):
        dipanggil = []
        from app.retrieval import db

        monkeypatch.setattr(db, "match_dokumen", lambda conn, tokens: dipanggil.append(tokens) or [])
        r = RetrieverAsli()  # tanpa default_wilayah

        hasil = r._pecah_seri_wilayah(None, ["rdtr-sleman-tengah", "rdtr-sleman-timur"])

        assert hasil == ["rdtr-sleman-tengah", "rdtr-sleman-timur"]
        assert dipanggil == [], "tak perlu query dokumen kalau wilayah default tidak diset"
