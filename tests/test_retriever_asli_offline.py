"""tests/test_retriever_asli_offline.py — bagian RetrieverAsli yang TIDAK butuh koneksi DB
sungguhan (murni in-memory: cache). Beda dari tests/test_retrieval.py (live, butuh DB+model
retrieval sungguhan) — constructor RetrieverAsli tidak connect DB secara eager, jadi aman
diinstansiasi di sini tanpa DATABASE_URL."""

from app.retrieval.base import Chunk, RetrievalFilters
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


class TestKunciCacheMembedakanKeluargaZona:
    """Regresi: kunci cache yang tak memuat `zona_prefix` menyajikan tabel keluarga zona yang SALAH.

    Dibuktikan dulu, bukan diantisipasi: dengan kunci lama, `search()` dengan query yang sama untuk
    pemohon keluarga "KT" mengembalikan hasil keluarga "R" yang sudah lebih dulu masuk cache — persis
    kelas bug APP-2026-6191 (sitasi lintas keluarga zona), kali ini lewat cache bukan lewat SQL.

    Sebelum query intensitas dipertajam, dampaknya nyaris tak terlihat: cabang keluarga memakai query
    pendek yang hampir tak pernah menemukan tabel ambang, jadi yang tertukar pun sama-sama kosong.
    Begitu query tajam dipakai di kedua cabang, tabrakan ini menyajikan tabel keluarga lain DENGAN
    ambang KDB/KLB/KDH yang berbeda, dan terlihat meyakinkan.
    """

    QUERY = "ambang KDB KLB KDH maksimal minimal"

    def _retriever_terekam(self):
        r = RetrieverAsli(default_wilayah="Sleman Tengah")
        r._cache._store.clear()
        terpanggil: list[str | None] = []

        def palsu(query, filters, top_k, *, tanpa_lexical=False):
            terpanggil.append(filters.zona_prefix)
            return [Chunk(id=f"chunk-{filters.zona_prefix}", level="tabel",
                          dokumen="RDTR Sleman Tengah",
                          teks=f"Lampiran VI keluarga {filters.zona_prefix}", skor=1.0)]

        r._search_uncached = palsu
        return r, terpanggil

    def test_keluarga_zona_berbeda_tidak_saling_pakai_cache(self):
        r, terpanggil = self._retriever_terekam()

        hasil_r = r.search(self.QUERY, RetrievalFilters(zona_prefix="R"), top_k=3)
        hasil_kt = r.search(self.QUERY, RetrievalFilters(zona_prefix="KT"), top_k=3)

        assert [c.id for c in hasil_r] == ["chunk-R"]
        assert [c.id for c in hasil_kt] == ["chunk-KT"], \
            "pemohon KT tidak boleh menerima tabel keluarga R dari cache"
        assert terpanggil == ["R", "KT"], "masing-masing keluarga wajib dihitung sendiri"

    def test_keluarga_zona_sama_tetap_memakai_cache(self):
        """Perbaikan tidak boleh mematikan cache-nya — permintaan identik tetap satu kali hitung."""
        r, terpanggil = self._retriever_terekam()

        r.search(self.QUERY, RetrievalFilters(zona_prefix="R"), top_k=3)
        r.search(self.QUERY, RetrievalFilters(zona_prefix="R"), top_k=3)

        assert terpanggil == ["R"]

    def test_jalur_kandidat_berbeda_tidak_saling_pakai_cache(self):
        """dense-saja dan hibrida menghasilkan kandidat berbeda — keduanya tak boleh bertukar hasil."""
        r = RetrieverAsli(default_wilayah="Sleman Tengah")
        r._cache._store.clear()
        terpanggil: list[bool] = []

        def palsu(query, filters, top_k, *, tanpa_lexical=False):
            terpanggil.append(tanpa_lexical)
            return [Chunk(id=f"chunk-{'dense' if tanpa_lexical else 'hibrida'}", level="tabel",
                          dokumen="RDTR Sleman Tengah", teks="x", skor=1.0)]

        r._search_uncached = palsu
        f = RetrievalFilters(zona_prefix="R")

        hibrida = r.search(self.QUERY, f, top_k=3)
        dense = r.search(self.QUERY, f, top_k=3, tanpa_lexical=True)

        assert [c.id for c in hibrida] == ["chunk-hibrida"]
        assert [c.id for c in dense] == ["chunk-dense"]
        assert terpanggil == [False, True]
