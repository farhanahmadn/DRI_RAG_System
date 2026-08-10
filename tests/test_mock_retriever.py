from app.retrieval.base import RetrievalFilters
from app.retrieval.mock import MockRetriever


def test_search_respects_top_k():
    retriever = MockRetriever()
    results = retriever.search("bangunan gedung rencana", RetrievalFilters(), top_k=1)
    assert len(results) == 1


def test_get_by_reference_matches_pasal():
    retriever = MockRetriever()
    results = retriever.get_by_reference(["RDTR Pasal 1 Ayat 107"])

    assert len(results) > 0
    assert all(r.pasal == "1" for r in results)


def test_get_parent_returns_pasal_chunk():
    retriever = MockRetriever()
    children = retriever.search("KDB Koefisien Dasar Bangunan", RetrievalFilters(), top_k=5)
    child = next(c for c in children if c.level == "ayat")

    parent = retriever.get_parent(child.id)

    assert parent is not None
    assert parent.level == "pasal"
    assert parent.id == child.parent_id


def test_get_parent_returns_none_for_root_chunk():
    retriever = MockRetriever()
    root_pasal = next(c for c in retriever._chunks if c.level == "pasal" and c.id == "rdtr-p1")

    assert retriever.get_parent(root_pasal.id) is None


def test_search_finds_kdb_lampiran_vi():
    retriever = MockRetriever()
    results = retriever.search("KDB zona C-1", RetrievalFilters(), top_k=5)

    assert len(results) > 0
    assert any(r.id == "rdtr-lampiran-vi-c1" for r in results)


def test_get_by_reference_kdb_pasal_1_ayat_107():
    retriever = MockRetriever()
    results = retriever.get_by_reference(["RDTR Pasal 1 Ayat 107"])

    assert len(results) > 0
    assert all(r.pasal == "1" for r in results)
    assert any(r.id == "rdtr-p1-a107" for r in results)


def test_get_by_reference_kdb_lampiran_vi():
    retriever = MockRetriever()
    results = retriever.get_by_reference(["RDTR Lampiran VI"])

    assert any(r.id == "rdtr-lampiran-vi-c1" for r in results)
    assert any(r.id == "rdtr-lampiran-vi-r1" for r in results)


def test_search_finds_kdh_lampiran_vi_r1():
    retriever = MockRetriever()
    results = retriever.search("KDH zona R-1", RetrievalFilters(), top_k=5)

    assert len(results) > 0
    assert any(r.id == "rdtr-lampiran-vi-r1" for r in results)


def test_get_parent_kdb_ayat_ke_pasal():
    retriever = MockRetriever()
    parent = retriever.get_parent("rdtr-p1-a107")

    assert parent is not None
    assert parent.id == "rdtr-p1"
    assert parent.level == "pasal"


def test_search_finds_kegiatan_lampiran_v():
    retriever = MockRetriever()
    results = retriever.search("kegiatan ITBX zona C-1", RetrievalFilters(), top_k=5)

    assert len(results) > 0
    assert any(r.id == "rdtr-lampiran-v-c1" for r in results)


def test_get_by_reference_kegiatan_pasal_1_ayat_108():
    retriever = MockRetriever()
    results = retriever.get_by_reference(["RDTR Pasal 1 Ayat 108"])

    assert len(results) > 0
    assert all(r.pasal == "1" for r in results)
    assert any(r.id == "rdtr-p1-a108" for r in results)


def test_get_by_reference_kegiatan_lampiran_v():
    retriever = MockRetriever()
    results = retriever.get_by_reference(["RDTR Lampiran V"])

    assert any(r.id == "rdtr-lampiran-v-c1" for r in results)


class TestZonaPrefixFilter:
    """zona_prefix (APP-2026-6191): cegah kontaminasi lintas-KELUARGA zona di hasil search()."""

    def test_zona_prefix_r_hanya_terima_chunk_keluarga_r(self):
        retriever = MockRetriever()
        results = retriever.search("KDB", RetrievalFilters(zona_prefix="R"), top_k=10)
        assert any(r.id == "rdtr-lampiran-vi-r1" for r in results)
        assert all(r.id != "rdtr-lampiran-vi-c1" for r in results)  # zona C-1, keluarga beda

    def test_zona_prefix_c_hanya_terima_chunk_keluarga_c(self):
        retriever = MockRetriever()
        results = retriever.search("KDB", RetrievalFilters(zona_prefix="C"), top_k=10)
        assert any(r.id == "rdtr-lampiran-vi-c1" for r in results)
        assert all(r.id != "rdtr-lampiran-vi-r1" for r in results)  # zona R-1, keluarga beda

    def test_chunk_tanpa_zona_selalu_lolos_filter(self):
        # Chunk umum (mis. definisi Pasal 1, zona=None) bukan spesifik ke satu zona — harus tetap
        # ikut apa pun zona_prefix-nya (semantik sama dgn filter zona/jenis lain di kode ini).
        retriever = MockRetriever()
        results = retriever.search("bangunan gedung rencana", RetrievalFilters(zona_prefix="R"), top_k=10)
        assert any(r.id == "rdtr-p1-a107" for r in results)  # zona=None

    def test_tanpa_zona_prefix_semua_zona_ikut(self):
        retriever = MockRetriever()
        results = retriever.search("KDB", RetrievalFilters(), top_k=10)
        ids = {r.id for r in results}
        assert "rdtr-lampiran-vi-r1" in ids
        assert "rdtr-lampiran-vi-c1" in ids
