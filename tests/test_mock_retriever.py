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
