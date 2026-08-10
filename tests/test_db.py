"""tests/test_db.py — `app/retrieval/db.py::_where()` murni pembentukan klausa SQL (tanpa koneksi
DB sungguhan) — testable offline. Fokus: `zona_prefix` (APP-2026-6191, cegah kontaminasi
lintas-keluarga zona di Lampiran V.B/VI, lihat docs/STATUS_RAG.md)."""

from datetime import date

from app.retrieval.base import RetrievalFilters
from app.retrieval.db import _where


def test_filters_none_klausa_kosong():
    assert _where(None) == ("", [])


def test_filters_kosong_klausa_kosong():
    assert _where(RetrievalFilters()) == ("", [])


def test_zona_exact_match():
    klausa, params = _where(RetrievalFilters(zona="R-3"))
    assert "zona = %s" in klausa
    assert "zona LIKE" not in klausa
    assert params == ["R-3"]


def test_zona_prefix_menghasilkan_klausa_exact_atau_like():
    klausa, params = _where(RetrievalFilters(zona_prefix="R"))
    assert "zona IS NULL OR zona = %s OR zona LIKE %s" in klausa
    assert params == ["R", "R-%"]


def test_zona_dan_zona_prefix_bisa_dipakai_bersamaan():
    # Independen — dua konsep beda (exact vs keluarga), keduanya bisa aktif sekaligus kalau perlu.
    klausa, params = _where(RetrievalFilters(zona="R-3", zona_prefix="R"))
    assert "zona = %s" in klausa
    assert "zona LIKE %s" in klausa
    assert params == ["R-3", "R", "R-%"]


def test_semua_filter_gabung_and():
    klausa, params = _where(RetrievalFilters(
        as_of=date(2024, 1, 1), zona_prefix="R", dokumen="Sleman Tengah", jenis="RDTR",
    ))
    assert klausa.count(" AND ") >= 4  # as_of(2) + zona_prefix(1) + dokumen(1) + jenis(1) klausa
    assert params == [date(2024, 1, 1), date(2024, 1, 1), "R", "R-%", "%Sleman Tengah%", "RDTR"]
