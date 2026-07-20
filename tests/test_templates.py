import pytest

from app.reasoning.templates import template_aman, template_low_confidence
from app.schemas import IndikatorJejak


def _indikator(**overrides) -> IndikatorJejak:
    defaults = dict(
        poin_id="X-01",
        kategori="Kegiatan Zonasi",
        bobot=10.0,
        skor=0.0,
        kontribusi=0.0,
        nilai_input=0.0,
        ambang=0.0,
        operator="<=",
        formula="",
    )
    defaults.update(overrides)
    return IndikatorJejak(**defaults)


def test_template_aman_kategori_non_numerik():
    indikator = _indikator(kategori="Kegiatan Zonasi", skor=0.0)
    poin = template_aman(indikator)

    assert poin.status == "Aman"
    assert poin.low_confidence is False
    assert poin.sitasi == []
    assert poin.rekomendasi.tipe == "kegiatan"
    assert poin.rekomendasi.target is None


def test_template_aman_kategori_numerik_isi_target():
    indikator = _indikator(
        kategori="KDB",
        skor=0.0,
        nilai_input=500.0,
        ambang=0.6,
        operator="<=",
        luas_lahan=1000.0,
    )
    poin = template_aman(indikator)

    assert poin.rekomendasi.tipe == "numerik"
    assert poin.rekomendasi.target == 600.0


def test_template_aman_raises_jika_skor_bukan_nol():
    indikator = _indikator(skor=5.0)
    with pytest.raises(ValueError):
        template_aman(indikator)


@pytest.mark.parametrize("skor,expected_status", [(0.0, "Aman"), (7.5, "Tidak Aman")])
def test_template_low_confidence_status_mengikuti_skor(skor, expected_status):
    indikator = _indikator(skor=skor, kontribusi=skor)
    poin = template_low_confidence(indikator)

    assert poin.status == expected_status
    assert poin.low_confidence is True
    assert poin.sitasi == []


def test_template_low_confidence_tetap_hitung_target_numerik():
    indikator = _indikator(
        kategori="KDH",
        skor=3.0,
        kontribusi=3.0,
        nilai_input=250.0,
        ambang=0.3,
        operator=">=",
        luas_lahan=1000.0,
    )
    poin = template_low_confidence(indikator)

    assert poin.rekomendasi.tipe == "numerik"
    assert poin.rekomendasi.target == 300.0
    assert poin.rekomendasi.disclaimer is not None
