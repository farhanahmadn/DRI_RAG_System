from app.reasoning.kegiatan_data import cari_kegiatan_diizinkan, kegiatan_diizinkan_untuk_prompt
from app.schemas import IndikatorJejak


def _indikator(**overrides) -> IndikatorJejak:
    defaults = dict(
        poin_id="KEG-01",
        kategori="Kegiatan: Industri Besar/Pabrik",
        bobot=20.0,
        skor=60.0,
        kontribusi=60.0,
        nilai_input="X",
        ambang="I",
        operator="==",
        formula="",
        zona="C-1",
    )
    defaults.update(overrides)
    return IndikatorJejak(**defaults)


def test_cari_kegiatan_diizinkan_zona_dikenal():
    hasil = cari_kegiatan_diizinkan("C-1")
    assert len(hasil) > 0
    assert "Rumah toko (ruko) skala kecil" in hasil


def test_cari_kegiatan_diizinkan_zona_tak_dikenal():
    assert cari_kegiatan_diizinkan("ZONA-TAK-ADA") == []


def test_cari_kegiatan_diizinkan_zona_none():
    assert cari_kegiatan_diizinkan(None) == []


def test_kegiatan_diizinkan_untuk_prompt_klasifikasi_x():
    indikator = _indikator(nilai_input="X", zona="C-1")
    hasil = kegiatan_diizinkan_untuk_prompt(indikator)
    assert hasil is not None
    assert len(hasil) > 0


def test_kegiatan_diizinkan_untuk_prompt_none_untuk_i_t_b():
    for klasifikasi in ("I", "T", "B"):
        indikator = _indikator(nilai_input=klasifikasi, zona="C-1")
        assert kegiatan_diizinkan_untuk_prompt(indikator) is None


def test_kegiatan_diizinkan_untuk_prompt_none_untuk_kategori_lain():
    indikator = _indikator(kategori="KDB", nilai_input="X")
    assert kegiatan_diizinkan_untuk_prompt(indikator) is None
