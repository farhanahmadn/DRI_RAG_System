import pytest

from app.reasoning.calculator import (
    hitung_target_rekomendasi,
    klasifikasi_tipe_rekomendasi,
    rakit_fakta_verdict,
    rakit_status_kegiatan,
    rakit_status_numerik,
)
from app.schemas import IndikatorJejak


def _indikator(**overrides) -> IndikatorJejak:
    defaults = dict(
        poin_id="X-01",
        kategori="KDB",
        bobot=10.0,
        skor=5.0,
        kontribusi=5.0,
        nilai_input=0.0,
        ambang=0.0,
        operator="<=",
        formula="",
    )
    defaults.update(overrides)
    return IndikatorJejak(**defaults)


def test_klasifikasi_tipe_numerik():
    assert klasifikasi_tipe_rekomendasi("KDB") == "numerik"
    assert klasifikasi_tipe_rekomendasi("Koefisien Lantai Bangunan (KLB)") == "numerik"
    assert klasifikasi_tipe_rekomendasi("kdh") == "numerik"


def test_klasifikasi_tipe_lokasional():
    assert klasifikasi_tipe_rekomendasi("Banjir") == "lokasional"
    assert klasifikasi_tipe_rekomendasi("Resapan Air") == "lokasional"
    assert klasifikasi_tipe_rekomendasi("Status LP2B") == "lokasional"
    assert klasifikasi_tipe_rekomendasi("Sempadan Sungai") == "lokasional"


def test_klasifikasi_tipe_kegiatan_default():
    assert klasifikasi_tipe_rekomendasi("Kegiatan Zonasi") == "kegiatan"


def test_target_dari_backend_tidak_dihitung_ulang():
    indikator = _indikator(
        kategori="KDB",
        nilai_input=999.0,
        ambang=0.1,
        luas_lahan=1.0,
        target_rekomendasi={"footprint_maks": 123.0, "selisih": 45.0},
    )
    assert hitung_target_rekomendasi(indikator) == {"footprint_maks": 123.0, "selisih": 45.0}


def test_kdb_dengan_luas_lahan_patuh():
    indikator = _indikator(
        kategori="KDB",
        nilai_input=550.0,
        ambang=0.6,
        operator="<=",
        luas_lahan=1000.0,
    )
    result = hitung_target_rekomendasi(indikator)
    assert result == {"target_maks": 600.0, "selisih": -50.0}


def test_kdb_dengan_luas_lahan_melanggar():
    indikator = _indikator(
        kategori="KDB",
        nilai_input=650.0,
        ambang=0.6,
        operator="<=",
        luas_lahan=1000.0,
    )
    result = hitung_target_rekomendasi(indikator)
    assert result == {"target_maks": 600.0, "selisih": 50.0}


def test_kdb_batas_tepat_di_ambang():
    indikator = _indikator(
        kategori="KDB",
        nilai_input=600.0,
        ambang=0.6,
        operator="<=",
        luas_lahan=1000.0,
    )
    result = hitung_target_rekomendasi(indikator)
    assert result["selisih"] == 0.0


def test_kdh_minimal_dengan_luas_lahan():
    indikator = _indikator(
        kategori="KDH",
        nilai_input=250.0,
        ambang=0.3,
        operator=">=",
        luas_lahan=1000.0,
    )
    result = hitung_target_rekomendasi(indikator)
    assert result == {"target_maks": 300.0, "selisih": 50.0}


def test_klb_fallback_tanpa_luas_lahan():
    indikator = _indikator(
        kategori="KLB",
        nilai_input=2.6,
        ambang=2.4,
        operator="<=",
        luas_lahan=None,
    )
    result = hitung_target_rekomendasi(indikator)
    assert result["target"] == pytest.approx(2.4)
    assert result["selisih"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "kategori",
    ["Kegiatan Zonasi", "Banjir", "Status LP2B", "Resapan Air", "Sempadan Sungai"],
)
def test_kategori_non_numerik_selalu_none(kategori):
    indikator = _indikator(
        kategori=kategori,
        nilai_input=100.0,
        ambang=50.0,
        luas_lahan=1000.0,
    )
    assert hitung_target_rekomendasi(indikator) is None


def test_nilai_non_numerik_pada_kategori_numerik_raises_typeerror():
    indikator = _indikator(kategori="KDB", nilai_input="tinggi", ambang=0.6)
    with pytest.raises(TypeError):
        hitung_target_rekomendasi(indikator)


def test_operator_tidak_didukung_raises_valueerror():
    indikator = _indikator(kategori="KDB", nilai_input=0.5, ambang=0.6, operator="==")
    with pytest.raises(ValueError):
        hitung_target_rekomendasi(indikator)


def test_rakit_status_numerik_kdb_melebihi():
    indikator = _indikator(kategori="KDB", nilai_input=90.0, ambang=80.0, operator="<=")
    status = rakit_status_numerik(indikator)
    assert status is not None
    assert "MELEBIHI" in status
    assert "90.0" in status
    assert "80.0" in status


def test_rakit_status_numerik_kdh_kurang():
    indikator = _indikator(kategori="KDH", nilai_input=15.0, ambang=20.0, operator=">=")
    status = rakit_status_numerik(indikator)
    assert status is not None
    assert "KURANG" in status
    assert "15.0" in status
    assert "20.0" in status


def test_rakit_status_numerik_sesuai():
    indikator = _indikator(kategori="KDB", nilai_input=70.0, ambang=80.0, operator="<=")
    status = rakit_status_numerik(indikator)
    assert status is not None
    assert "SESUAI" in status


def test_rakit_status_numerik_none_untuk_kategori_non_numerik():
    indikator = _indikator(kategori="Lokasional LP2B", nilai_input=1.0, ambang=1.0, operator="==")
    assert rakit_status_numerik(indikator) is None


@pytest.mark.parametrize(
    "klasifikasi,label_diharapkan",
    [("I", "Diizinkan"), ("T", "Terbatas"), ("B", "Bersyarat"), ("X", "Dilarang")],
)
def test_rakit_status_kegiatan_semua_kelas(klasifikasi, label_diharapkan):
    indikator = _indikator(
        kategori="Kegiatan: Gudang",
        nilai_input=klasifikasi,
        ambang="I",
        operator="==",
        zona="C-1",
    )
    status = rakit_status_kegiatan(indikator)
    assert status is not None
    assert f"KLASIFIKASI: {klasifikasi}" in status
    assert label_diharapkan in status


def test_rakit_status_kegiatan_klasifikasi_tak_dikenal():
    indikator = _indikator(kategori="Kegiatan: Gudang", nilai_input="Z", ambang="I", operator="==")
    assert rakit_status_kegiatan(indikator) is None


def test_rakit_status_kegiatan_none_untuk_kategori_non_kegiatan():
    indikator = _indikator(kategori="KDB", nilai_input="X", ambang="I", operator="==")
    assert rakit_status_kegiatan(indikator) is None


def test_rakit_fakta_verdict_dispatch_numerik():
    indikator = _indikator(kategori="KDB", nilai_input=90.0, ambang=80.0, operator="<=")
    assert rakit_fakta_verdict(indikator) == rakit_status_numerik(indikator)
    assert "STATUS" in rakit_fakta_verdict(indikator)


def test_rakit_fakta_verdict_dispatch_kegiatan():
    indikator = _indikator(kategori="Kegiatan: Gudang", nilai_input="X", ambang="I", operator="==", zona="C-1")
    assert rakit_fakta_verdict(indikator) == rakit_status_kegiatan(indikator)
    assert "KLASIFIKASI" in rakit_fakta_verdict(indikator)


def test_rakit_fakta_verdict_dispatch_lokasional_none():
    indikator = _indikator(kategori="Lokasional LP2B", nilai_input="dalam_lp2b", ambang="tidak_dalam_lp2b", operator="==")
    assert rakit_fakta_verdict(indikator) is None
