import pytest

from app.reasoning.rekomendasi import turunkan_rekomendasi


@pytest.mark.parametrize("impact_category", [None, "Rendah", "Sangat Tinggi"])
def test_tidak_lolos_selalu_tidak_setuju(impact_category):
    assert turunkan_rekomendasi("Tidak Lolos", impact_category) == "Tidak Setuju"


@pytest.mark.parametrize("impact_category", [None, "Sangat Tinggi"])
def test_lolos_bersyarat_selalu_setuju_bersyarat(impact_category):
    assert turunkan_rekomendasi("Lolos Bersyarat", impact_category) == "Setuju Bersyarat"


def test_lolos_dampak_tinggi_setuju_bersyarat():
    assert turunkan_rekomendasi("Lolos", "Tinggi") == "Setuju Bersyarat"


def test_lolos_dampak_sangat_tinggi_setuju_bersyarat():
    assert turunkan_rekomendasi("Lolos", "Sangat Tinggi") == "Setuju Bersyarat"


def test_lolos_dampak_sedang_setuju():
    assert turunkan_rekomendasi("Lolos", "Sedang") == "Setuju"


def test_lolos_dampak_rendah_setuju():
    assert turunkan_rekomendasi("Lolos", "Rendah") == "Setuju"


def test_lolos_dampak_belum_dinilai_setuju():
    assert turunkan_rekomendasi("Lolos", None) == "Setuju"


def test_gate_status_tak_dikenal_raise():
    with pytest.raises(ValueError, match="tidak dikenal"):
        turunkan_rekomendasi("Status Aneh", "Sedang")
