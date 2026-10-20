"""Matriks rekomendasi_sistem sesuai Spesifikasi Kontrak L3 BE v1.0.0 bagian 4 (prioritas Tolak > Lolos Terbatas >
Setuju Bersyarat > Setuju)."""

import pytest

from app.reasoning.rekomendasi import turunkan_rekomendasi


@pytest.mark.parametrize("impact_category", [None, "Rendah", "Sangat Tinggi"])
def test_tidak_lolos_selalu_tolak(impact_category):
    assert turunkan_rekomendasi("Tidak Lolos", impact_category) == "Tolak"


@pytest.mark.parametrize("gate", ["Lolos", "Lolos Bersyarat", "Lolos Terbatas"])
def test_itbx_x_selalu_tolak(gate):
    assert turunkan_rekomendasi(gate, "Rendah", "X") == "Tolak"


@pytest.mark.parametrize("impact_category", [None, "Rendah", "Sangat Tinggi"])
def test_gate_lolos_terbatas(impact_category):
    assert turunkan_rekomendasi("Lolos Terbatas", impact_category) == "Lolos Terbatas"


@pytest.mark.parametrize("gate", ["Lolos", "Lolos Bersyarat"])
@pytest.mark.parametrize("itbx", ["T", "TB"])
def test_itbx_t_mengalahkan_bersyarat(gate, itbx):
    assert turunkan_rekomendasi(gate, "Tinggi", itbx) == "Lolos Terbatas"


@pytest.mark.parametrize("impact_category", [None, "Sangat Tinggi"])
def test_lolos_bersyarat_selalu_setuju_bersyarat(impact_category):
    assert turunkan_rekomendasi("Lolos Bersyarat", impact_category) == "Setuju Bersyarat"


def test_itbx_b_setuju_bersyarat_walau_gate_lolos():
    assert turunkan_rekomendasi("Lolos", "Rendah", "B") == "Setuju Bersyarat"


@pytest.mark.parametrize("kategori", ["Sedang", "Tinggi", "Sangat Tinggi"])
def test_dampak_perlu_mitigasi_setuju_bersyarat(kategori):
    assert turunkan_rekomendasi("Lolos", kategori) == "Setuju Bersyarat"


@pytest.mark.parametrize("kategori", ["Rendah", None])
def test_lolos_dampak_rendah_atau_belum_dinilai_setuju(kategori):
    assert turunkan_rekomendasi("Lolos", kategori, "I") == "Setuju"


def test_kategori_dampak_tak_mengalahkan_tolak_atau_terbatas():
    assert turunkan_rekomendasi("Tidak Lolos", "Sangat Tinggi") == "Tolak"
    assert turunkan_rekomendasi("Lolos Terbatas", "Sangat Tinggi") == "Lolos Terbatas"


def test_gate_status_tak_dikenal_raise():
    with pytest.raises(ValueError, match="tidak dikenal"):
        turunkan_rekomendasi("Status Aneh", "Sedang")
