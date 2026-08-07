import pytest

from app.reasoning.calculator import (
    hitung_target_intensitas,
    hitung_target_kdb,
    hitung_target_kdh,
    hitung_target_klb,
    hitung_target_mitigasi_dampak,
    normalisasi_kategori_dampak,
    pilih_target_mitigasi_dampak,
    pilih_target_utama_intensitas,
    sarankan_arah_mitigasi_dampak,
)
from app.schemas import ImpactAssessment, IntensitasTahap, ParameterIntensitas


class TestHitungTargetKdb:
    def test_melampaui_ambang_dengan_luas_lahan(self):
        param = ParameterIntensitas(usulan=70, ambang_maks=60, memenuhi=False, satuan="persen")
        hasil = hitung_target_kdb(param, luas_lahan_m2=850)
        assert hasil["target_kdb"] == 60
        assert hasil["selisih"] == 10
        assert hasil["footprint_maks_m2"] == 510  # 850 * 0.6

    def test_melampaui_ambang_tanpa_luas_lahan(self):
        param = ParameterIntensitas(usulan=70, ambang_maks=60, memenuhi=False, satuan="persen")
        hasil = hitung_target_kdb(param, luas_lahan_m2=None)
        assert hasil == {"target_kdb": 60, "selisih": 10}

    def test_patuh_mengembalikan_none(self):
        param = ParameterIntensitas(usulan=47, ambang_maks=60, memenuhi=True, satuan="persen")
        assert hitung_target_kdb(param, luas_lahan_m2=850) is None

    def test_ambang_maks_null_mengembalikan_none_bukan_raise(self):
        param = ParameterIntensitas(usulan=70, ambang_maks=None, memenuhi=True, satuan="persen")
        assert hitung_target_kdb(param, luas_lahan_m2=850) is None


class TestHitungTargetKlb:
    def test_melampaui_ambang_rasio_tidak_dibagi_100(self):
        param = ParameterIntensitas(usulan=2.2, ambang_maks=1.8, memenuhi=False, satuan="rasio")
        hasil = hitung_target_klb(param, luas_lahan_m2=850)
        assert hasil["target_klb"] == 1.8
        assert hasil["selisih"] == pytest.approx(0.4)
        assert hasil["luas_lantai_maks_m2"] == pytest.approx(1530)  # 850 * 1.8, bukan /100

    def test_patuh_mengembalikan_none(self):
        param = ParameterIntensitas(usulan=0.94, ambang_maks=1.8, memenuhi=True, satuan="rasio")
        assert hitung_target_klb(param, luas_lahan_m2=850) is None

    def test_ambang_maks_null_mengembalikan_none(self):
        param = ParameterIntensitas(usulan=2.2, ambang_maks=None, memenuhi=True, satuan="rasio")
        assert hitung_target_klb(param, luas_lahan_m2=850) is None


class TestHitungTargetKdh:
    def test_kurang_dari_ambang_dengan_luas_lengkap(self):
        param = ParameterIntensitas(usulan=15, ambang_min=20, memenuhi=False, satuan="persen")
        hasil = hitung_target_kdh(param, luas_lahan_m2=850, luas_rth_usulan_m2=100)
        assert hasil["target_kdh"] == 20
        assert hasil["selisih"] == 5
        assert hasil["rth_dibutuhkan_m2"] == 170  # 850 * 0.2
        assert hasil["rth_kurang_m2"] == 70  # 170 - 100

    def test_kurang_dari_ambang_tanpa_luas_rth_usulan(self):
        param = ParameterIntensitas(usulan=15, ambang_min=20, memenuhi=False, satuan="persen")
        hasil = hitung_target_kdh(param, luas_lahan_m2=850, luas_rth_usulan_m2=None)
        assert hasil["rth_dibutuhkan_m2"] == 170
        assert "rth_kurang_m2" not in hasil

    def test_kurang_dari_ambang_tanpa_luas_lahan(self):
        param = ParameterIntensitas(usulan=15, ambang_min=20, memenuhi=False, satuan="persen")
        hasil = hitung_target_kdh(param, luas_lahan_m2=None, luas_rth_usulan_m2=100)
        assert hasil == {"target_kdh": 20, "selisih": 5}

    def test_patuh_mengembalikan_none(self):
        param = ParameterIntensitas(usulan=29.4, ambang_min=0, memenuhi=True, satuan="persen")
        assert hitung_target_kdh(param, luas_lahan_m2=850, luas_rth_usulan_m2=250) is None

    def test_ambang_min_null_mengembalikan_none(self):
        param = ParameterIntensitas(usulan=15, ambang_min=None, memenuhi=True, satuan="persen")
        assert hitung_target_kdh(param, luas_lahan_m2=850, luas_rth_usulan_m2=100) is None


class TestHitungTargetIntensitas:
    def test_satu_parameter_melanggar_setara_fixture_bersyarat(self):
        intensitas = IntensitasTahap(
            status="MELAMPAUI_BATAS",
            lolos=False,
            parameter={
                "kdb": ParameterIntensitas(usulan=70, ambang_maks=60, memenuhi=False, satuan="persen"),
                "klb": ParameterIntensitas(usulan=0.94, ambang_maks=1.8, memenuhi=True, satuan="rasio"),
                "kdh": ParameterIntensitas(usulan=29.4, ambang_min=0, memenuhi=True, satuan="persen"),
            },
            luas_tapak_m2=400,
            jumlah_lantai=2,
            luas_rth_usulan_m2=250,
            reason="Lolos Bersyarat karena usulan KDB melampaui ambang maksimum.",
        )
        hasil = hitung_target_intensitas(intensitas, luas_lahan_m2=850)
        assert set(hasil) == {"kdb"}
        assert hasil["kdb"]["footprint_maks_m2"] == 510

    def test_semua_patuh_setara_fixture_lolos(self):
        intensitas = IntensitasTahap(
            status="MEMENUHI_SYARAT",
            lolos=True,
            parameter={
                "kdb": ParameterIntensitas(usulan=47, ambang_maks=60, memenuhi=True, satuan="persen"),
                "klb": ParameterIntensitas(usulan=0.94, ambang_maks=1.8, memenuhi=True, satuan="rasio"),
                "kdh": ParameterIntensitas(usulan=29.4, ambang_min=0, memenuhi=True, satuan="persen"),
            },
            luas_tapak_m2=400,
            jumlah_lantai=2,
            luas_rth_usulan_m2=250,
            reason="Lolos karena usulan KDB, KLB, dan KDH memenuhi standar regulasi.",
        )
        assert hitung_target_intensitas(intensitas, luas_lahan_m2=850) == {}


class TestPilihTargetUtamaIntensitas:
    def test_satu_parameter_melanggar(self):
        target_map = {"kdb": {"target_kdb": 60.0, "selisih": 10.0, "footprint_maks_m2": 510.0}}
        assert pilih_target_utama_intensitas(target_map) == 60.0

    def test_kosong_mengembalikan_none(self):
        assert pilih_target_utama_intensitas({}) is None


class TestNormalisasiKategoriDampak:
    def test_uppercase_ke_title_case(self):
        assert normalisasi_kategori_dampak("SEDANG") == "Sedang"

    def test_multi_kata(self):
        assert normalisasi_kategori_dampak("SANGAT TINGGI") == "Sangat Tinggi"

    def test_none_tetap_none(self):
        assert normalisasi_kategori_dampak(None) is None

    def test_sudah_title_case_idempoten(self):
        assert normalisasi_kategori_dampak("Sangat Tinggi") == "Sangat Tinggi"


class TestSarankanArahMitigasiDampak:
    def _impact(self, **override) -> ImpactAssessment:
        default = dict(dinilai=True, impact_category="Sedang")
        default.update(override)
        return ImpactAssessment(**default)

    def test_tidak_dinilai(self):
        hasil = sarankan_arah_mitigasi_dampak(self._impact(dinilai=False, impact_category=None))
        assert hasil == {"perlu_mitigasi": False, "arah": []}

    def test_kategori_rendah_tidak_perlu_mitigasi(self):
        hasil = sarankan_arah_mitigasi_dampak(self._impact(impact_category="RENDAH"))
        assert hasil == {"perlu_mitigasi": False, "arah": []}

    def test_kategori_sedang_tidak_perlu_mitigasi(self):
        hasil = sarankan_arah_mitigasi_dampak(self._impact(impact_category="SEDANG"))
        assert hasil == {"perlu_mitigasi": False, "arah": []}

    def test_kategori_tinggi_perlu_mitigasi(self):
        hasil = sarankan_arah_mitigasi_dampak(self._impact(impact_category="TINGGI"))
        assert hasil["perlu_mitigasi"] is True
        assert len(hasil["arah"]) == 4
        assert "sumur resapan" in hasil["arah"]
        assert "kolam retensi" in hasil["arah"]

    def test_kategori_sangat_tinggi_perlu_mitigasi(self):
        hasil = sarankan_arah_mitigasi_dampak(self._impact(impact_category="SANGAT TINGGI"))
        assert hasil["perlu_mitigasi"] is True

    def test_tidak_menghitung_ulang_c_hanya_echo_apa_adanya(self):
        koefisien = {"C_Atap": 0.95, "C_Taman": 0.3, "C_Daerah_tak_Terbangun": 0.3}
        detail = "C_Sesudah = (Proporsi KDB x C_Atap) + (Proporsi KDH x C_Taman)"
        hasil = sarankan_arah_mitigasi_dampak(
            self._impact(impact_category="Tinggi", c_coefficients=koefisien, calculation_details=detail)
        )
        assert hasil["c_coefficients_referensi"] == koefisien
        assert hasil["calculation_details_referensi"] == detail

    def test_tanpa_c_coefficients_tidak_muncul_di_hasil(self):
        hasil = sarankan_arah_mitigasi_dampak(self._impact(impact_category="Tinggi"))
        assert "c_coefficients_referensi" not in hasil
        assert "calculation_details_referensi" not in hasil


_THRESHOLD_BANDS_STANDAR = {
    "Rendah": "index < 1.5",
    "Sedang": "1.5-2.5",
    "Tinggi": "2.5-4.0",
    "Sangat Tinggi": "> 4.0",
}


class TestHitungTargetMitigasiDampak:
    def _impact(self, **override) -> ImpactAssessment:
        default = dict(
            dinilai=True, impact_category="Tinggi", runoff_change_index=2.85,
            threshold_bands=_THRESHOLD_BANDS_STANDAR,
        )
        default.update(override)
        return ImpactAssessment(**default)

    def test_kasus_nyata_app_2026_6191_tinggi_ke_sedang(self):
        # Fixture nyata: index=2.85 (band Tinggi 2.5-4.0) -> target turun ke bawah 2.5 (band Sedang).
        hasil = hitung_target_mitigasi_dampak(self._impact())
        assert hasil == {
            "runoff_change_index_maks": 2.5,
            "kategori_target": "Sedang",
            "index_saat_ini": 2.85,
        }

    def test_sangat_tinggi_ke_tinggi(self):
        hasil = hitung_target_mitigasi_dampak(
            self._impact(impact_category="Sangat Tinggi", runoff_change_index=5.0)
        )
        assert hasil["runoff_change_index_maks"] == 4.0
        assert hasil["kategori_target"] == "Tinggi"

    def test_kategori_rendah_tak_ada_target_lebih_ringan(self):
        hasil = hitung_target_mitigasi_dampak(
            self._impact(impact_category="Rendah", runoff_change_index=1.0)
        )
        assert hasil == {}

    def test_index_none_return_kosong(self):
        hasil = hitung_target_mitigasi_dampak(self._impact(runoff_change_index=None))
        assert hasil == {}

    def test_threshold_bands_none_return_kosong(self):
        hasil = hitung_target_mitigasi_dampak(self._impact(threshold_bands=None))
        assert hasil == {}

    def test_kategori_tak_dikenal_return_kosong(self):
        hasil = hitung_target_mitigasi_dampak(self._impact(impact_category="Entah Apa"))
        assert hasil == {}

    def test_tidak_menghitung_ulang_rumus_c_murni_aritmatika_threshold(self):
        # Ganti threshold_bands custom -> hasil HARUS ikut angka baru (bukan hardcode), membuktikan
        # ini murni baca `threshold_bands` yang diberi, bukan rumus C yang di-hardcode.
        bands_custom = {"Rendah": "index < 1.0", "Sedang": "1.0-9.0", "Tinggi": "9.0-20.0",
                        "Sangat Tinggi": "> 20.0"}
        hasil = hitung_target_mitigasi_dampak(
            self._impact(threshold_bands=bands_custom, runoff_change_index=10.0)
        )
        assert hasil["runoff_change_index_maks"] == 9.0


class TestPilihTargetMitigasiDampak:
    def test_ambil_runoff_change_index_maks(self):
        assert pilih_target_mitigasi_dampak({"runoff_change_index_maks": 2.5}) == 2.5

    def test_dict_kosong_return_none(self):
        assert pilih_target_mitigasi_dampak({}) is None
