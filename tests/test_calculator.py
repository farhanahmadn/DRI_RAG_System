import pytest

from app.reasoning.calculator import (
    bangun_langkah_konkret_dampak,
    bangun_langkah_konkret_intensitas,
    format_target_parameter,
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

    def test_kategori_sedang_return_kosong_selaras_sarankan_arah_mitigasi(self):
        # Bug ditemukan live (APP-2026-8376): kategori "Sedang" BUKAN anggota
        # KATEGORI_DAMPAK_BERSYARAT (cuma Tinggi/Sangat Tinggi) — sarankan_arah_mitigasi_dampak
        # sudah benar menandai perlu_mitigasi=False utk Sedang, tapi versi lama fungsi ini tetap
        # menghitung target (krn cuma cek "ada kategori lebih ringan?"), bikin poin "aman" dapat
        # target kontradiktif. Gate DISAMAKAN persis dgn KATEGORI_DAMPAK_BERSYARAT sekarang.
        hasil = hitung_target_mitigasi_dampak(
            self._impact(impact_category="Sedang", runoff_change_index=1.56)
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


class TestHitungTargetMitigasiDampakDariBE:
    """APP-2026-8025/-5067: BE kini (kadang) kirim `rekomendasi_mitigasi` — target + rincian
    penyesuaian lahan (m²/KDB%/KDH%) + dimensi sumur resapan SUDAH DIHITUNG PENUH. Diprioritaskan
    di atas rekonstruksi band lama kalau ada (echo apa adanya, FAITHFUL)."""

    _REKOMENDASI_MITIGASI = {
        "status_dampak": "TINGGI",
        "target_kategori": "SEDANG",
        "target_indeks_maks": 2.5,
        "rekomendasi_penyesuaian_lahan": {
            "luas_bangunan_maks_m2": 11551.06,
            "luas_rth_min_m2": 3850.35,
            "kdb_maks_persen": 75,
            "kdh_min_persen": 25,
            "catatan": "Agar indeks runoff turun ke <= 2.5 (kategori Sedang).",
        },
        "dimensi_minimum": {"nilai": 19.01, "satuan": "m³"},
        "saran": "Untuk menurunkan dampak dari TINGGI menjadi SEDANG...",
    }

    def _impact_dgn_rekomendasi(self, **override) -> ImpactAssessment:
        default = dict(
            dinilai=True, impact_category="Tinggi", runoff_change_index=2.923,
            threshold_bands=_THRESHOLD_BANDS_STANDAR,
            rekomendasi_mitigasi=self._REKOMENDASI_MITIGASI,
        )
        default.update(override)
        return ImpactAssessment(**default)

    def test_target_indeks_dari_be_dipakai_bukan_dihitung_ulang(self):
        hasil = hitung_target_mitigasi_dampak(self._impact_dgn_rekomendasi())
        assert hasil["runoff_change_index_maks"] == 2.5
        assert hasil["kategori_target"] == "Sedang"  # dinormalisasi dari "SEDANG" BE
        assert hasil["index_saat_ini"] == 2.923

    def test_penyesuaian_lahan_dan_dimensi_diteruskan_apa_adanya(self):
        hasil = hitung_target_mitigasi_dampak(self._impact_dgn_rekomendasi())
        assert hasil["penyesuaian_lahan"]["luas_bangunan_maks_m2"] == 11551.06
        assert hasil["dimensi_minimum_resapan"]["nilai"] == 19.01
        assert "menurunkan dampak" in hasil["saran_be"]

    def test_nilai_saat_ini_bangunan_rth_dari_detailed_surface_breakdown(self):
        impact = self._impact_dgn_rekomendasi(
            detailed_surface_breakdown={
                "Bangunan/Atap": {"luas_m2": 14800},
                "Taman/RTH": {"luas_m2": 100},
            }
        )
        hasil = hitung_target_mitigasi_dampak(impact)
        assert hasil["luas_bangunan_saat_ini_m2"] == 14800
        assert hasil["luas_rth_saat_ini_m2"] == 100

    def test_tanpa_detailed_surface_breakdown_tetap_jalan_tanpa_nilai_saat_ini(self):
        hasil = hitung_target_mitigasi_dampak(self._impact_dgn_rekomendasi())
        assert "luas_bangunan_saat_ini_m2" not in hasil

    def test_rekomendasi_mitigasi_none_fallback_ke_logika_band_lama(self):
        # Fixture lama / BE belum kirim rekomendasi_mitigasi utk kasus ini -> perilaku band lama.
        impact = self._impact_dgn_rekomendasi(rekomendasi_mitigasi=None)
        hasil = hitung_target_mitigasi_dampak(impact)
        assert hasil["runoff_change_index_maks"] == 2.5
        assert "penyesuaian_lahan" not in hasil

    def test_rekomendasi_mitigasi_tanpa_target_indeks_maks_fallback_ke_band_lama(self):
        # rekomendasi_mitigasi ADA tapi tak lengkap (mis. respons null-heavy) -> jangan setengah pakai.
        impact = self._impact_dgn_rekomendasi(rekomendasi_mitigasi={"saran": "x"})
        hasil = hitung_target_mitigasi_dampak(impact)
        assert hasil["runoff_change_index_maks"] == 2.5
        assert "saran_be" not in hasil


class TestPilihTargetMitigasiDampak:
    def test_ambil_runoff_change_index_maks(self):
        assert pilih_target_mitigasi_dampak({"runoff_change_index_maks": 2.5}) == 2.5

    def test_dict_kosong_return_none(self):
        assert pilih_target_mitigasi_dampak({}) is None


class TestFormatTargetParameter:
    """Sejak 2026-09-21: angka bilangan bulat TANPA nol berlebihan ("60.0"->"60"), angka pecahan
    dibulatkan maks 2 desimal ("_fmt_angka", dipakai jg oleh langkah_konkret) — item permintaan
    user: rapikan tampilan angka bagi reviewer non-teknis."""

    def test_kdb(self):
        hasil = format_target_parameter({"target_kdb": 60.0, "selisih": 10.0, "footprint_maks_m2": 510.0})
        assert hasil == "KDB harus turun ke maksimal 60% (selisih 10 poin dari usulan) → luas lantai dasar bangunan maksimal 510 m²"

    def test_kdb_angka_pecahan_dibulatkan_2_desimal(self):
        hasil = format_target_parameter({"target_kdb": 51.98273913, "selisih": 26.09509778650137})
        assert hasil == "KDB harus turun ke maksimal 51.98% (selisih 26.1 poin dari usulan)"

    def test_kdb_tanpa_footprint_m2(self):
        hasil = format_target_parameter({"target_kdb": 60.0, "selisih": 10.0})
        assert hasil == "KDB harus turun ke maksimal 60% (selisih 10 poin dari usulan)"

    def test_klb(self):
        hasil = format_target_parameter({"target_klb": 1.8, "selisih": 0.2, "luas_lantai_maks_m2": 900.0})
        assert "KLB harus turun ke maksimal 1.8" in hasil
        assert "luas total lantai bangunan maksimal 900 m²" in hasil

    def test_kdh_tanpa_rth_kurang(self):
        hasil = format_target_parameter({"target_kdh": 30.0, "selisih": 5.0, "rth_dibutuhkan_m2": 255.0})
        assert "KDH harus naik ke minimal 30%" in hasil
        assert "kurang" not in hasil.split("RTH dibutuhkan")[1]  # tak ada klausa "masih kurang"

    def test_bentuk_tak_dikenal_fallback_str(self):
        assert format_target_parameter({"aneh": 1}) == "{'aneh': 1}"


class TestBangunLangkahKonkretIntensitas:
    def test_satu_parameter_melanggar(self):
        parameter = {"kdb": {"usulan": 40.0, "satuan": "persen"}}
        target_map = {"kdb": {"target_kdb": 10.0, "selisih": 30.0, "footprint_maks_m2": 85.0}}
        hasil = bangun_langkah_konkret_intensitas(parameter, target_map)
        assert len(hasil) == 1
        assert hasil[0]["parameter"] == "KDB"
        assert hasil[0]["nilai_saat_ini"] == 40.0
        assert hasil[0]["nilai_target"] == 10.0
        assert hasil[0]["satuan"] == "persen"
        assert "luas lantai dasar bangunan maksimal 85" in hasil[0]["deskripsi"]

    def test_dua_parameter_melanggar_sekaligus(self):
        # APP-2026-8090 (live nyata): KDB & KDH melanggar BERSAMAAN — RekomendasiOutput.target
        # (pilih_target_utama_intensitas) cuma bisa kasih 1 representatif, langkah_konkret HARUS
        # kasih keduanya, bukan cuma yang pertama.
        parameter = {
            "kdb": {"usulan": 40.0, "satuan": "persen"},
            "kdh": {"usulan": 29.4, "satuan": "persen"},
        }
        target_map = {
            "kdb": {"target_kdb": 10.0, "selisih": 30.0},
            "kdh": {"target_kdh": 88.0, "selisih": 58.6},
        }
        hasil = bangun_langkah_konkret_intensitas(parameter, target_map)
        assert len(hasil) == 2
        assert {h["parameter"] for h in hasil} == {"KDB", "KDH"}

    def test_target_map_kosong_return_kosong(self):
        assert bangun_langkah_konkret_intensitas({"kdb": {"usulan": 40.0}}, {}) == []

    def test_nilai_target_nol_tetap_muncul_bukan_dianggap_kosong(self):
        # Regresi APP-2026-8913: ambang_maks=0 (bug data BE) -> target_kdb=0 — nilai 0 itu FALSY
        # di Python, JANGAN sampai `or`-chain bikin nilai_target salah jadi None.
        parameter = {"kdb": {"usulan": 40.0, "satuan": "persen"}}
        target_map = {"kdb": {"target_kdb": 0.0, "selisih": 40.0}}
        hasil = bangun_langkah_konkret_intensitas(parameter, target_map)
        assert hasil[0]["nilai_target"] == 0.0


class TestBangunLangkahKonkretDampak:
    def test_ada_target_mitigasi(self):
        target_mitigasi = {"runoff_change_index_maks": 1.5, "index_saat_ini": 2.8, "kategori_target": "Sedang"}
        hasil = bangun_langkah_konkret_dampak(target_mitigasi)
        assert len(hasil) == 1
        assert hasil[0]["parameter"] == "Indeks Limpasan (Runoff)"
        assert hasil[0]["nilai_saat_ini"] == 2.8
        assert hasil[0]["nilai_target"] == 1.5
        assert "Sedang" in hasil[0]["deskripsi"]

    def test_target_mitigasi_kosong_return_kosong(self):
        assert bangun_langkah_konkret_dampak({}) == []
        assert bangun_langkah_konkret_dampak({"perlu_mitigasi": False, "arah": []}) == []

    def test_rincian_be_hasilkan_3_item_konkret_bukan_1_item_abstrak(self):
        # APP-2026-8025/-5067: BE kirim rincian lengkap -> 3 item KONKRET (m²/dimensi resapan),
        # bukan 1 item abstrak "runoff_change_index" lama.
        target_mitigasi = {
            "runoff_change_index_maks": 2.5, "index_saat_ini": 2.923, "kategori_target": "Sedang",
            "penyesuaian_lahan": {
                "luas_bangunan_maks_m2": 11551.06, "luas_rth_min_m2": 3850.35,
                "kdb_maks_persen": 75, "kdh_min_persen": 25,
            },
            "dimensi_minimum_resapan": {"nilai": 19.01, "satuan": "m³"},
            "luas_bangunan_saat_ini_m2": 14800, "luas_rth_saat_ini_m2": 100,
        }
        hasil = bangun_langkah_konkret_dampak(target_mitigasi)
        assert len(hasil) == 3
        parameter = {h["parameter"] for h in hasil}
        assert parameter == {"Luas Bangunan (Atap)", "Luas RTH", "Dimensi Minimum Sumur Resapan"}

        bangunan = next(h for h in hasil if h["parameter"] == "Luas Bangunan (Atap)")
        assert bangunan["nilai_saat_ini"] == 14800
        assert bangunan["nilai_target"] == 11551.06
        assert bangunan["satuan"] == "m²"
        assert "75%" in bangunan["deskripsi"]

        rth = next(h for h in hasil if h["parameter"] == "Luas RTH")
        assert rth["nilai_saat_ini"] == 100
        assert rth["nilai_target"] == 3850.35

        sumur = next(h for h in hasil if h["parameter"] == "Dimensi Minimum Sumur Resapan")
        assert sumur["nilai_target"] == 19.01
        assert sumur["satuan"] == "m³"

    def test_rincian_be_tanpa_nilai_saat_ini_tetap_muncul_dgn_none(self):
        # detailed_surface_breakdown tak dikirim BE -> nilai_saat_ini None, item TETAP tampil
        # (bukan dibuang) — reviewer tetap dapat angka target walau tanpa pembanding "saat ini".
        target_mitigasi = {
            "runoff_change_index_maks": 2.5, "index_saat_ini": 2.923, "kategori_target": "Sedang",
            "penyesuaian_lahan": {"luas_bangunan_maks_m2": 11551.06, "luas_rth_min_m2": 3850.35},
        }
        hasil = bangun_langkah_konkret_dampak(target_mitigasi)
        assert len(hasil) == 2
        assert all(h["nilai_saat_ini"] is None for h in hasil)

    def test_tanpa_rincian_be_fallback_ke_item_abstrak_lama(self):
        # target_mitigasi TANPA penyesuaian_lahan/dimensi_minimum_resapan (fixture lama/band lama)
        # -> perilaku lama tak berubah (1 item abstrak runoff_change_index).
        target_mitigasi = {"runoff_change_index_maks": 1.5, "index_saat_ini": 2.8, "kategori_target": "Sedang"}
        hasil = bangun_langkah_konkret_dampak(target_mitigasi)
        assert len(hasil) == 1
        assert hasil[0]["parameter"] == "Indeks Limpasan (Runoff)"
