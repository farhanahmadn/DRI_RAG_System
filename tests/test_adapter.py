import json
from pathlib import Path

import pytest

from app.adapter import _bangun_poin_intensitas, deteksi_fallback_itbx, adaptasi, cek_konsistensi_intensitas
from app.schemas import L2Assessment

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _muat_assessment(nama_file: str) -> L2Assessment:
    payload = json.loads((FIXTURES_DIR / nama_file).read_text(encoding="utf-8"))
    return L2Assessment.model_validate(payload["data"])


# Normalisasi kategori dampak dipindah & diuji di tests/test_calculator.py
# (app.reasoning.calculator.normalisasi_kategori_dampak) — adapter.py sekarang cuma memakainya.


class TestFixtureLolos:
    def test_parse_tanpa_error(self):
        _muat_assessment("l2_sample_lolos.json")

    def test_tiga_poin_dengan_status_benar(self):
        hasil = adaptasi(_muat_assessment("l2_sample_lolos.json"))
        poin_by_id = {p.poin_id: p for p in hasil.poin}

        assert set(poin_by_id) == {"itbx", "intensitas", "dampak"}
        assert poin_by_id["itbx"].status == "I"
        assert poin_by_id["itbx"].tipe_rekomendasi == "kategorikal"
        assert poin_by_id["intensitas"].status == "MEMENUHI_SYARAT"
        assert poin_by_id["intensitas"].tipe_rekomendasi == "numerik"
        assert poin_by_id["dampak"].status == "Sedang"
        assert poin_by_id["dampak"].tipe_rekomendasi == "numerik-mitigasi"

    def test_ketiga_poin_bawa_zona_pemohon(self):
        # APP-2026-6191: zona pemohon (apa adanya dari lokasi.rdtr_zone) HARUS ikut ke ketiga poin —
        # dipakai app/reasoning/generator.py::ambil_chunks_pendukung utk filter keluarga zona.
        assessment = _muat_assessment("l2_sample_lolos.json")
        hasil = adaptasi(assessment)
        poin_by_id = {p.poin_id: p for p in hasil.poin}
        for poin_id in ("itbx", "intensitas", "dampak"):
            assert poin_by_id[poin_id].zona == assessment.lokasi.rdtr_zone

    def test_ketiga_poin_zona_subzone_none_kalau_be_tak_kirim(self):
        # Fixture lama (sebelum APP-2026-8090) tak punya rdtr_subzone -> None, BUKAN error/tebakan.
        assessment = _muat_assessment("l2_sample_lolos.json")
        assert assessment.lokasi.rdtr_subzone is None
        hasil = adaptasi(assessment)
        for p in hasil.poin:
            assert p.zona_subzone is None

    def test_ketiga_poin_bawa_zona_subzone_kalau_be_kirim(self):
        # APP-2026-8090: BE mulai kirim rdtr_subzone (mis. "P-1") — HARUS diteruskan apa adanya
        # ke ketiga poin, dipakai app/reasoning/generator.py utk filter retrieval EXACT.
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(
            update={"lokasi": assessment.lokasi.model_copy(update={"rdtr_subzone": "P-1"})}
        )
        hasil = adaptasi(assessment)
        for p in hasil.poin:
            assert p.zona_subzone == "P-1"

    def test_rekomendasi_sistem_setuju(self):
        hasil = adaptasi(_muat_assessment("l2_sample_lolos.json"))
        assert hasil.rekomendasi_sistem == "Setuju"


class TestFixtureLolosBersyarat:
    def test_parse_tanpa_error(self):
        _muat_assessment("l2_sample_amplop_6191.json")

    def test_status_dan_parameter_konsisten(self):
        assessment = _muat_assessment("l2_sample_amplop_6191.json")
        assert assessment.gate_hukum.tahapan.intensitas.lolos is False
        assert assessment.gate_hukum.tahapan.intensitas.status == "MELAMPAUI_BATAS"

        hasil = adaptasi(assessment)
        poin_intensitas = next(p for p in hasil.poin if p.poin_id == "intensitas")
        assert poin_intensitas.status == "MELAMPAUI_BATAS"
        assert poin_intensitas.fakta["parameter"]["kdb"]["memenuhi"] is False
        assert cek_konsistensi_intensitas(assessment) == []

    def test_rekomendasi_sistem_setuju_bersyarat(self):
        hasil = adaptasi(_muat_assessment("l2_sample_amplop_6191.json"))
        assert hasil.rekomendasi_sistem == "Setuju Bersyarat"

    def test_status_poin_ikut_status_bukan_lolos_walau_dibuat_berbeda(self):
        # Kasus sintetis (bukan fixture kontradiktif) — buktikan adapter tetap memakai `.status`
        # utk menentukan status poin, terlepas dari nilai `.lolos`, bila suatu saat keduanya beda lagi.
        assessment = _muat_assessment("l2_sample_amplop_6191.json")
        intensitas_dibalik = assessment.gate_hukum.tahapan.intensitas.model_copy(
            update={"lolos": not assessment.gate_hukum.tahapan.intensitas.lolos}
        )
        tahapan_dibalik = assessment.gate_hukum.tahapan.model_copy(update={"intensitas": intensitas_dibalik})
        gate_dibalik = assessment.gate_hukum.model_copy(update={"tahapan": tahapan_dibalik})
        assessment_dibalik = assessment.model_copy(update={"gate_hukum": gate_dibalik})

        poin_asli = _bangun_poin_intensitas(assessment)
        poin_dibalik = _bangun_poin_intensitas(assessment_dibalik)
        assert poin_asli.status == poin_dibalik.status == "MELAMPAUI_BATAS"

    def test_poin_dampak_dapat_target_mitigasi_kuantitatif(self):
        # Fixture nyata: impact_category=TINGGI, runoff_change_index=2.85, threshold_bands standar
        # -> target_mitigasi WAJIB terisi (bukan {}), angka persis sesuai kasus live yang mendasari
        # perbaikan ini (analisis output APP-2026-6191).
        hasil = adaptasi(_muat_assessment("l2_sample_amplop_6191.json"))
        poin_dampak = next(p for p in hasil.poin if p.poin_id == "dampak")
        assert poin_dampak.status == "Tinggi"
        assert poin_dampak.fakta["target_mitigasi"] == {
            "runoff_change_index_maks": 2.5,
            "kategori_target": "Sedang",
            "index_saat_ini": 2.85,
        }


class TestFixtureTidakLolos:
    def test_parse_tanpa_error(self):
        _muat_assessment("l2_sample_tidak_lolos.json")

    def test_intensitas_reason_null_diterima(self):
        # Back-end kirim intensitas.reason=null saat intensitas BUKAN decisive_stage (gate sudah
        # short-circuit di ITBX) — schemas.py IntensitasTahap.reason harus nullable utk kasus ini.
        assessment = _muat_assessment("l2_sample_tidak_lolos.json")
        assert assessment.gate_hukum.tahapan.intensitas.reason is None

    def test_itbx_x_reason_ambigu_memicu_fallback_heuristik(self):
        # reason back-end: "Gagal karena kegiatan dilarang (X) atau tidak ditemukan di zona ..."
        # — teks itu sendiri ambigu antara "dilarang" vs "tidak ditemukan", mengandung kata kunci
        # "tidak ditemukan" -> heuristik fallback_data_kosong AKTIF (respons yg tepat: lebih baik
        # menandai perlu kehati-hatian ekstra drpd back-end sendiri tak yakin).
        assessment = _muat_assessment("l2_sample_tidak_lolos.json")
        itbx = assessment.gate_hukum.tahapan.itbx
        assert itbx.status == "X"
        assert deteksi_fallback_itbx(itbx.reason) is True

        hasil = adaptasi(assessment)
        poin_itbx = next(p for p in hasil.poin if p.poin_id == "itbx")
        assert poin_itbx.fakta["fallback_data_kosong"] is True

    def test_tiga_poin_dengan_status_benar(self):
        hasil = adaptasi(_muat_assessment("l2_sample_tidak_lolos.json"))
        poin_by_id = {p.poin_id: p for p in hasil.poin}

        assert poin_by_id["itbx"].status == "X"
        assert poin_by_id["intensitas"].status == "MEMENUHI_SYARAT"
        assert poin_by_id["dampak"].status == "Tidak Dinilai"  # impact_assessment.dinilai=False

    def test_rekomendasi_sistem_tidak_setuju(self):
        hasil = adaptasi(_muat_assessment("l2_sample_tidak_lolos.json"))
        assert hasil.rekomendasi_sistem == "Tidak Setuju"

    def test_cek_konsistensi_intensitas_tetap_konsisten(self):
        # ITBX yang menggagalkan gate, tapi intensitas sendiri (MEMENUHI_SYARAT, semua parameter
        # patuh) tetap konsisten secara internal — cek_konsistensi_intensitas cuma soal intensitas.
        assert cek_konsistensi_intensitas(_muat_assessment("l2_sample_tidak_lolos.json")) == []


class TestCekKonsistensiIntensitas:
    def test_fixture_lolos_konsisten(self):
        assert cek_konsistensi_intensitas(_muat_assessment("l2_sample_lolos.json")) == []

    def test_fixture_lolos_bersyarat_konsisten(self):
        assert cek_konsistensi_intensitas(_muat_assessment("l2_sample_amplop_6191.json")) == []

    def test_melampaui_batas_tapi_semua_parameter_memenuhi(self):
        assessment = _muat_assessment("l2_sample_amplop_6191.json")
        parameter_bersih = {
            nama: p.model_copy(update={"memenuhi": True})
            for nama, p in assessment.gate_hukum.tahapan.intensitas.parameter.items()
        }
        intensitas = assessment.gate_hukum.tahapan.intensitas.model_copy(update={"parameter": parameter_bersih})
        tahapan = assessment.gate_hukum.tahapan.model_copy(update={"intensitas": intensitas})
        gate = assessment.gate_hukum.model_copy(update={"tahapan": tahapan})
        assessment = assessment.model_copy(update={"gate_hukum": gate})

        masalah = cek_konsistensi_intensitas(assessment)
        assert any("semua parameter.memenuhi=True" in m for m in masalah)

    def test_memenuhi_syarat_tapi_ada_parameter_gagal(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        parameter_gagal = dict(assessment.gate_hukum.tahapan.intensitas.parameter)
        parameter_gagal["kdb"] = parameter_gagal["kdb"].model_copy(update={"memenuhi": False})
        intensitas = assessment.gate_hukum.tahapan.intensitas.model_copy(update={"parameter": parameter_gagal})
        tahapan = assessment.gate_hukum.tahapan.model_copy(update={"intensitas": intensitas})
        gate = assessment.gate_hukum.model_copy(update={"tahapan": tahapan})
        assessment = assessment.model_copy(update={"gate_hukum": gate})

        masalah = cek_konsistensi_intensitas(assessment)
        assert any("ada parameter.memenuhi=False" in m for m in masalah)

    def test_melampaui_batas_tapi_final_gate_status_lolos(self):
        assessment = _muat_assessment("l2_sample_amplop_6191.json")
        gate = assessment.gate_hukum.model_copy(update={"final_gate_status": "Lolos"})
        assessment = assessment.model_copy(update={"gate_hukum": gate})

        masalah = cek_konsistensi_intensitas(assessment)
        assert any("harusnya Lolos Bersyarat" in m for m in masalah)


@pytest.mark.parametrize(
    "nama_file", ["l2_sample_lolos.json", "l2_sample_amplop_6191.json", "l2_sample_tidak_lolos.json"]
)
def test_poin_dampak_tidak_dinilai_fallback(nama_file):
    assessment = _muat_assessment(nama_file)
    assessment.impact_assessment.dinilai = False
    assessment.impact_assessment.impact_category = None

    hasil = adaptasi(assessment)
    poin_dampak = next(p for p in hasil.poin if p.poin_id == "dampak")
    assert poin_dampak.status == "Tidak Dinilai"


class TestFixtureItbxBersyarat:
    """APP-2026-8913: itbx status "B" (kegiatan bersyarat) + threshold_bands berformat deskriptif
    ("index < 1.5", bukan angka murni) + c_coefficients campur label string."""

    def test_parse_tanpa_error(self):
        _muat_assessment("l2_sample_amplop_8913.json")

    def test_itbx_status_b_diterima_dan_dipetakan(self):
        assessment = _muat_assessment("l2_sample_amplop_8913.json")
        hasil = adaptasi(assessment)
        poin_itbx = next(p for p in hasil.poin if p.poin_id == "itbx")
        assert poin_itbx.status == "B"
        assert poin_itbx.fakta["kegiatan_bersyarat"] == ["Reparasi dan Perawatan Mobil"]

    def test_dampak_target_mitigasi_kosong_krn_kategori_rendah(self):
        # kategori "Rendah" -> tak ada target mitigasi (hanya Tinggi/Sangat Tinggi yg dapat target).
        assessment = _muat_assessment("l2_sample_amplop_8913.json")
        hasil = adaptasi(assessment)
        poin_dampak = next(p for p in hasil.poin if p.poin_id == "dampak")
        assert poin_dampak.fakta["target_mitigasi"] == {}


class TestTahapanIntensitasAbsen:
    """APP-2026-003: gate berhenti di ITBX (status X) -> back-end OMIT `tahapan.intensitas` sama
    sekali (bukan kirim objek dgn reason=null spt sebelumnya, lihat l2_sample_tidak_lolos.json)."""

    def test_poin_intensitas_tetap_terbentuk_status_tidak_dinilai(self):
        assessment = _muat_assessment("l2_sample_itbx_x_tanpa_intensitas.json")
        hasil = adaptasi(assessment)
        assert len(hasil.poin) == 3  # itbx, intensitas, dampak — SELALU 3 poin, bukan diomit.
        poin_intensitas = next(p for p in hasil.poin if p.poin_id == "intensitas")
        assert poin_intensitas.status == "Tidak Dinilai"
        assert poin_intensitas.fakta == {"dinilai": False}
        assert poin_intensitas.tipe_rekomendasi == "numerik"

    def test_bangun_poin_intensitas_langsung(self):
        assessment = _muat_assessment("l2_sample_itbx_x_tanpa_intensitas.json")
        poin = _bangun_poin_intensitas(assessment)
        assert poin.status == "Tidak Dinilai"
        assert poin.dasar_hukum == []

    def test_cek_konsistensi_intensitas_kosong_kalau_absen(self):
        assessment = _muat_assessment("l2_sample_itbx_x_tanpa_intensitas.json")
        assert cek_konsistensi_intensitas(assessment) == []

    def test_adaptasi_tidak_error_dgn_intensitas_absen(self):
        assessment = _muat_assessment("l2_sample_itbx_x_tanpa_intensitas.json")
        hasil = adaptasi(assessment)
        poin_itbx = next(p for p in hasil.poin if p.poin_id == "itbx")
        assert poin_itbx.status == "X"
        assert hasil.rekomendasi_sistem == "Tidak Setuju"


class TestFixtureAmplop8090:
    """APP-2026-8090: payload real pertama dgn rdtr_subzone terisi — itbx status "I" (diizinkan,
    tak butuh keterangan_ketentuan) + intensitas MELAMPAUI_BATAS ganda (KDB & KDH, ambang nyata
    bukan artefak 0 spt fixture 8913 sebelumnya)."""

    def test_parse_tanpa_error(self):
        _muat_assessment("l2_sample_amplop_8090.json")

    def test_zona_subzone_p1_sampai_ke_semua_poin(self):
        assessment = _muat_assessment("l2_sample_amplop_8090.json")
        hasil = adaptasi(assessment)
        for p in hasil.poin:
            assert p.zona == "Zona Pertanian"
            assert p.zona_subzone == "P-1"

    def test_intensitas_dua_parameter_melanggar(self):
        assessment = _muat_assessment("l2_sample_amplop_8090.json")
        hasil = adaptasi(assessment)
        poin_intensitas = next(p for p in hasil.poin if p.poin_id == "intensitas")
        assert poin_intensitas.status == "MELAMPAUI_BATAS"
        assert poin_intensitas.fakta["parameter"]["kdb"]["memenuhi"] is False
        assert poin_intensitas.fakta["parameter"]["kdh"]["memenuhi"] is False
        assert poin_intensitas.fakta["parameter"]["klb"]["memenuhi"] is True
        assert cek_konsistensi_intensitas(assessment) == []  # data BE konsisten kali ini

    def test_itbx_status_i_tak_perlu_keterangan_ketentuan(self):
        assessment = _muat_assessment("l2_sample_amplop_8090.json")
        hasil = adaptasi(assessment)
        poin_itbx = next(p for p in hasil.poin if p.poin_id == "itbx")
        assert poin_itbx.status == "I"

    def test_rekomendasi_sistem_setuju_bersyarat(self):
        hasil = adaptasi(_muat_assessment("l2_sample_amplop_8090.json"))
        assert hasil.rekomendasi_sistem == "Setuju Bersyarat"
