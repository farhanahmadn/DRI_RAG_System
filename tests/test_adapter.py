import json
from pathlib import Path

import pytest

from app.adapter import _bangun_poin_intensitas, adaptasi, cek_konsistensi_intensitas
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

    def test_rekomendasi_sistem_setuju(self):
        hasil = adaptasi(_muat_assessment("l2_sample_lolos.json"))
        assert hasil.rekomendasi_sistem == "Setuju"


class TestFixtureLolosBersyarat:
    def test_parse_tanpa_error(self):
        _muat_assessment("l2_sample_lolos_bersyarat.json")

    def test_status_dan_parameter_konsisten(self):
        # Fixture ini sebelumnya punya intensitas.lolos=True yang kontradiktif dgn KDB melampaui
        # ambang (bug back-end asli, sudah dilaporkan terpisah) — sekarang fixture sudah diperbaiki
        # jadi internal-konsisten.
        assessment = _muat_assessment("l2_sample_lolos_bersyarat.json")
        assert assessment.gate_hukum.tahapan.intensitas.lolos is False
        assert assessment.gate_hukum.tahapan.intensitas.status == "MELAMPAUI_BATAS"

        hasil = adaptasi(assessment)
        poin_intensitas = next(p for p in hasil.poin if p.poin_id == "intensitas")
        assert poin_intensitas.status == "MELAMPAUI_BATAS"
        assert poin_intensitas.fakta["parameter"]["kdb"]["memenuhi"] is False
        assert cek_konsistensi_intensitas(assessment) == []

    def test_rekomendasi_sistem_setuju_bersyarat(self):
        hasil = adaptasi(_muat_assessment("l2_sample_lolos_bersyarat.json"))
        assert hasil.rekomendasi_sistem == "Setuju Bersyarat"

    def test_status_poin_ikut_status_bukan_lolos_walau_dibuat_berbeda(self):
        # Kasus sintetis (bukan fixture kontradiktif) — buktikan adapter tetap memakai `.status`
        # utk menentukan status poin, terlepas dari nilai `.lolos`, bila suatu saat keduanya beda lagi.
        assessment = _muat_assessment("l2_sample_lolos_bersyarat.json")
        intensitas_dibalik = assessment.gate_hukum.tahapan.intensitas.model_copy(
            update={"lolos": not assessment.gate_hukum.tahapan.intensitas.lolos}
        )
        tahapan_dibalik = assessment.gate_hukum.tahapan.model_copy(update={"intensitas": intensitas_dibalik})
        gate_dibalik = assessment.gate_hukum.model_copy(update={"tahapan": tahapan_dibalik})
        assessment_dibalik = assessment.model_copy(update={"gate_hukum": gate_dibalik})

        poin_asli = _bangun_poin_intensitas(assessment)
        poin_dibalik = _bangun_poin_intensitas(assessment_dibalik)
        assert poin_asli.status == poin_dibalik.status == "MELAMPAUI_BATAS"


class TestCekKonsistensiIntensitas:
    def test_fixture_lolos_konsisten(self):
        assert cek_konsistensi_intensitas(_muat_assessment("l2_sample_lolos.json")) == []

    def test_fixture_lolos_bersyarat_konsisten(self):
        assert cek_konsistensi_intensitas(_muat_assessment("l2_sample_lolos_bersyarat.json")) == []

    def test_melampaui_batas_tapi_semua_parameter_memenuhi(self):
        assessment = _muat_assessment("l2_sample_lolos_bersyarat.json")
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
        assessment = _muat_assessment("l2_sample_lolos_bersyarat.json")
        gate = assessment.gate_hukum.model_copy(update={"final_gate_status": "Lolos"})
        assessment = assessment.model_copy(update={"gate_hukum": gate})

        masalah = cek_konsistensi_intensitas(assessment)
        assert any("harusnya Lolos Bersyarat" in m for m in masalah)


@pytest.mark.parametrize("nama_file", ["l2_sample_lolos.json", "l2_sample_lolos_bersyarat.json"])
def test_poin_dampak_tidak_dinilai_fallback(nama_file):
    assessment = _muat_assessment(nama_file)
    assessment.impact_assessment.dinilai = False
    assessment.impact_assessment.impact_category = None

    hasil = adaptasi(assessment)
    poin_dampak = next(p for p in hasil.poin if p.poin_id == "dampak")
    assert poin_dampak.status == "Tidak Dinilai"
