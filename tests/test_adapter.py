import json
from pathlib import Path

import pytest

from app.adapter import _normalisasi_kategori_dampak, adaptasi
from app.schemas import L2Assessment

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _muat_assessment(nama_file: str) -> L2Assessment:
    payload = json.loads((FIXTURES_DIR / nama_file).read_text(encoding="utf-8"))
    return L2Assessment.model_validate(payload["data"])


class TestNormalisasiKategoriDampak:
    def test_uppercase_ke_title_case(self):
        assert _normalisasi_kategori_dampak("SEDANG") == "Sedang"

    def test_multi_kata(self):
        assert _normalisasi_kategori_dampak("SANGAT TINGGI") == "Sangat Tinggi"

    def test_none_tetap_none(self):
        assert _normalisasi_kategori_dampak(None) is None

    def test_sudah_title_case_idempoten(self):
        assert _normalisasi_kategori_dampak("Sangat Tinggi") == "Sangat Tinggi"


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

    def test_status_intensitas_dari_status_bukan_lolos(self):
        # Fixture ini punya intensitas.lolos=True yang kontradiktif dgn KDB melampaui ambang (bug
        # back-end) — adapter harus tetap melaporkan MELAMPAUI_BATAS berdasar `status`, bukan `.lolos`.
        assessment = _muat_assessment("l2_sample_lolos_bersyarat.json")
        assert assessment.gate_hukum.tahapan.intensitas.lolos is True
        assert assessment.gate_hukum.tahapan.intensitas.status == "MELAMPAUI_BATAS"

        hasil = adaptasi(assessment)
        poin_intensitas = next(p for p in hasil.poin if p.poin_id == "intensitas")
        assert poin_intensitas.status == "MELAMPAUI_BATAS"
        assert poin_intensitas.fakta["parameter"]["kdb"]["memenuhi"] is False

    def test_rekomendasi_sistem_setuju_bersyarat(self):
        hasil = adaptasi(_muat_assessment("l2_sample_lolos_bersyarat.json"))
        assert hasil.rekomendasi_sistem == "Setuju Bersyarat"


@pytest.mark.parametrize("nama_file", ["l2_sample_lolos.json", "l2_sample_lolos_bersyarat.json"])
def test_poin_dampak_tidak_dinilai_fallback(nama_file):
    assessment = _muat_assessment(nama_file)
    assessment.impact_assessment.dinilai = False
    assessment.impact_assessment.impact_category = None

    hasil = adaptasi(assessment)
    poin_dampak = next(p for p in hasil.poin if p.poin_id == "dampak")
    assert poin_dampak.status == "Tidak Dinilai"
