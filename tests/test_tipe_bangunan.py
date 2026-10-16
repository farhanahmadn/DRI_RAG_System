"""Tipe bangunan di gate dampak hidrologi (rapat 2026-10-16) dan payload uji set kedua."""

import json
from pathlib import Path

import pytest

from app.adapter import adaptasi
from app.ketentuan_khusus import tentukan_kawasan_resapan
from app.reasoning import assemble
from app.reasoning.guardrail import _pastikan_tipe_bangunan
from app.reasoning.prompts import _bangun_fakta_dampak
from app.schemas import L2Envelope, PoinKonteks, PoinOutput, RekomendasiOutput
from app.tipe_bangunan import ambil_tipe_bangunan, frasa_tipe_bangunan

DIR = Path(__file__).parent


def _muat(path: Path):
    return L2Envelope.model_validate(json.loads(path.read_text(encoding="utf-8"))).data


@pytest.fixture(autouse=True)
def _wilayah(monkeypatch):
    monkeypatch.setenv("RETRIEVER_WILAYAH", "Sleman Tengah")


class TestAmbilTipeBangunan:
    def test_dari_payload_be_0014(self):
        a = _muat(DIR / "fixtures" / "l2_sample_amplop_0014.json")
        assert ambil_tipe_bangunan(a) == {
            "kegiatan": "Rumah Tinggal", "kbli": "Rumah", "kelas_atap": "Perdagangan Sekeliling Pusat Kota",
            "jumlah_lantai": 2}

    def test_kelas_dari_breakdown_bila_c_coefficients_kosong(self):
        a = _muat(DIR / "fixtures" / "l2_sample_amplop_0014.json")
        a = a.model_copy(update={"impact_assessment": a.impact_assessment.model_copy(update={"c_coefficients": None})})
        assert ambil_tipe_bangunan(a)["kelas_atap"] == "Perdagangan Sekeliling Pusat Kota"

    def test_frasa(self):
        assert frasa_tipe_bangunan({"kegiatan": "Pergudangan", "kelas_atap": "Industri"}) == \
            " untuk kegiatan Pergudangan (kelas bangunan: Industri)"
        assert frasa_tipe_bangunan({"kelas_atap": "Industri"}) == " untuk bangunan (kelas bangunan: Industri)"
        assert frasa_tipe_bangunan({}) == ""


class TestMasukKeDampak:
    def setup_method(self):
        self.a = _muat(DIR / "fixtures" / "l2_sample_amplop_0014.json")
        self.dampak = {p.poin_id: p for p in adaptasi(self.a).poin}["dampak"]

    def test_ada_di_fakta_dan_prompt(self):
        assert self.dampak.fakta["tipe_bangunan"]["kegiatan"] == "Rumah Tinggal"
        teks = "\n".join(_bangun_fakta_dampak(self.dampak))
        assert "TIPE_BANGUNAN" in teks and "Rumah Tinggal" in teks and "Perdagangan Sekeliling Pusat Kota" in teks
        assert "WAJIB sebut jenis kegiatan" in teks and "2 lantai" in teks

    def test_ada_di_ringkasan_dampak_deterministik(self):
        r = assemble._rakit_ringkasan_dampak(self.a)
        assert r.kalimat == ("Dampak terhadap lingkungan (hidrologi) tergolong Rendah untuk kegiatan Rumah Tinggal "
                             "(kelas bangunan: Perdagangan Sekeliling Pusat Kota).")

    def test_tanpa_data_ringkasan_tidak_berubah(self):
        a = self.a.model_copy(update={"impact_assessment": self.a.impact_assessment.model_copy(
            update={"c_coefficients": None, "detailed_surface_breakdown": None})})
        itbx = a.gate_hukum.tahapan.itbx.model_copy(update={"kegiatan_diusulkan": None, "kbli_diusulkan": None})
        a = a.model_copy(update={"gate_hukum": a.gate_hukum.model_copy(
            update={"tahapan": a.gate_hukum.tahapan.model_copy(update={"itbx": itbx, "intensitas": None})})})
        assert assemble._rakit_ringkasan_dampak(a).kalimat == "Dampak terhadap lingkungan (hidrologi) tergolong Rendah."


class TestPastikanTipeBangunan:
    def _poin(self, kegiatan="Rumah Tinggal", kelas="Perumahan Tunggal"):
        return PoinKonteks(poin_id="dampak", kategori="D", tipe_rekomendasi="numerik-mitigasi", status="Rendah",
                           fakta={"dinilai": True, "tipe_bangunan": {"kegiatan": kegiatan, "kelas_atap": kelas}})

    def _out(self, pendek, panjang):
        return PoinOutput(poin_id="dampak", kategori="K", status="Rendah", reasoning_pendek=pendek,
                          reasoning_panjang=panjang, sitasi=[],
                          rekomendasi=RekomendasiOutput(tipe="numerik-mitigasi", saran="s"), low_confidence=False)

    def test_lupa_menyebut_ditambahkan_deterministik(self):
        hasil = _pastikan_tipe_bangunan(self._out("Dampak rendah.", "Indeks limpasan rendah."), self._poin())
        assert hasil.reasoning_panjang.endswith(
            "Tipe bangunan yang diusulkan adalah Rumah Tinggal (kelas bangunan: Perumahan Tunggal), yang "
            "memengaruhi koefisien limpasan atap pada perhitungan dampak.")

    def test_sudah_disebut_tak_diubah(self):
        out = self._out("Untuk rumah tinggal, dampak rendah.", "Indeks limpasan rendah.")
        assert _pastikan_tipe_bangunan(out, self._poin()) == out

    def test_satu_kata_bermakna_cukup(self):
        out = self._out("p", "Kegiatan penyediaan akomodasi menambah atap.")
        assert _pastikan_tipe_bangunan(out, self._poin("Penyediaan Akomodasi Lainnya", None)) == out

    def test_poin_lain_dan_tanpa_data_tak_tersentuh(self):
        out = self._out("p", "q")
        poin_itbx = PoinKonteks(poin_id="itbx", kategori="I", tipe_rekomendasi="kategorikal", status="B", fakta={})
        assert _pastikan_tipe_bangunan(out, poin_itbx) == out
        assert _pastikan_tipe_bangunan(out, PoinKonteks(
            poin_id="dampak", kategori="D", tipe_rekomendasi="numerik-mitigasi", status="Rendah", fakta={})) == out


SET2 = [
    ("hidrologi_set2_1_rendah_resapan.json", True, "Rendah", "Rumah Tinggal", "I"),
    ("hidrologi_set2_2_tinggi_resapan.json", True, "Tinggi", "Pergudangan", "B"),
    ("hidrologi_set2_3_rendah_nonresapan.json", False, "Rendah", "Penyediaan Makan Minum", "B"),
    ("hidrologi_set2_4_tinggi_nonresapan.json", False, "Tinggi", "Perdagangan Eceran Minimarket", "B"),
]


@pytest.mark.parametrize("nama,resapan,kategori,kegiatan,itbx", SET2)
class TestPayloadUjiSet2:
    def test_diterima_dan_konsisten(self, nama, resapan, kategori, kegiatan, itbx):
        a = _muat(DIR / "payload_uji" / nama)
        assert tentukan_kawasan_resapan(a) is resapan
        poin = {p.poin_id: p for p in adaptasi(a).poin}
        assert poin["dampak"].status == kategori and poin["itbx"].status == itbx
        assert poin["dampak"].fakta["tipe_bangunan"]["kegiatan"] == kegiatan
        assert a.impact_assessment.impact_score is None and a.meta is None

    def test_kegiatan_dan_angka_berbeda_dari_set_pertama(self, nama, resapan, kategori, kegiatan, itbx):
        set1 = {"Penyediaan Akomodasi Lainnya"}
        a = _muat(DIR / "payload_uji" / nama)
        assert a.gate_hukum.tahapan.itbx.kegiatan_diusulkan not in set1
        assert a.lokasi.luas_lahan_m2 != 1436

    def test_ringkasan_dampak_menyebut_kegiatan(self, nama, resapan, kategori, kegiatan, itbx):
        a = _muat(DIR / "payload_uji" / nama)
        assert kegiatan in assemble._rakit_ringkasan_dampak(a).kalimat
