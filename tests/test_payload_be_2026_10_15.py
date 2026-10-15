"""Payload L2 terbaru dari BE (APP-2026-0014, 2026-10-15) dan keempat payload uji hidrologi.

Perbedaan dari payload lama yang HARUS tetap diterima: tanpa `meta`, tanpa `impact_score`, label teks di
dalam `proposed_surface_details` (dulu membuat seluruh payload ditolak 422), dan field baru.
"""

import json
from pathlib import Path

import pytest

from app.adapter import adaptasi
from app.ketentuan_khusus import tentukan_kawasan_resapan
from app.reasoning import assemble
from app.schemas import L2Envelope

DIR = Path(__file__).parent


def _muat(path: Path) -> L2Envelope:
    return L2Envelope.model_validate(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture(autouse=True)
def _wilayah(monkeypatch):
    monkeypatch.setenv("RETRIEVER_WILAYAH", "Sleman Tengah")


class TestPayload0014:
    def setup_method(self):
        self.a = _muat(DIR / "fixtures" / "l2_sample_amplop_0014.json").data

    def test_diterima_tanpa_meta_dan_skor(self):
        assert self.a.meta is None
        assert self.a.impact_assessment.impact_score is None

    def test_label_teks_di_permukaan_diterima(self):
        """Regresi: `dict[str, float]` menolak payload ini dengan 422."""
        p = self.a.impact_assessment.proposed_surface_details
        assert p["kawasan_hidrologi"] == "Zona Pertanian" and p["Bangunan/Atap"] == 3.6

    def test_status_resapan_dan_zona(self):
        assert tentukan_kawasan_resapan(self.a) is True
        poin = {p.poin_id: p for p in adaptasi(self.a).poin}
        assert poin["dampak"].kawasan_resapan is True
        assert (poin["intensitas"].zona, poin["intensitas"].zona_subzone) == ("Zona Pertanian", "P-1")
        assert poin["intensitas"].fakta["dasar_ambang"] == {
            "nama_zona": "Zona Pertanian", "kode_zona": "P-1", "hierarki_jalan": "lokal", "kawasan_resapan": True}

    def test_kawasan_hidrologi_ikut_fakta_dampak(self):
        dampak = {p.poin_id: p for p in adaptasi(self.a).poin}["dampak"]
        assert dampak.fakta["kawasan_hidrologi"] == {"nama": "Zona Pertanian", "kode": "P", "status_itbx": "B"}

    def test_ringkasan_dampak_tanpa_skor(self):
        r = assemble._rakit_ringkasan_dampak(self.a)
        assert r.impact_category == "Rendah" and r.impact_score is None
        assert "None" not in r.kalimat

    def test_dampak_rendah_tetap_punya_langkah_pemeliharaan(self):
        dampak = {p.poin_id: p for p in adaptasi(self.a).poin}["dampak"]
        from app.reasoning.calculator import bangun_langkah_dampak_poin
        nama = [l["parameter"] for l in bangun_langkah_dampak_poin(dampak.fakta)]
        assert nama == ["Luas RTH", "Luas Bangunan (Atap)", "Sumur/Kolam Resapan"]


class TestSumberStatusResapan:
    def test_is_kawasan_resapan_dampak_dipakai_bila_sumber_lain_kosong(self):
        payload = json.loads((DIR / "fixtures" / "l2_sample_amplop_0014.json").read_text(encoding="utf-8"))
        lok = payload["data"]["lokasi"]
        for k in ("kawasan_resapan", "ketentuan_khusus", "blok", "blok_list", "swp"):
            lok.pop(k, None)
        payload["data"]["gate_hukum"]["tahapan"]["intensitas"].pop("kawasan_resapan")
        a = L2Envelope.model_validate(payload).data
        assert tentukan_kawasan_resapan(a) is True  # dari impact_assessment.is_kawasan_resapan

    def test_blok_resapan_cocok_dengan_flag_be(self):
        """I.B.3 ada di daftar Pasal 53 ayat (1): turunan blok dan flag BE harus sepakat."""
        from app.ketentuan_khusus import BLOK_RESAPAN_SLEMAN_TENGAH
        assert "I.B.3" in BLOK_RESAPAN_SLEMAN_TENGAH


@pytest.mark.parametrize("nama,resapan,kategori", [
    ("hidrologi_1_rendah_resapan.json", True, "Rendah"),
    ("hidrologi_2_tinggi_resapan.json", True, "Tinggi"),
    ("hidrologi_3_rendah_nonresapan.json", False, "Rendah"),
    ("hidrologi_4_tinggi_nonresapan.json", False, "Tinggi"),
])
class TestEmpatPayloadUji:
    def test_bentuk_mengikuti_be_terbaru(self, nama, resapan, kategori):
        mentah = json.loads((DIR / "payload_uji" / nama).read_text(encoding="utf-8"))
        data = mentah["data"]
        assert "meta" not in data and "impact_score" not in data["impact_assessment"]
        assert data["is_revision"] is False and data["revision_count"] == 0
        assert data["gate_hukum"]["is_multi_zone"] is False and len(data["gate_hukum"]["intensitas_zones"]) == 1
        assert data["impact_assessment"]["is_kawasan_resapan"] is resapan
        assert data["lokasi"]["kawasan_resapan"] is resapan
        assert data["lokasi"]["ketentuan_khusus"]["kawasan_resapan_air"] is resapan

    def test_diterima_dan_diproses(self, nama, resapan, kategori):
        a = _muat(DIR / "payload_uji" / nama).data
        assert tentukan_kawasan_resapan(a) is resapan
        poin = {p.poin_id: p for p in adaptasi(a).poin}
        assert poin["dampak"].status == kategori
        assert poin["dampak"].kawasan_resapan is resapan
        assert poin["dampak"].fakta["kawasan_hidrologi"]["nama"] == "Zona Perumahan"
        assert assemble._rakit_ringkasan_dampak(a).impact_score is None
