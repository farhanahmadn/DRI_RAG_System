import json
import logging
from pathlib import Path

import pytest

from app.adapter import adaptasi
from app.ketentuan_khusus import (
    normalisasi_blok,
    tentukan_kawasan_resapan,
    tentukan_rawan_kekeringan,
)
from app.schemas import L2Assessment

FIXTURES = Path(__file__).parent / "fixtures"
PAYLOAD_UJI = Path(__file__).parent / "payload_uji"


def _assessment(**lokasi) -> L2Assessment:
    payload = json.loads((FIXTURES / "l2_sample_lolos.json").read_text(encoding="utf-8"))
    payload["data"]["lokasi"].update(lokasi)
    return L2Assessment.model_validate(payload["data"])


def _payload_uji(nama: str) -> L2Assessment:
    return L2Assessment.model_validate(json.loads((PAYLOAD_UJI / nama).read_text(encoding="utf-8"))["data"])


@pytest.fixture(autouse=True)
def _wilayah_tengah(monkeypatch):
    monkeypatch.setenv("RETRIEVER_WILAYAH", "Sleman Tengah")


class TestNormalisasiBlok:
    def test_variasi_penulisan(self):
        assert normalisasi_blok("I.B.2") == "I.B.2"
        assert normalisasi_blok(" i.b.2 ") == "I.B.2"
        assert normalisasi_blok("B.2") == "I.B.2"
        assert normalisasi_blok("") is None and normalisasi_blok(None) is None


class TestKawasanResapanDariBlok:
    """Pasal 53 ayat (1): SWP B I.B.1-6, SWP C I.C.1,2,3,5,6,7, SWP F I.F.3."""

    @pytest.mark.parametrize("blok", ["I.B.1", "I.B.6", "I.C.3", "I.C.7", "I.F.3", "B.2"])
    def test_blok_resapan(self, blok):
        assert tentukan_kawasan_resapan(_assessment(blok=blok)) is True

    @pytest.mark.parametrize("blok", ["I.A.3", "I.C.4", "I.F.1", "I.D.2", "I.E.1"])
    def test_blok_non_resapan(self, blok):
        assert tentukan_kawasan_resapan(_assessment(blok=blok)) is False

    def test_tanpa_blok_tak_diketahui(self):
        assert tentukan_kawasan_resapan(_assessment()) is None

    def test_blok_list_campuran_tak_dipastikan(self):
        assert tentukan_kawasan_resapan(_assessment(blok="I.B.2", blok_list=["I.B.2", "I.A.3"])) is None

    def test_blok_list_semua_resapan(self):
        assert tentukan_kawasan_resapan(_assessment(blok_list=["I.B.2", "I.C.1"])) is True

    def test_wilayah_bukan_tengah_tidak_menebak(self, monkeypatch):
        """Daftar blok milik dokumen Sleman Tengah; dokumen wilayah lain punya daftar berbeda."""
        monkeypatch.setenv("RETRIEVER_WILAYAH", "Sleman Barat")
        assert tentukan_kawasan_resapan(_assessment(blok="I.B.2")) is None


class TestPrioritasSumber:
    def test_overlay_be_menang_atas_blok(self, caplog):
        with caplog.at_level(logging.WARNING, logger="app.ketentuan_khusus"):
            hasil = tentukan_kawasan_resapan(
                _assessment(blok="I.B.2", ketentuan_khusus={"kawasan_resapan_air": False}))
        assert hasil is False
        assert "berbeda dari daftar blok" in caplog.text

    def test_flat_lokasi_kawasan_resapan(self):
        assert tentukan_kawasan_resapan(_assessment(kawasan_resapan=True)) is True

    def test_ketentuan_khusus_menang_atas_flat(self):
        a = _assessment(kawasan_resapan=True, ketentuan_khusus={"kawasan_resapan_air": False})
        assert tentukan_kawasan_resapan(a) is False

    def test_null_di_ketentuan_khusus_jatuh_ke_sumber_berikutnya(self):
        a = _assessment(kawasan_resapan=True, ketentuan_khusus={"kawasan_resapan_air": None})
        assert tentukan_kawasan_resapan(a) is True


class TestRawanKekeringan:
    def test_blok_kekeringan(self):
        assert tentukan_rawan_kekeringan(_assessment(blok="I.E.1")) is True
        assert tentukan_rawan_kekeringan(_assessment(blok="I.E.3")) is False
        assert tentukan_rawan_kekeringan(_assessment()) is None

    def test_overlay_be_menang(self):
        a = _assessment(blok="I.E.1", ketentuan_khusus={"rawan_kekeringan": False})
        assert tentukan_rawan_kekeringan(a) is False


class TestPayloadUjiKontrakBaru:
    @pytest.mark.parametrize("nama,resapan", [
        ("hidrologi_1_rendah_resapan.json", True), ("hidrologi_2_tinggi_resapan.json", True),
        ("hidrologi_3_rendah_nonresapan.json", False), ("hidrologi_4_tinggi_nonresapan.json", False),
    ])
    def test_status_resapan_dan_poin(self, nama, resapan):
        a = _payload_uji(nama)
        assert tentukan_kawasan_resapan(a) is resapan
        poin = {p.poin_id: p for p in adaptasi(a).poin}
        assert poin["dampak"].kawasan_resapan is resapan
        assert poin["dampak"].rawan_kekeringan is False
        dasar = poin["intensitas"].fakta["dasar_ambang"]
        assert dasar == {"nama_zona": "Zona Perumahan", "kode_zona": "R-3",
                         "hierarki_jalan": "lokal", "kawasan_resapan": resapan}


def _dengan_intensitas(a, **update):
    i = a.gate_hukum.tahapan.intensitas.model_copy(update=update)
    tahap = a.gate_hukum.tahapan.model_copy(update={"intensitas": i})
    return a.model_copy(update={"gate_hukum": a.gate_hukum.model_copy(update={"tahapan": tahap})})


class TestIntensitasZonaEksplisit:
    def test_zona_pemilik_ambang_dari_be_dipakai_langsung(self):
        a = _dengan_intensitas(_assessment(rdtr_zone="Zona Pertanian", rdtr_subzone="P-1"),
                               nama_zona="Zona Perumahan", kode_zona="R-3")
        p = {x.poin_id: x for x in adaptasi(a).poin}["intensitas"]
        assert (p.zona, p.zona_subzone) == ("Zona Perumahan", "R-3")

    def test_kode_zona_induk_bukan_subzona(self):
        a = _dengan_intensitas(_assessment(rdtr_zone="Zona Pertanian", rdtr_subzone="P-1"),
                               nama_zona="Zona Perumahan", kode_zona="R")
        p = {x.poin_id: x for x in adaptasi(a).poin}["intensitas"]
        assert (p.zona, p.zona_subzone) == ("Zona Perumahan", None)


def test_prompt_intensitas_memuat_dasar_ambang():
    from app.reasoning.prompts import _bangun_fakta_intensitas
    a = _payload_uji("hidrologi_3_rendah_nonresapan.json")
    p = {x.poin_id: x for x in adaptasi(a).poin}["intensitas"]
    teks = "\n".join(_bangun_fakta_intensitas(p))
    assert "Dasar pemilihan ambang" in teks
    assert "hierarki jalan lokal" in teks and "lokasi bukan kawasan resapan air" in teks


def test_kekeringan_dikutip_hanya_bila_blok_rawan_kekeringan():
    from app.reasoning.generator import _sesuaikan_dgn_status_resapan
    from app.retrieval.base import Chunk
    p50 = Chunk(id="p50-a11", level="ayat", dokumen="Perbup", pasal="50",
                teks="(11) kawasan rawan bencana kekeringan: zero delta Q dengan pembuatan sumur")

    class _R:
        def search(self, *a, **k):
            return []

    for blok, harapan in (("I.E.1", True), ("I.A.3", False), (None, False)):
        a = _assessment(**({"blok": blok} if blok else {}))
        poin = {x.poin_id: x for x in adaptasi(a).poin}["dampak"]
        ids = [c.id for c in _sesuaikan_dgn_status_resapan([p50], poin, _R(), 3)]
        assert ("p50-a11" in ids) is harapan, blok
