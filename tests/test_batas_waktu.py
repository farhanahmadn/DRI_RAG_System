"""Anggaran waktu per permintaan (Cloudflare memutus di 100 detik; lihat app/batas_waktu.py)."""

import json
import time
from pathlib import Path

import pytest

import app.reasoning.llm_client as lc
from app import batas_waktu
from app.batas_waktu import BatasWaktuTerlampaui
from app.reasoning import assemble
from app.reasoning import llm_client as llm_client_module
from app.reasoning.guardrail import DiagnosaPoin, generate_poin_terdiagnosis
from app.retrieval import _provider_http as provider_http
from app.retrieval.mock import MockRetriever
from app.schemas import L2Assessment, PoinKonteks

FIXTURES = Path(__file__).parent / "fixtures"


def _asesmen() -> L2Assessment:
    payload = json.loads((FIXTURES / "l2_sample_lolos.json").read_text(encoding="utf-8"))
    return L2Assessment.model_validate(payload["data"])


class TestModul:
    def test_tanpa_anggaran_tidak_ada_batas(self):
        assert batas_waktu.sisa() is None and batas_waktu.habis() is False
        batas_waktu.periksa(9999, "apa pun")  # tidak raise

    def test_anggaran_aktif_lalu_dilepas(self):
        with batas_waktu.anggaran(50):
            assert 48 < batas_waktu.sisa() <= 50
            batas_waktu.periksa(10, "cukup")
            with pytest.raises(BatasWaktuTerlampaui):
                batas_waktu.periksa(60, "kurang")
        assert batas_waktu.sisa() is None

    def test_env_dan_nonaktif(self, monkeypatch):
        monkeypatch.setenv("REASONING_DEADLINE_S", "30")
        assert batas_waktu.durasi_dari_env() == 30.0
        monkeypatch.setenv("REASONING_DEADLINE_S", "0")
        assert batas_waktu.durasi_dari_env() is None
        monkeypatch.setenv("REASONING_DEADLINE_S", "bukan-angka")
        assert batas_waktu.durasi_dari_env() == batas_waktu.DURASI_BAWAAN_S
        monkeypatch.delenv("REASONING_DEADLINE_S")
        assert batas_waktu.durasi_dari_env() == 85.0

    def test_ikut_ke_thread_lewat_copy_context(self):
        import concurrent.futures
        import contextvars
        with batas_waktu.anggaran(40), concurrent.futures.ThreadPoolExecutor(1) as ex:
            terlihat = ex.submit(contextvars.copy_context().run, batas_waktu.sisa).result()
            tanpa_konteks = ex.submit(batas_waktu.sisa).result()
        assert terlihat is not None and tanpa_konteks is None


class TestProviderHttp:
    def test_tidak_memanggil_provider_bila_sisa_kurang(self, monkeypatch):
        import httpx
        monkeypatch.setattr(httpx, "Client", lambda *a, **k: pytest.fail("tidak boleh menelepon provider"))
        with batas_waktu.anggaran(1.0), pytest.raises(BatasWaktuTerlampaui):
            provider_http.post_json("jina-uji-1", "https://x", headers={}, json_body={}, timeout_s=30, max_retries=2)


class TestLlmClient:
    def test_klien_dibatasi_sisa_anggaran(self, monkeypatch):
        monkeypatch.setattr(lc, "_rotator", None)
        monkeypatch.setenv("GROQ_API_KEYS", "k0")
        monkeypatch.setenv("LLM_MODEL", "m")
        klien = lc._get_rotator().client_utk(0)
        assert lc._batasi_client(klien) is klien  # tanpa anggaran: apa adanya
        with batas_waktu.anggaran(20):
            terbatas = lc._batasi_client(klien)
        assert terbatas is not klien
        assert terbatas.max_retries == 0 and 0 < terbatas.timeout <= 19.0

    def test_menolak_memulai_panggilan_bila_sisa_kurang(self, monkeypatch):
        monkeypatch.setattr(lc, "_rotator", None)
        monkeypatch.setenv("GROQ_API_KEYS", "k0")
        monkeypatch.setenv("LLM_MODEL", "m")
        monkeypatch.setattr(lc, "_panggil_llm_sekali", lambda *a, **k: pytest.fail("tidak boleh menelepon Groq"))
        with batas_waktu.anggaran(3), pytest.raises(BatasWaktuTerlampaui):
            lc.generate("p", {"type": "object"}, schema_name="t")


class TestGuardrail:
    def test_percobaan_dihentikan_dan_sebab_dilabeli(self, monkeypatch):
        monkeypatch.setattr(llm_client_module, "generate", lambda *a, **k: pytest.fail("anggaran sudah habis"))
        poin = PoinKonteks(poin_id="itbx", kategori="ITBX", tipe_rekomendasi="kategorikal", status="B",
                           fakta={"lolos": True, "reason": "ok"})
        with batas_waktu.anggaran(2):
            hasil, diagnosa = generate_poin_terdiagnosis(poin, MockRetriever(), _asesmen(), max_retry=2)
        assert diagnosa.sebab() == "melewati_batas_waktu" and hasil.low_confidence is True

    def test_label_sebab_langsung(self):
        d = DiagnosaPoin(poin_id="itbx", berhasil=False, percobaan=1)
        d.exception_terakhir = "BatasWaktuTerlampaui: anggaran habis"
        assert d.sebab() == "melewati_batas_waktu"


class TestEndToEnd:
    def _fake(self, tidur):
        def fake(prompt, schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=2048):
            if schema_name == "poin_reasoning":
                time.sleep(tidur)
                return {"reasoning_pendek": "p", "reasoning_panjang": "q", "sitasi": [], "saran": "s", "disclaimer": None}
            if schema_name == "narasi_rekomendasi":
                return {"paragraf_gate_intensitas": "A.", "paragraf_dampak": "B."}
            return {"langkah_berdampak": ["Langkah."], "catatan_lokasi": None}
        return fake

    def test_respons_tiba_dalam_anggaran_walau_llm_macet(self, monkeypatch):
        """LLM palsu macet 3 detik; anggaran 1 detik: respons HARUS tiba jauh sebelum 3 detik dengan
        semua poin low_confidence dan catatan batas waktu, bukan menunggu sampai LLM selesai."""
        monkeypatch.setenv("REASONING_DEADLINE_S", "1")
        monkeypatch.setattr(assemble, "_TOLERANSI_TUNGGU_S", 0.0)
        monkeypatch.setattr(assemble.llm_client, "generate", self._fake(3.0))
        monkeypatch.setattr(assemble, "log_precheck", lambda *a, **k: None)
        mulai = time.perf_counter()
        hasil = assemble.jalankan_precheck(_asesmen(), MockRetriever())
        lama = time.perf_counter() - mulai
        assert lama < 2.5, lama
        assert all(p.low_confidence for p in hasil.poin)
        assert assemble.CATATAN_BATAS_WAKTU in hasil.catatan_global
        assert hasil.low_confidence_keseluruhan is True

    def test_tanpa_hambatan_perilaku_normal_dan_sintesis_paralel(self, monkeypatch):
        monkeypatch.setenv("REASONING_DEADLINE_S", "60")
        monkeypatch.setattr(assemble.llm_client, "generate", self._fake(0.0))
        rekam = {}
        monkeypatch.setattr(assemble, "log_precheck", lambda a, r, **k: rekam.update(k))
        hasil = assemble.jalankan_precheck(_asesmen(), MockRetriever())
        assert assemble.CATATAN_BATAS_WAKTU not in hasil.catatan_global
        assert rekam["sintesis"] == {"narasi": "llm", "kesimpulan": "llm"}
        assert isinstance(rekam["durasi_detik"], float) and rekam["durasi_detik"] < 30

    def test_sintesis_dilewati_bila_sisa_anggaran_sedikit(self, monkeypatch):
        monkeypatch.setattr(assemble.llm_client, "generate", lambda *a, **k: pytest.fail("tidak boleh dipanggil"))
        catatan: dict = {}
        with batas_waktu.anggaran(5):  # < _MIN_SISA_SINTESIS_S
            narasi = assemble._rakit_narasi_rekomendasi([], "Setuju", catatan)
            assemble._rakit_kesimpulan([], "Setuju", catatan)
        assert catatan["narasi"].startswith("dilewati: sisa waktu")
        assert catatan["kesimpulan"].startswith("dilewati: sisa waktu")
        assert narasi.paragraf_gate_intensitas  # fallback deterministik tetap terisi
