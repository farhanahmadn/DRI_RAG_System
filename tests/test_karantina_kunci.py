"""Karantina kunci Groq, rotasi saat `organization_restricted`, dan penghentian percobaan guardrail.

Insiden 2026-10-15: 5 dari 8 kunci kena `organization_restricted` dan kuota harian kunci lain habis;
kode lama menemukan ulang kunci mati di setiap panggilan dan menghabiskan percobaan guardrail.
"""

import httpx
import pytest
from openai import BadRequestError, RateLimitError

import app.reasoning.llm_client as lc
from app.reasoning import llm_client as llm_client_module
from app.reasoning.guardrail import DiagnosaPoin, generate_poin_terdiagnosis
from app.reasoning.llm_client import LayananTidakTersediaError
from app.retrieval.mock import MockRetriever
from app.schemas import L2Assessment, PoinKonteks

PESAN_RESTRICTED = "Organization has been restricted. Please reach out to support if you believe this was in error."


def _restricted() -> BadRequestError:
    resp = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com/x"),
                          json={"error": {"message": PESAN_RESTRICTED, "code": "organization_restricted"}})
    return BadRequestError(
        f"Error code: 400 - {{'error': {{'message': '{PESAN_RESTRICTED}', 'code': 'organization_restricted'}}}}",
        response=resp, body=None)


def _tpd() -> RateLimitError:
    pesan = "Rate limit reached for model x on tokens per day (TPD): Limit 200000, Used 199992"
    resp = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/x"), json={"error": {"message": pesan}})
    return RateLimitError(message=pesan, response=resp, body=None)


class _Msg:
    content = '{"ok": true}'


class _Completion:
    choices = [type("C", (), {"message": _Msg()})()]


@pytest.fixture
def rotator_tiga_kunci(monkeypatch):
    monkeypatch.setattr(lc, "_rotator", None)
    monkeypatch.setenv("GROQ_API_KEYS", "k0,k1,k2")
    monkeypatch.setenv("LLM_MODEL", "model-tes")
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    return lc._get_rotator()


class TestRotator:
    def test_kunci_karantina_dilewati_round_robin(self, rotator_tiga_kunci):
        r = rotator_tiga_kunci
        r.karantina(1, 100, "uji")
        urutan = [r.kunci_awal() for _ in range(6)]
        assert 1 not in urutan and set(urutan) == {0, 2}

    def test_karantina_berakhir(self, rotator_tiga_kunci, monkeypatch):
        r = rotator_tiga_kunci
        waktu = {"t": 1000.0}
        monkeypatch.setattr(lc.time, "monotonic", lambda: waktu["t"])
        r.karantina(0, 10, "uji")  # berakhir 1010
        r.karantina(1, 10, "uji")
        assert r.jumlah_tersedia() == 1
        waktu["t"] = 1011.0
        assert r.jumlah_tersedia() == 3

    def test_semua_kunci_mati_gagal_cepat(self, rotator_tiga_kunci):
        r = rotator_tiga_kunci
        for i in range(3):
            r.karantina(i, 100, "uji")
        with pytest.raises(LayananTidakTersediaError, match="3 kunci Groq"):
            r.kunci_awal()

    def test_kunci_berikutnya_melewati_yang_mati(self, rotator_tiga_kunci):
        r = rotator_tiga_kunci
        r.karantina(1, 100, "uji")
        assert r.kunci_berikutnya(0) == 2
        r.karantina(2, 100, "uji")
        assert r.kunci_berikutnya(0) is None


class TestGenerate:
    def test_organization_restricted_pindah_kunci_dan_berhasil(self, rotator_tiga_kunci, monkeypatch):
        dipakai = []

        def _stub(client, messages, json_schema, schema_name, temperature, max_tokens):
            dipakai.append(client.api_key)
            if client.api_key in ("k0", "k1"):
                raise _restricted()
            return _Completion()

        monkeypatch.setattr(lc, "_panggil_llm_sekali", _stub)
        assert lc.generate("p", {"type": "object"}, schema_name="t") == {"ok": True}
        assert dipakai == ["k0", "k1", "k2"]
        # k0 dan k1 kini dikarantina -> panggilan berikutnya langsung ke k2, tanpa menemukan ulang.
        dipakai.clear()
        lc.generate("p", {"type": "object"}, schema_name="t")
        assert dipakai == ["k2"]

    def test_semua_restricted_menjadi_layanan_tidak_tersedia(self, rotator_tiga_kunci, monkeypatch):
        monkeypatch.setattr(lc, "_panggil_llm_sekali", lambda *a, **k: (_ for _ in ()).throw(_restricted()))
        with pytest.raises(LayananTidakTersediaError):
            lc.generate("p", {"type": "object"}, schema_name="t")
        # Panggilan berikutnya gagal CEPAT dari karantina, tanpa menyentuh jaringan sama sekali.
        monkeypatch.setattr(lc, "_panggil_llm_sekali", lambda *a, **k: pytest.fail("tidak boleh menelepon Groq"))
        with pytest.raises(LayananTidakTersediaError):
            lc.generate("p", {"type": "object"}, schema_name="t")

    def test_tpd_mengarantina_kunci(self, rotator_tiga_kunci, monkeypatch):
        dipakai = []

        def _stub(client, messages, json_schema, schema_name, temperature, max_tokens):
            dipakai.append(client.api_key)
            if client.api_key == "k0":
                raise _tpd()
            return _Completion()

        monkeypatch.setattr(lc, "_panggil_llm_sekali", _stub)
        lc.generate("p", {"type": "object"}, schema_name="t")
        assert dipakai == ["k0", "k1"]
        dipakai.clear()
        for _ in range(4):
            lc.generate("p", {"type": "object"}, schema_name="t")
        assert "k0" not in dipakai  # tidak lagi dicoba selama karantina

    def test_restricted_tidak_memicu_fallback_json_object(self, rotator_tiga_kunci, monkeypatch):
        """_panggil_llm_sekali: BadRequest biasa -> fallback json_object; restricted -> langsung raise."""
        n = {"i": 0}

        class _Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        n["i"] += 1
                        raise _restricted()

        with pytest.raises(BadRequestError):
            lc._panggil_llm_sekali(_Client(), [], {"type": "object"}, "t", 0.0, 100)
        assert n["i"] == 1


class TestGuardrailBerhentiSaatLLMMati:
    def test_percobaan_dihentikan_dan_sebab_dilabeli(self, monkeypatch):
        panggilan = {"n": 0}

        def _stub(*a, **k):
            panggilan["n"] += 1
            raise LayananTidakTersediaError("Semua 8 kunci Groq sedang dikarantina")

        monkeypatch.setattr(llm_client_module, "generate", _stub)
        poin = PoinKonteks(poin_id="itbx", kategori="ITBX", tipe_rekomendasi="kategorikal", status="B",
                           fakta={"lolos": True, "reason": "ok"})
        import json
        from pathlib import Path
        payload = json.loads((Path(__file__).parent / "fixtures" / "l2_sample_lolos.json").read_text(encoding="utf-8"))
        asesmen = L2Assessment.model_validate(payload["data"])
        hasil, diagnosa = generate_poin_terdiagnosis(poin, MockRetriever(), asesmen, max_retry=2)
        assert panggilan["n"] == 1  # tidak ada percobaan ulang yang sia-sia
        assert diagnosa.sebab() == "llm_tidak_tersedia"
        assert hasil.low_confidence is True

    def test_label_sebab_langsung(self):
        d = DiagnosaPoin(poin_id="itbx", berhasil=False, percobaan=1)
        d.exception_terakhir = "LayananTidakTersediaError: Semua kunci Groq sedang dikarantina"
        assert d.sebab() == "llm_tidak_tersedia"
