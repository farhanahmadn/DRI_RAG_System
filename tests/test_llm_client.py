import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def test_get_client_pakai_timeout_dan_max_retries_dari_konfigurasi(monkeypatch):
    # Config timeout/max_retries dibaca sekali saat modul di-load (mirip pola LLM_BASE_URL, beda
    # dari LLM_MODEL yang sengaja dibaca ulang tiap panggilan utk bakeoff). Test ini verifikasi
    # WIRING-nya benar (client dibangun pakai konstanta modul), bukan nilai numerik env spesifik.
    import app.reasoning.llm_client as llm_client_module

    monkeypatch.setenv("GROQ_API_KEY", "fake-key-utk-test")
    monkeypatch.setattr(llm_client_module, "_client", None)

    client = llm_client_module._get_client()

    assert client.timeout == llm_client_module._TIMEOUT_S
    assert client.max_retries == llm_client_module._MAX_RETRIES
    # monkeypatch otomatis kembalikan _client ke nilai semula saat test ini selesai


def test_get_client_default_timeout_dan_retries_masuk_akal():
    import app.reasoning.llm_client as llm_client_module

    # default 30s / 2 retries kalau env tidak diisi — cukup responsif utk precheck sinkron,
    # tidak selama default SDK (~10 menit) yang terlalu lama utk web request.
    assert llm_client_module._TIMEOUT_S <= 60
    assert 0 <= llm_client_module._MAX_RETRIES <= 5


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY tidak diset — skip smoke test panggilan LLM nyata.",
)
def test_generate_returns_valid_json():
    from app.reasoning.llm_client import generate

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    result = generate(
        prompt="Balas dengan JSON {\"ok\": true} saja, tanpa penjelasan apapun.",
        json_schema=schema,
        schema_name="ok_check",
        max_tokens=50,
    )

    assert isinstance(result, dict)
    assert "ok" in result
    assert isinstance(result["ok"], bool)
