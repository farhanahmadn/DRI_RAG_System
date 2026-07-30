import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import main as main_module
from app.api.dependencies import get_retriever
from app.api.main import app
from app.api.rate_limit import _LIMIT, reset_rate_limiter
from app.schemas import (
    KesimpulanOutput,
    OutputL3,
    PoinOutput,
    RekomendasiOutput,
    RingkasanDampakOutput,
    RingkasanGateOutput,
)

client = TestClient(app, raise_server_exceptions=False)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _bersihkan_rate_limiter():
    # State limiter global (in-memory, per-IP) — reset supaya test lain di file ini tidak ikut
    # menghabiskan/terpengaruh kuota dari TestClient yang berbagi IP palsu yang sama.
    reset_rate_limiter()
    yield
    reset_rate_limiter()


def _request_body() -> dict:
    payload = json.loads((FIXTURES_DIR / "l2_sample_lolos.json").read_text(encoding="utf-8"))
    return copy.deepcopy(payload["data"])


def _request_body_amplop() -> dict:
    # Bentuk asli respons back-end L2 Spatial Risk Assessment — {statusCode, message, data}
    # (APP-2026-6191: ITBX 'T', intensitas MELAMPAUI_BATAS, dampak Tinggi — real payload berbeda
    # dari _request_body(), supaya test amplop vs polos benar-benar independen).
    payload = json.loads((FIXTURES_DIR / "l2_sample_amplop_6191.json").read_text(encoding="utf-8"))
    return copy.deepcopy(payload)


def _output_kanonik() -> OutputL3:
    return OutputL3(
        ringkasan_gate=RingkasanGateOutput(final_gate_status="Lolos", decisive_stage=None, kalimat="kalimat gate"),
        ringkasan_dampak=RingkasanDampakOutput(impact_category="Sedang", impact_score=65, kalimat="kalimat dampak"),
        poin=[
            PoinOutput(
                poin_id="itbx",
                kategori="Klasifikasi Kegiatan (ITBX)",
                status="I",
                reasoning_pendek="pendek",
                reasoning_panjang="panjang",
                sitasi=[],
                rekomendasi=RekomendasiOutput(tipe="kategorikal", target=None, saran="saran", disclaimer=None),
            )
        ],
        rekomendasi_sistem="Setuju",
        kesimpulan=KesimpulanOutput(langkah_berdampak=["a"], catatan_lokasi="catatan"),
        catatan_global=[],
        low_confidence_keseluruhan=False,
    )


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_reasoning_happy_path_mock(monkeypatch):
    output = _output_kanonik()
    monkeypatch.setattr(main_module, "jalankan_precheck", lambda request, retriever: output)

    response = client.post("/reasoning", json=_request_body())

    assert response.status_code == 200
    assert response.json() == output.model_dump(mode="json")


def test_reasoning_terima_payload_amplop_statuscode_message_data(monkeypatch):
    # L2Envelope: payload sudah ber-amplop (fixture asli l2_sample_lolos.json, {statusCode,
    # message, data}) -> dipakai apa adanya, .data diteruskan ke jalankan_precheck.
    output = _output_kanonik()
    monkeypatch.setattr(main_module, "jalankan_precheck", lambda request, retriever: output)

    response = client.post("/reasoning", json=_request_body_amplop())

    assert response.status_code == 200
    assert response.json() == output.model_dump(mode="json")


def test_reasoning_terima_payload_polos_kompatibel_mundur(monkeypatch):
    # L2Envelope: payload polos (langsung lokasi/gate_hukum/...) -> dibungkus otomatis jadi
    # {"data": v} oleh _bungkus_kalau_polos. Sama assersi dgn test_reasoning_happy_path_mock,
    # ditulis eksplisit di sini supaya niat "kompatibel mundur" untuk L2Envelope jelas terlihat.
    output = _output_kanonik()
    monkeypatch.setattr(main_module, "jalankan_precheck", lambda request, retriever: output)

    response = client.post("/reasoning", json=_request_body())

    assert response.status_code == 200
    assert response.json() == output.model_dump(mode="json")


def test_reasoning_input_buruk_422():
    body = _request_body()
    del body["gate_hukum"]

    response = client.post("/reasoning", json=body)

    assert response.status_code == 422


def test_reasoning_amplop_tanpa_data_422():
    # {statusCode, message} tanpa 'data' -> bukan L2Assessment valid juga (dibungkus ulang jadi
    # {"data": {...tanpa lokasi/gate_hukum/...}} oleh _bungkus_kalau_polos) -> tetap 422.
    response = client.post("/reasoning", json={"statusCode": 200, "message": "tanpa data"})

    assert response.status_code == 422


def test_reasoning_amplop_data_bukan_l2assessment_valid_422():
    response = client.post(
        "/reasoning",
        json={"statusCode": 200, "message": "x", "data": {"foo": "bar"}},
    )

    assert response.status_code == 422


def test_reasoning_retriever_via_dependency_injection(monkeypatch):
    class _RetrieverStub:
        pass

    stub = _RetrieverStub()
    diterima = {}

    def _stub_jalankan_precheck(request, retriever):
        diterima["retriever"] = retriever
        return _output_kanonik()

    monkeypatch.setattr(main_module, "jalankan_precheck", _stub_jalankan_precheck)
    app.dependency_overrides[get_retriever] = lambda: stub
    try:
        response = client.post("/reasoning", json=_request_body())
    finally:
        app.dependency_overrides.pop(get_retriever, None)

    assert response.status_code == 200
    assert diterima["retriever"] is stub


def test_reasoning_exception_tak_terduga_500_terstruktur(monkeypatch):
    def _stub_raise(request, retriever):
        raise RuntimeError("boom - detail internal rahasia")

    monkeypatch.setattr(main_module, "jalankan_precheck", _stub_raise)

    response = client.post("/reasoning", json=_request_body())

    assert response.status_code == 500
    body = response.json()
    assert set(body.keys()) == {"error", "message"}
    assert "boom" not in response.text


def test_reasoning_rate_limit_429_setelah_melebihi_batas(monkeypatch):
    output = _output_kanonik()
    monkeypatch.setattr(main_module, "jalankan_precheck", lambda request, retriever: output)

    for _ in range(_LIMIT):
        response = client.post("/reasoning", json=_request_body())
        assert response.status_code == 200

    response = client.post("/reasoning", json=_request_body())
    assert response.status_code == 429


def test_health_tidak_kena_rate_limit():
    for _ in range(_LIMIT + 5):
        response = client.get("/health")
        assert response.status_code == 200
