from fastapi.testclient import TestClient

from app.api import main as main_module
from app.api.dependencies import get_retriever
from app.api.main import app
from app.schemas import (
    KesimpulanOutput,
    OutputPreCheck,
    PoinOutput,
    RekomendasiOutput,
    RingkasanOutput,
)

client = TestClient(app, raise_server_exceptions=False)


def _output_kanonik() -> OutputPreCheck:
    return OutputPreCheck(
        ringkasan=RingkasanOutput(skor_total=20.0, level="Tinggi", kalimat="kalimat ringkasan"),
        poin=[
            PoinOutput(
                poin_id="LP2B-01",
                kategori="Lokasional LP2B",
                status="Tidak Aman",
                kontribusi=20.0,
                reasoning_pendek="pendek",
                reasoning_panjang="panjang",
                sitasi=[],
                rekomendasi=RekomendasiOutput(tipe="lokasional", target=None, saran="saran", disclaimer=None),
            )
        ],
        kesimpulan=KesimpulanOutput(langkah_berdampak=["a"], catatan_lokasi="catatan"),
    )


def _request_body() -> dict:
    return {
        "skor_total": 20.0,
        "level": "Tinggi",
        "zona": "LP2B",
        "indikator": [
            {
                "poin_id": "LP2B-01",
                "kategori": "Lokasional LP2B",
                "bobot": 20.0,
                "skor": 100.0,
                "kontribusi": 20.0,
                "nilai_input": "dalam_lp2b",
                "ambang": "tidak_dalam_lp2b",
                "operator": "==",
                "formula": "in_lp2b == True",
                "referensi_hukum": ["UU No. 41 Tahun 2009 Pasal 44"],
            }
        ],
    }


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


def test_reasoning_input_buruk_422():
    body = _request_body()
    del body["indikator"]

    response = client.post("/reasoning", json=body)

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
