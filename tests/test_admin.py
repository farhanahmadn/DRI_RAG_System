"""tests/test_admin.py — POST /admin/cache/clear (app/api/admin.py).

Endpoint operasional (bukan reasoning) dilindungi ADMIN_TOKEN — dipanggil scripts/ingest.py /
scripts/reembed.py setelah korpus berubah, supaya cache retrieval proses API yang SEDANG HIDUP
ikut ter-invalidasi (lihat app/api/admin.py utk penjelasan lengkap kenapa perlu endpoint HTTP,
bukan cukup cache.clear() di dalam skrip CLI — beda proses OS, beda memori).
"""

from fastapi.testclient import TestClient

from app.api.dependencies import get_retriever
from app.api.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestVerifikasiAdminToken:
    def test_admin_token_tak_dikonfigurasi_di_server_selalu_403(self, monkeypatch):
        # Fail-closed: ADMIN_TOKEN kosong -> endpoint MENOLAK SEMUA permintaan, BUKAN default
        # terbuka tanpa proteksi (domain ini publik).
        monkeypatch.delenv("ADMIN_TOKEN", raising=False)
        response = client.post("/admin/cache/clear", headers={"X-Admin-Token": "apa-saja"})
        assert response.status_code == 403

    def test_tanpa_header_token_403(self, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "token-rahasia-benar")
        response = client.post("/admin/cache/clear")
        assert response.status_code == 403

    def test_token_salah_403(self, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "token-rahasia-benar")
        response = client.post("/admin/cache/clear", headers={"X-Admin-Token": "token-salah"})
        assert response.status_code == 403

    def test_token_benar_lolos_200(self, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "token-rahasia-benar")

        class _RetrieverStub:
            def __init__(self):
                self.dipanggil = False

            def clear_cache(self):
                self.dipanggil = True

        stub = _RetrieverStub()
        app.dependency_overrides[get_retriever] = lambda: stub
        try:
            response = client.post("/admin/cache/clear", headers={"X-Admin-Token": "token-rahasia-benar"})
        finally:
            app.dependency_overrides.pop(get_retriever, None)

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "cleared": True}
        assert stub.dipanggil is True


class TestClearRetrievalCache:
    def test_retriever_tanpa_clear_cache_tak_error(self, monkeypatch):
        # MockRetriever (mode dev/test tanpa DB) tak punya clear_cache() -> ditangani via getattr,
        # BUKAN AttributeError yang bikin endpoint 500.
        monkeypatch.setenv("ADMIN_TOKEN", "token-rahasia-benar")

        class _RetrieverTanpaCache:
            pass

        app.dependency_overrides[get_retriever] = lambda: _RetrieverTanpaCache()
        try:
            response = client.post("/admin/cache/clear", headers={"X-Admin-Token": "token-rahasia-benar"})
        finally:
            app.dependency_overrides.pop(get_retriever, None)

        assert response.status_code == 200
        assert response.json() == {
            "status": "ok", "cleared": False,
            "catatan": "Retriever aktif tak punya cache (mis. mode mock).",
        }
