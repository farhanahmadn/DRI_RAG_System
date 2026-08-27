"""tests/test_cache.py — app/retrieval/cache.py::notify_admin_cache_clear (dipakai
scripts/ingest.py & scripts/reembed.py). TIDAK memanggil jaringan sungguhan — urllib.request.urlopen
di-monkeypatch di semua test."""

import urllib.error
import urllib.request

from app.retrieval.cache import notify_admin_cache_clear


class _FakeResponse:
    def __init__(self, status: int):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestNotifyAdminCacheClear:
    def test_sukses_2xx_return_true(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(200))
        assert notify_admin_cache_clear("http://localhost:8000/admin/cache/clear", "token-benar") is True

    def test_status_bukan_2xx_return_false(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(403))
        assert notify_admin_cache_clear("http://localhost:8000/admin/cache/clear", "token-salah") is False

    def test_koneksi_gagal_return_false_tak_raise(self, monkeypatch):
        # Server API tak jalan/tak bisa dihubungi -> TIDAK BOLEH raise (ingest/reembed yg sudah
        # sukses tak boleh gagal gara-gara notifikasi cache ini, cuma peringatan non-fatal).
        def _raise(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        assert notify_admin_cache_clear("http://localhost:8000/admin/cache/clear", "token") is False

    def test_header_token_terpasang_di_request(self, monkeypatch):
        tertangkap = {}

        def _stub_urlopen(req, timeout=None):
            tertangkap["headers"] = dict(req.headers)
            tertangkap["method"] = req.get_method()
            return _FakeResponse(200)

        monkeypatch.setattr(urllib.request, "urlopen", _stub_urlopen)
        notify_admin_cache_clear("http://localhost:8000/admin/cache/clear", "token-rahasia-123")

        assert tertangkap["method"] == "POST"
        assert tertangkap["headers"].get("X-admin-token") == "token-rahasia-123"

    def test_tanpa_admin_token_tetap_kirim_request_tanpa_header(self, monkeypatch):
        # admin_token=None (mis. ADMIN_TOKEN tak diset lokal) -> tetap coba (server yg akan tolak
        # via 403 kalau memang butuh token), bukan skip diam-diam.
        tertangkap = {}

        def _stub_urlopen(req, timeout=None):
            tertangkap["headers"] = dict(req.headers)
            return _FakeResponse(403)

        monkeypatch.setattr(urllib.request, "urlopen", _stub_urlopen)
        hasil = notify_admin_cache_clear("http://localhost:8000/admin/cache/clear", None)

        assert "X-admin-token" not in tertangkap["headers"]
        assert hasil is False  # 403 -> bukan 2xx
