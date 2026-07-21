"""Test observability.py — pastikan graceful no-op saat Langfuse tidak dikonfigurasi.

Tidak ada test terhadap Langfuse SUNGGUHAN (tidak ada server tersedia di sesi ini) — cuma
memverifikasi modul tidak pernah raise & tidak diam-diam butuh env var utk berfungsi normal.
"""

from app.reasoning import observability


def test_catat_generation_no_op_tanpa_env_langfuse(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(observability, "_client", None)
    monkeypatch.setattr(observability, "_sudah_dicoba_init", False)

    # Tidak boleh raise apa pun, walau argumen "aneh" (None, dict kosong, dsb).
    observability.catat_generation("test", "prompt", None, "model-x", 0.5, error="boom")
    observability.catat_generation("test", "prompt", {"ok": True}, "model-x", 0.1)


def test_catat_precheck_trace_no_op_tanpa_env_langfuse(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(observability, "_client", None)
    monkeypatch.setattr(observability, "_sudah_dicoba_init", False)

    observability.catat_precheck_trace("test-trace", {"a": 1}, {"b": 2})


def test_get_client_none_saat_key_tidak_diisi(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(observability, "_client", None)
    monkeypatch.setattr(observability, "_sudah_dicoba_init", False)

    assert observability._get_client() is None


def test_get_client_hasil_di_cache_setelah_percobaan_pertama(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setattr(observability, "_client", None)
    monkeypatch.setattr(observability, "_sudah_dicoba_init", False)

    assert observability._get_client() is None
    assert observability._sudah_dicoba_init is True
    # Panggilan kedua tidak mencoba inisialisasi ulang (state _sudah_dicoba_init sudah True)
    assert observability._get_client() is None
