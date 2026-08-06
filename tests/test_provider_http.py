"""tests/test_provider_http.py — semaphore concurrency (`PROVIDER_MAX_CONCURRENT`) di
`app/retrieval/_provider_http.py`: panggilan ke-N+1 HARUS antre, bukan gagal, kalau sudah ada N
panggilan in-flight. Ditambahkan setelah observasi live (satu permohonan + stress-test 3x paralel
murni Jina tidak memicu 429, tapi REASONING_MAX_WORKERS adalah executor global lintas permohonan —
beban puncak sungguhan belum teruji) — semaphore ini jaring pengaman proaktif, bukan reaksi ke bug.
"""

from __future__ import annotations

import threading
import time

import app.retrieval._provider_http as http


def _reset_state(monkeypatch):
    monkeypatch.setattr(http, "_state", {})


class _FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"ok": True}


def test_semaphore_membatasi_jumlah_in_flight_bersamaan(monkeypatch):
    _reset_state(monkeypatch)
    monkeypatch.setattr(http, "_semaphore", threading.Semaphore(2))

    lock = threading.Lock()
    current = 0
    peak = 0

    class _FakeClient:
        def __init__(self, timeout: float) -> None:
            pass

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, *exc) -> None:
            pass

        def post(self, url: str, headers: dict, json: dict) -> _FakeResponse:
            nonlocal current, peak
            with lock:
                current += 1
                peak = max(peak, current)
            time.sleep(0.08)  # cukup lama supaya thread lain BENAR overlap kalau semaphore tak jalan
            with lock:
                current -= 1
            return _FakeResponse()

    import httpx
    monkeypatch.setattr(httpx, "Client", _FakeClient)

    def _panggil(i: int) -> None:
        http.post_json(f"test-semaphore-{i}", "http://fake", headers={}, json_body={},
                       timeout_s=1.0, max_retries=0)

    threads = [threading.Thread(target=_panggil, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert peak == 2, f"peak in-flight={peak}, seharusnya PERSIS 2 (semaphore dipaksa penuh dgn 6 panggilan serentak)"


def test_semaphore_tidak_menggagalkan_panggilan_hanya_antre(monkeypatch):
    """Semua panggilan HARUS tetap sukses (bukan raise/timeout) meski antre di semaphore."""
    _reset_state(monkeypatch)
    monkeypatch.setattr(http, "_semaphore", threading.Semaphore(1))

    class _FakeClient:
        def __init__(self, timeout: float) -> None:
            pass

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, *exc) -> None:
            pass

        def post(self, url: str, headers: dict, json: dict) -> _FakeResponse:
            time.sleep(0.02)
            return _FakeResponse()

    import httpx
    monkeypatch.setattr(httpx, "Client", _FakeClient)

    hasil: list[dict] = []
    errors: list[Exception] = []

    def _panggil(i: int) -> None:
        try:
            hasil.append(http.post_json(f"test-antre-{i}", "http://fake", headers={}, json_body={},
                                        timeout_s=1.0, max_retries=0))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_panggil, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(hasil) == 5
    assert all(h == {"ok": True} for h in hasil)


def test_default_max_concurrent_dari_env_masuk_akal():
    # Sanity: default kecil (bukan 0/tak terbatas) — nilai persis diatur PROVIDER_MAX_CONCURRENT.
    assert 1 <= http._MAX_CONCURRENT <= 20
