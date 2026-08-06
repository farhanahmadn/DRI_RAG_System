"""app/retrieval/_provider_http.py — POST JSON tipis dgn retry+timeout+circuit-breaker+semaphore.

Dipakai `embeddings.py` & `rerank.py` untuk provider serverless (Jina) supaya retry/timeout/
circuit-breaker/pembatas concurrency tidak diduplikasi di kedua modul. Pola sama seperti
`app/reasoning/llm_client.py`: timeout/retry di sini adalah hardening TRANSPORT (koneksi macet/5xx
sesaat), bukan retry semantik.

Circuit breaker: per `name` (mis. "jina-embed"), buka setelah N kegagalan beruntun supaya
panggilan berikutnya gagal cepat (tanpa menunggu timeout) selama masa cooldown alih-alih membombardir
provider yang sedang down — lalu coba lagi satu kali (half-open) setelah cooldown lewat.

Semaphore concurrency (`PROVIDER_MAX_CONCURRENT`, default 2): dibuat SETELAH observasi live —
1 permohonan (3 poin, `REASONING_MAX_WORKERS` ThreadPoolExecutor) & stress-test 3x paralel murni
Jina TIDAK memicu 429, TAPI `REASONING_MAX_WORKERS` (default 8) adalah satu executor GLOBAL yang
dipakai bersama SEMUA permohonan bersamaan — beban puncak sungguhan (beberapa permohonan sekaligus)
belum teruji. Semaphore ini SENGAJA terpisah total dari `REASONING_MAX_WORKERS` (yang mengatur
paralelisme thread CPU-bound reasoning secara umum, bukan cuma panggilan Jina) — membatasi berapa
banyak panggilan HTTP ke Jina yang boleh IN-FLIGHT bersamaan, apa pun jumlah thread reasoning yang
aktif. Berbagi SATU semaphore lintas embed+rerank (bukan dipisah per operasi) — konservatif: rate
limit Jina kemungkinan dihitung per-akun/per-API-key, bukan per-endpoint.
"""

from __future__ import annotations

import os
import threading
import time
from threading import Lock

_FAILURE_THRESHOLD = int(os.getenv("PROVIDER_CIRCUIT_FAILURE_THRESHOLD", "5"))
_COOLDOWN_S = float(os.getenv("PROVIDER_CIRCUIT_COOLDOWN_S", "30"))
_MAX_CONCURRENT = int(os.getenv("PROVIDER_MAX_CONCURRENT", "2"))

_state: dict[str, dict] = {}
_lock = Lock()
_semaphore = threading.Semaphore(_MAX_CONCURRENT)


class CircuitOpenError(RuntimeError):
    """Circuit breaker sedang terbuka: provider gagal berulang kali, panggilan di-skip sementara."""


def _st(name: str) -> dict:
    return _state.setdefault(name, {"failures": 0, "opened_until": 0.0})


def _check_circuit(name: str) -> None:
    with _lock:
        st = _st(name)
        if st["failures"] >= _FAILURE_THRESHOLD and time.monotonic() < st["opened_until"]:
            sisa = st["opened_until"] - time.monotonic()
            raise CircuitOpenError(
                f"Circuit breaker '{name}' terbuka ({st['failures']} gagal beruntun); "
                f"coba lagi setelah {sisa:.0f}s."
            )


def _record_success(name: str) -> None:
    with _lock:
        _st(name)["failures"] = 0


def _record_failure(name: str) -> None:
    with _lock:
        st = _st(name)
        st["failures"] += 1
        if st["failures"] >= _FAILURE_THRESHOLD:
            st["opened_until"] = time.monotonic() + _COOLDOWN_S


def reset_circuit(name: str | None = None) -> None:
    """Reset state breaker (test helper). `None` = reset semua."""
    with _lock:
        if name is None:
            _state.clear()
        else:
            _state.pop(name, None)


def post_json(
    name: str,
    url: str,
    *,
    headers: dict,
    json_body: dict,
    timeout_s: float,
    max_retries: int,
) -> dict:
    """POST JSON dgn retry (backoff eksponensial dibatasi 8s) + circuit breaker per `name` + semaphore
    concurrency global (`PROVIDER_MAX_CONCURRENT`) — panggilan ke-N+1 ANTRE (bukan gagal) kalau sudah
    ada N panggilan in-flight, tidak menambah beban ke provider yang sedang sibuk."""
    import httpx

    _check_circuit(name)
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with _semaphore:
                with httpx.Client(timeout=timeout_s) as client:
                    resp = client.post(url, headers=headers, json=json_body)
            resp.raise_for_status()
            _record_success(name)
            return resp.json()
        except Exception as exc:  # noqa: BLE001 — transport apa pun dicoba ulang (timeout/5xx/DNS/dll)
            last_exc = exc
            _record_failure(name)
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(
        f"Panggilan provider '{name}' gagal setelah {max_retries + 1} percobaan: {last_exc}"
    ) from last_exc
