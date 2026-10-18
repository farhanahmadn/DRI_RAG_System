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


def post_json(
    name: str,
    url: str,
    *,
    headers: dict,
    json_body: dict,
    timeout_s: float,
    max_retries: int,
    kunci=None,
) -> dict:
    """POST JSON dgn retry (backoff eksponensial dibatasi 8s) + circuit breaker per `name` + semaphore
    concurrency global (`PROVIDER_MAX_CONCURRENT`) — panggilan ke-N+1 ANTRE (bukan gagal) kalau sudah
    ada N panggilan in-flight, tidak menambah beban ke provider yang sedang sibuk.

    `kunci` (opsional, `app.retrieval.jina_keys.KumpulanKunciJina`): header Authorization diisi dari key
    aktif. Bila balasan menandakan key HABIS/DITOLAK (402/401/403 atau 429 saldo), key ditandai habis dan
    permintaan yang SAMA langsung diulang dengan key berikutnya, tanpa menghabiskan jatah retry dan tanpa
    dihitung sebagai kegagalan circuit breaker."""
    import httpx

    from app import batas_waktu
    from app.retrieval import jina_keys

    _check_circuit(name)
    last_exc: Exception | None = None
    attempt = 0
    ganti_kunci = 0
    while True:
        # Anggaran waktu permintaan (app/batas_waktu.py): jangan mulai panggilan yang tak mungkin selesai,
        # dan jangan biarkan timeout per panggilan melampaui sisa anggaran.
        batas_waktu.periksa(2.0, f"panggilan provider {name}")
        sisa = batas_waktu.sisa()
        timeout_efektif = timeout_s if sisa is None else max(1.0, min(timeout_s, sisa - 1.0))
        idx = None
        hdr = headers
        if kunci is not None:
            idx, key = kunci.aktif()  # SemuaKunciJinaHabisError langsung naik ke pemanggil
            hdr = {**headers, "Authorization": f"Bearer {key}"}
        try:
            with _semaphore:
                with httpx.Client(timeout=timeout_efektif) as client:
                    resp = client.post(url, headers=hdr, json=json_body)
            if kunci is not None and jina_keys.kunci_habis(resp.status_code, resp.text):
                kunci.tandai_habis(idx, f"HTTP {resp.status_code}")
                ganti_kunci += 1
                if ganti_kunci <= kunci.jumlah():
                    continue  # ulang permintaan yang sama dgn key berikutnya (aktif() akan raise bila habis)
            resp.raise_for_status()
            _record_success(name)
            return resp.json()
        except jina_keys.SemuaKunciJinaHabisError:
            raise
        except Exception as exc:  # noqa: BLE001 — transport apa pun dicoba ulang (timeout/5xx/DNS/dll)
            last_exc = exc
            _record_failure(name)
            if attempt >= max_retries:
                break
            jeda = min(2 ** attempt, 8)
            batas_waktu.periksa(jeda + 3.0, f"ulang panggilan provider {name}")
            time.sleep(jeda)
            attempt += 1
    raise RuntimeError(
        f"Panggilan provider '{name}' gagal setelah {max_retries + 1} percobaan: {last_exc}"
    ) from last_exc
