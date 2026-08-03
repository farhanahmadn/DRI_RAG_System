"""Observability opsional via Langfuse (self-host) — CLAUDE.md § Eval/observability.

DEGRADE GRACEFULLY: kalau `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` tidak diisi di env, atau
library `langfuse` tidak terpasang (opsional — lihat `pyproject.toml` extras `observability`),
semua fungsi di sini jadi no-op. Tracing TIDAK BOLEH PERNAH menggagalkan pipeline reasoning utama
— setiap pemanggilan Langfuse dibungkus try/except terpisah, termasuk saat server tak terjangkau
(SDK v4 sendiri sudah membungkus kegagalan ekspor span secara async, tapi kita tetap defensif).

CATATAN DESAIN: tiap generation/trace di sini berdiri sendiri (TIDAK di-nest lewat trace_context)
supaya sederhana & aman lintas-thread — `assemble.py` memproses indikator paralel via
ThreadPoolExecutor, dan propagasi konteks OTel antar-thread rapuh/di luar scope fase ini.
"""

import logging
import os

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse

    _TERSEDIA = True
except ImportError:
    _TERSEDIA = False

_client = None
_sudah_dicoba_init = False


def _get_client():
    global _client, _sudah_dicoba_init
    if _sudah_dicoba_init:
        return _client
    _sudah_dicoba_init = True

    if not _TERSEDIA:
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None

    try:
        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST") or None,
        )
    except Exception:
        logger.warning("Gagal inisialisasi Langfuse — observability dinonaktifkan.", exc_info=True)
        _client = None
    return _client


def catat_generation(
    nama: str,
    prompt: str,
    hasil: dict | None,
    model: str,
    latensi_detik: float,
    error: str | None = None,
) -> None:
    """Catat satu panggilan LLM mentah ke Langfuse. No-op kalau Langfuse tidak dikonfigurasi."""
    client = _get_client()
    if client is None:
        return
    try:
        with client.start_as_current_observation(
            name=nama,
            as_type="generation",
            input=prompt,
            output=hasil,
            model=model,
            metadata={"latensi_detik": latensi_detik, "error": error},
            level="ERROR" if error else "DEFAULT",
        ):
            pass
    except Exception:
        logger.warning("Gagal mengirim generation ke Langfuse (diabaikan).", exc_info=True)


def catat_precheck_trace(nama: str, request_json: dict, output_json: dict) -> None:
    """Catat satu ringkasan precheck (1 request). No-op kalau Langfuse tidak dikonfigurasi."""
    client = _get_client()
    if client is None:
        return
    try:
        with client.start_as_current_observation(
            name=nama,
            as_type="span",
            input=request_json,
            output=output_json,
        ):
            pass
    except Exception:
        logger.warning("Gagal mengirim trace ke Langfuse (diabaikan).", exc_info=True)
