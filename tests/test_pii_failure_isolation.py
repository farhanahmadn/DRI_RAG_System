"""tests/test_pii_failure_isolation.py — Tahap 6 (lanjutan): `PIIDetectedError` di SATU poin TIDAK
menjatuhkan seluruh permohonan. Dua jalur kemunculan diuji terpisah (beda titik tangkap):

1. **`ambil_chunks_pendukung`** — dipanggil `guardrail.py::generate_poin_dengan_guardrail` di baris
   PERTAMA, DI LUAR loop retry-nya sendiri. Kalau raise di sini, exception keluar MENTAH dari
   `generate_poin_dengan_guardrail` — HANYA tertangkap oleh `assemble.py::_generate_poin_defensif`
   (`except Exception` generik, safety-net terluar). Ini merepresentasikan `RetrieverAsli.search()`/
   `get_by_reference()` sungguhan yang melempar `PIIDetectedError` dari
   `app/sanitize.py::sanitize_query_text`.
2. **`sanitize.sanitize_fakta`** — dipanggil `app/reasoning/prompts.py::build_user_prompt`, DI DALAM
   loop retry `generate_poin_dengan_guardrail` (lewat `generate_poin()`). Raise di sini dihitung
   guardrail sbg "percobaan gagal", di-retry sampai `max_retry`, lalu guardrail SENDIRI fallback ke
   `template_low_confidence` (tak perlu ditangkap assemble.py — jalur retry-exhaustion guardrail).

Kedua jalur berujung SAMA dari sudut pandang permohonan: satu poin turun ke low_confidence, dua
poin lain tetap normal, response tetap 200 — dibuktikan lewat `jalankan_precheck()` langsung DAN
lewat endpoint HTTP `/reasoning` sungguhan (TestClient).

Konfirmasi: `PIIDetectedError(ValueError)` — subclass `ValueError`, yang subclass `Exception` biasa
(BUKAN `BaseException` langsung/`SystemExit`/`KeyboardInterrupt` yang bisa lolos dari
`except Exception`) — jadi PASTI tertangkap oleh `except Exception` di `assemble.py` maupun
`guardrail.py`. Dites eksplisit di `TestPIIDetectedErrorAdalahExceptionBiasa` di bawah.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.reasoning import guardrail as guardrail_module
from app.reasoning import llm_client
from app.reasoning.assemble import jalankan_precheck
from app.retrieval.mock import MockRetriever
from app.sanitize import PIIDetectedError
from app.schemas import L2Assessment

FIXTURES_DIR = Path(__file__).parent / "fixtures"
_RE_CITATION_ID = re.compile(r"citation_id=(\S+)")


def _muat_assessment(nama_file: str) -> L2Assessment:
    payload = json.loads((FIXTURES_DIR / nama_file).read_text(encoding="utf-8"))
    return L2Assessment.model_validate(payload["data"])


def _stub_llm_generate(prompt, json_schema, *, schema_name="response", system=None,
                       temperature=0.0, max_tokens=1024):
    """Stub deterministik (TIDAK memanggil Groq sungguhan) — echo citation_id pertama yang muncul
    di prompt supaya guardrail.perbaiki_poin (cek sitasi valid) lolos utk poin yang MEMANG harus
    sukses normal (bukan bagian yang diuji gagal di sini)."""
    if schema_name == "kesimpulan":
        return {"langkah_berdampak": ["Langkah stub A", "Langkah stub B"], "catatan_lokasi": None}
    m = _RE_CITATION_ID.search(prompt)
    sitasi = [{"citation_id": m.group(1), "kutipan": "kutipan stub"}] if m else []
    return {
        "reasoning_pendek": "Reasoning pendek stub yang jelas.",
        "reasoning_panjang": (
            "Reasoning panjang stub yang cukup lengkap untuk lolos validasi guardrail tanpa "
            "memicu regenerasi tambahan yang tidak perlu."
        ),
        "sitasi": sitasi,
        "saran": "Saran stub yang jelas dan dapat ditindaklanjuti.",
        "disclaimer": None,
    }


def _assert_isolasi_per_poin(output, poin_bermasalah: str = "intensitas") -> None:
    """Assersi bersama (1-4 sesuai permintaan) — dipakai kedua jalur."""
    by_id = {p.poin_id: p for p in output.poin}
    assert set(by_id) == {"itbx", "intensitas", "dampak"}

    # (1) Poin bermasalah -> low_confidence=True, isi persis dari template_low_confidence(), disclaimer ada.
    bermasalah = by_id[poin_bermasalah]
    assert bermasalah.low_confidence is True
    assert bermasalah.reasoning_pendek == "Penjelasan otomatis tidak tersedia untuk poin ini."
    assert bermasalah.sitasi == []
    assert "peninjauan manual" in bermasalah.rekomendasi.saran.lower()
    assert bermasalah.rekomendasi.disclaimer  # ada isinya (bukan None/string kosong)
    assert "verifikasi manual" in bermasalah.rekomendasi.disclaimer.lower()

    # (2) Poin lain -> tetap normal, TIDAK terdampak.
    for poin_id, poin in by_id.items():
        if poin_id == poin_bermasalah:
            continue
        assert poin.low_confidence is False, f"{poin_id} seharusnya tidak ikut terdampak"
        assert poin.reasoning_pendek == "Reasoning pendek stub yang jelas."

    # (3) Level permohonan: low_confidence_keseluruhan + catatan_global menyebut poin bermasalah.
    assert output.low_confidence_keseluruhan is True
    assert any(poin_bermasalah in c for c in output.catatan_global), (
        f"catatan_global harus menyebut poin {poin_bermasalah!r} perlu ditinjau manual: "
        f"{output.catatan_global}"
    )


class TestPIIDetectedErrorAdalahExceptionBiasa:
    def test_subclass_exception_bukan_baseexception_lain(self):
        assert issubclass(PIIDetectedError, ValueError)
        assert issubclass(PIIDetectedError, Exception)
        # Pastikan bukan turunan BaseException "eksotis" (SystemExit/KeyboardInterrupt/GeneratorExit)
        # yang TIDAK tertangkap `except Exception` — sudah tercakup oleh assert subclass Exception di
        # atas (ketiganya BUKAN subclass Exception), tapi ditegaskan eksplisit di sini.
        assert not issubclass(PIIDetectedError, (SystemExit, KeyboardInterrupt, GeneratorExit))


class TestPIIDetectedErrorTakMenjatuhkanPermohonan:
    """Dua jalur kemunculan PIIDetectedError, keduanya HARUS terisolasi ke satu poin."""

    def test_via_ambil_chunks_pendukung_tak_terguard_retry(self, monkeypatch):
        """Jalur 1: raise di panggilan PERTAMA (guardrail.py, di luar try/except-nya sendiri) —
        hanya tertangkap assemble.py::_generate_poin_defensif (safety net terluar)."""
        monkeypatch.setattr(llm_client, "generate", _stub_llm_generate)
        asli = guardrail_module.ambil_chunks_pendukung

        def _stub(poin, retriever, top_k_dukungan=3):
            if poin.poin_id == "intensitas":
                raise PIIDetectedError("simulasi: pola PII terdeteksi di query retrieval intensitas")
            return asli(poin, retriever, top_k_dukungan)

        monkeypatch.setattr(guardrail_module, "ambil_chunks_pendukung", _stub)

        assessment = _muat_assessment("l2_sample_amplop_6191.json")
        output = jalankan_precheck(assessment, MockRetriever())

        _assert_isolasi_per_poin(output, "intensitas")

    def test_via_sanitize_fakta_guardrail_retry_exhaustion(self, monkeypatch):
        """Jalur 2: raise di DALAM loop retry guardrail (lewat prompts.py::build_user_prompt) —
        guardrail menghitungnya sbg percobaan gagal, exhaust retry, fallback SENDIRI (assemble.py
        tak sampai turun tangan — tapi hasil akhir dari sudut pandang endpoint tetap sama)."""
        monkeypatch.setattr(llm_client, "generate", _stub_llm_generate)
        import app.sanitize as sanitize_module
        asli = sanitize_module.sanitize_fakta

        def _stub(poin_id, fakta):
            if poin_id == "intensitas":
                raise PIIDetectedError("simulasi: pola PII terdeteksi di fakta poin intensitas")
            return asli(poin_id, fakta)

        monkeypatch.setattr(sanitize_module, "sanitize_fakta", _stub)

        assessment = _muat_assessment("l2_sample_amplop_6191.json")
        output = jalankan_precheck(assessment, MockRetriever())

        _assert_isolasi_per_poin(output, "intensitas")

    def test_endpoint_reasoning_tetap_200_saat_pii_di_satu_poin(self, monkeypatch):
        """(4) Bukti end-to-end lewat HTTP sungguhan: endpoint /reasoning TIDAK 500 walau satu poin
        kena PIIDetectedError — hanya poin itu low_confidence, response tetap 200."""
        monkeypatch.setattr(llm_client, "generate", _stub_llm_generate)
        import app.sanitize as sanitize_module
        asli = sanitize_module.sanitize_fakta

        def _stub(poin_id, fakta):
            if poin_id == "intensitas":
                raise PIIDetectedError("simulasi: pola PII terdeteksi di fakta poin intensitas")
            return asli(poin_id, fakta)

        monkeypatch.setattr(sanitize_module, "sanitize_fakta", _stub)

        from app.api.main import app as fastapi_app
        from app.api.rate_limit import reset_rate_limiter

        reset_rate_limiter()
        client = TestClient(fastapi_app, raise_server_exceptions=False)
        payload = json.loads((FIXTURES_DIR / "l2_sample_amplop_6191.json").read_text(encoding="utf-8"))
        try:
            response = client.post("/reasoning", json=payload)
        finally:
            reset_rate_limiter()

        assert response.status_code == 200
        body = response.json()
        by_id = {p["poin_id"]: p for p in body["poin"]}
        assert by_id["intensitas"]["low_confidence"] is True
        assert by_id["itbx"]["low_confidence"] is False
        assert by_id["dampak"]["low_confidence"] is False
        assert body["low_confidence_keseluruhan"] is True
        assert any("intensitas" in c for c in body["catatan_global"])
