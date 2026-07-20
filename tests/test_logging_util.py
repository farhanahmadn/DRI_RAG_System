import json

from app.logging_util import log_precheck
from app.schemas import (
    IndikatorJejak,
    JejakAturanRequest,
    KesimpulanOutput,
    OutputPreCheck,
    PoinOutput,
    RekomendasiOutput,
    RingkasanOutput,
)


def _request() -> JejakAturanRequest:
    return JejakAturanRequest(
        skor_total=20.0,
        level="Tinggi",
        zona="LP2B",
        indikator=[
            IndikatorJejak(
                poin_id="LP2B-01",
                kategori="Lokasional LP2B",
                bobot=20.0,
                skor=100.0,
                kontribusi=20.0,
                nilai_input="dalam_lp2b",
                ambang="tidak_dalam_lp2b",
                operator="==",
                formula="in_lp2b == True",
            )
        ],
    )


def _response() -> OutputPreCheck:
    return OutputPreCheck(
        ringkasan=RingkasanOutput(skor_total=20.0, level="Tinggi", kalimat="kalimat"),
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


def test_log_precheck_menulis_satu_baris_jsonl_roundtrip(tmp_path):
    log_path = tmp_path / "precheck.jsonl"
    request = _request()
    response = _response()

    log_precheck(request, response, log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert "timestamp" in record
    assert JejakAturanRequest.model_validate(record["request"]) == request
    assert OutputPreCheck.model_validate(record["response"]) == response


def test_log_precheck_append_membuat_banyak_baris(tmp_path):
    log_path = tmp_path / "nested" / "precheck.jsonl"
    request = _request()
    response = _response()

    log_precheck(request, response, log_path=log_path)
    log_precheck(request, response, log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
