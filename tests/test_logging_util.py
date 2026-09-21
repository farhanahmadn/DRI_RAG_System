import json
from pathlib import Path

from app.logging_util import log_precheck
from app.schemas import (
    KesimpulanOutput,
    L2Assessment,
    NarasiRekomendasiOutput,
    OutputL3,
    PoinOutput,
    RekomendasiOutput,
    RingkasanDampakOutput,
    RingkasanGateOutput,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _assessment() -> L2Assessment:
    payload = json.loads((FIXTURES_DIR / "l2_sample_lolos.json").read_text(encoding="utf-8"))
    return L2Assessment.model_validate(payload["data"])


def _response() -> OutputL3:
    return OutputL3(
        ringkasan_gate=RingkasanGateOutput(final_gate_status="Lolos", decisive_stage=None, kalimat="kalimat gate"),
        ringkasan_dampak=RingkasanDampakOutput(impact_category="Sedang", impact_score=65, kalimat="kalimat dampak"),
        poin=[
            PoinOutput(
                poin_id="itbx",
                kategori="Klasifikasi Kegiatan (ITBX)",
                status="I",
                reasoning_pendek="pendek",
                reasoning_panjang="panjang",
                sitasi=[],
                rekomendasi=RekomendasiOutput(tipe="kategorikal", target=None, saran="saran", disclaimer=None),
            )
        ],
        rekomendasi_sistem="Setuju",
        narasi_rekomendasi=NarasiRekomendasiOutput(paragraf_gate_intensitas="x", paragraf_dampak="x"),
        kesimpulan=KesimpulanOutput(langkah_berdampak=["a"], catatan_lokasi="catatan"),
        catatan_global=[],
        low_confidence_keseluruhan=False,
    )


def test_log_precheck_menulis_satu_baris_jsonl_roundtrip(tmp_path):
    log_path = tmp_path / "precheck.jsonl"
    assessment = _assessment()
    response = _response()

    log_precheck(assessment, response, log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert "timestamp" in record
    assert L2Assessment.model_validate(record["request"]) == assessment
    assert OutputL3.model_validate(record["response"]) == response


def test_log_precheck_append_membuat_banyak_baris(tmp_path):
    log_path = tmp_path / "nested" / "precheck.jsonl"
    assessment = _assessment()
    response = _response()

    log_precheck(assessment, response, log_path=log_path)
    log_precheck(assessment, response, log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
