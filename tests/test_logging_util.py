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


def test_kunci_diagnostik_selalu_ada_walau_pemanggil_tak_memasoknya(tmp_path):
    """Ketiadaan kunci `diagnostik` dulu punya DUA arti yang tak bisa dipisahkan pembaca mana pun:
    "baris lebih tua dari instrumentasinya" (632 dari 798 baris di logs/precheck.jsonl, semuanya
    sebelum 2026-09-07) versus "pemanggil tak memasoknya". Dengan kunci yang selalu ada, hanya
    arti pertama yang tersisa — dan `eval/metrik_generasi.py` bisa membuang baris era lama dari
    penyebut dgn alasan yang jelas, bukan menebak."""
    log_path = tmp_path / "precheck.jsonl"

    log_precheck(_assessment(), _response(), log_path=log_path)

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert "diagnostik" in record
    assert record["diagnostik"] == []


def test_diagnostik_ditulis_apa_adanya_termasuk_riwayat_percobaan(tmp_path):
    """Riwayat per-percobaan harus sampai ke berkas utuh — ini satu-satunya rekaman tentang apa
    yang guardrail tolak dari keluaran LLM mentah (lihat guardrail.DiagnosaPoin)."""
    log_path = tmp_path / "precheck.jsonl"
    diagnostik = [{
        "poin_id": "itbx", "berhasil": True, "percobaan": 2, "sebab": "berhasil",
        "masalah_terakhir": [],
        "riwayat_percobaan": [
            {"percobaan": 1, "sebab": "guardrail_menolak",
             "masalah": ["reasoning/saran memakai tanda titik koma (;)"], "exception": None},
            {"percobaan": 2, "sebab": "berhasil", "masalah": [], "exception": None},
        ],
    }]

    log_precheck(_assessment(), _response(), diagnostik=diagnostik, log_path=log_path)

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["diagnostik"] == diagnostik
