import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import (
    KesimpulanOutput,
    L2Assessment,
    OutputL3,
    ParameterIntensitas,
    PoinOutput,
    RekomendasiOutput,
    RingkasanDampakOutput,
    RingkasanGateOutput,
    SitasiOutput,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("nama_file", ["l2_sample_lolos.json", "l2_sample_lolos_bersyarat.json"])
def test_l2_assessment_valid_dari_fixture_nyata(nama_file):
    payload = json.loads((FIXTURES_DIR / nama_file).read_text(encoding="utf-8"))
    assessment = L2Assessment.model_validate(payload["data"])

    dumped = assessment.model_dump()
    restored = L2Assessment.model_validate(dumped)
    assert restored == assessment


def test_l2_assessment_toleran_extra_fields_di_payload_asli():
    # payload["data"] punya application_id/application_number/timestamp yang TIDAK ada di daftar
    # field eksplisit kontrak awal, tapi kita sertakan di skema — pastikan tetap valid, bukan diabaikan.
    payload = json.loads((FIXTURES_DIR / "l2_sample_lolos.json").read_text(encoding="utf-8"))
    assessment = L2Assessment.model_validate(payload["data"])
    assert assessment.application_id == 11
    assert assessment.application_number == "APP-2026-3468"


def test_parameter_intensitas_ambang_null_toleran():
    # KDH cuma punya ambang_min, KDB/KLB cuma ambang_maks — keduanya optional generik.
    param = ParameterIntensitas(usulan=29.4, ambang_min=0, memenuhi=True, satuan="persen")
    assert param.ambang_maks is None


def test_meta_boleh_absen():
    payload = json.loads((FIXTURES_DIR / "l2_sample_lolos.json").read_text(encoding="utf-8"))
    data = dict(payload["data"])
    data.pop("meta", None)
    assessment = L2Assessment.model_validate(data)
    assert assessment.meta is None


def test_itbx_status_invalid_raises():
    payload = json.loads((FIXTURES_DIR / "l2_sample_lolos.json").read_text(encoding="utf-8"))
    data = json.loads(json.dumps(payload["data"]))
    data["gate_hukum"]["tahapan"]["itbx"]["status"] = "STATUS_TAK_DIKENAL"
    with pytest.raises(ValidationError):
        L2Assessment.model_validate(data)


def test_output_l3_roundtrip():
    output = OutputL3(
        ringkasan_gate=RingkasanGateOutput(final_gate_status="Lolos Bersyarat", kalimat="Lolos bersyarat karena KDB melampaui ambang."),
        ringkasan_dampak=RingkasanDampakOutput(impact_category="Sedang", impact_score=65, kalimat="Dampak tata guna lahan tergolong sedang."),
        poin=[
            PoinOutput(
                poin_id="intensitas",
                kategori="Intensitas Bangunan (KDB/KLB/KDH)",
                status="MELAMPAUI_BATAS",
                reasoning_pendek="KDB usulan melampaui ambang maksimum.",
                reasoning_panjang="KDB yang diusulkan 70% melampaui ambang maksimum 60% yang ditetapkan.",
                sitasi=[
                    SitasiOutput(
                        citation_id="c1",
                        dokumen="RDTR Sleman",
                        pasal="Lampiran VI",
                        halaman=12,
                        kutipan="KDB maksimum untuk zona perumahan kepadatan sedang adalah 60%.",
                        terverifikasi=True,
                    )
                ],
                rekomendasi=RekomendasiOutput(tipe="numerik", target=60.0, saran="Kurangi luas tapak agar KDB tidak melampaui 60%."),
            )
        ],
        rekomendasi_sistem="Setuju Bersyarat",
        kesimpulan=KesimpulanOutput(langkah_berdampak=["Revisi desain agar KDB memenuhi ambang."]),
    )

    dumped = output.model_dump()
    restored = OutputL3.model_validate(dumped)
    assert restored == output


def test_rekomendasi_tipe_invalid_raises():
    with pytest.raises(ValidationError):
        RekomendasiOutput(tipe="bukan_tipe_valid", saran="x")


def test_rekomendasi_tipe_taksonomi_baru_valid():
    for tipe in ("kategorikal", "numerik", "numerik-mitigasi"):
        RekomendasiOutput(tipe=tipe, saran="x")
