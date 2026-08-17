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


@pytest.mark.parametrize(
    "nama_file",
    [
        "l2_sample_lolos.json",
        "l2_sample_amplop_6191.json",
        "l2_sample_tidak_lolos.json",
        "l2_sample_amplop_8376.json",
        "l2_sample_itbx_x_tanpa_intensitas.json",
        "l2_sample_amplop_8913.json",
        "l2_sample_amplop_8090.json",
    ],
)
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


class TestKontrakBackendBerubah:
    """APP-2026-8376: back-end mulai kirim application_id UUID-string (bukan int) & geojson
    Polygon (bukan cuma Point) — 422 sebelum diperbaiki. Skema HARUS terima kedua format lama
    (fixture lain, int/Point) & baru (str/Polygon), bukan pilih salah satu."""

    def test_application_id_uuid_string_diterima(self):
        payload = json.loads((FIXTURES_DIR / "l2_sample_amplop_8376.json").read_text(encoding="utf-8"))
        assessment = L2Assessment.model_validate(payload["data"])
        assert assessment.application_id == "3a4a6f3a-6c9f-4a49-98ba-8c97c87d5048"

    def test_application_id_int_lama_tetap_diterima(self):
        payload = json.loads((FIXTURES_DIR / "l2_sample_lolos.json").read_text(encoding="utf-8"))
        assessment = L2Assessment.model_validate(payload["data"])
        assert assessment.application_id == 11
        assert isinstance(assessment.application_id, int)

    def test_geojson_polygon_diterima(self):
        payload = json.loads((FIXTURES_DIR / "l2_sample_amplop_8376.json").read_text(encoding="utf-8"))
        assessment = L2Assessment.model_validate(payload["data"])
        assert assessment.lokasi.geojson.type == "Polygon"
        # 3 level nesting (ring -> titik -> [lon,lat]) — HARUS lolos, bukan dipaksa flat.
        assert len(assessment.lokasi.geojson.coordinates[0]) == 5

    def test_geojson_point_lama_tetap_diterima(self):
        payload = json.loads((FIXTURES_DIR / "l2_sample_lolos.json").read_text(encoding="utf-8"))
        assessment = L2Assessment.model_validate(payload["data"])
        assert assessment.lokasi.geojson.type == "Point"

    def test_geojson_multipoint_diterima(self):
        # APP-2026-003: back-end kirim geometry "MultiPoint" (nesting 2 level, bukan Point/Polygon).
        payload = json.loads(
            (FIXTURES_DIR / "l2_sample_itbx_x_tanpa_intensitas.json").read_text(encoding="utf-8")
        )
        assessment = L2Assessment.model_validate(payload["data"])
        assert assessment.lokasi.geojson.type == "MultiPoint"

    def test_tahapan_intensitas_absen_diterima(self):
        # APP-2026-003: back-end OMIT `tahapan.intensitas` sama sekali kalau gate berhenti di ITBX
        # (status "X") — sebelumnya field wajib -> 422. Field ini opsional & default None.
        payload = json.loads(
            (FIXTURES_DIR / "l2_sample_itbx_x_tanpa_intensitas.json").read_text(encoding="utf-8")
        )
        assessment = L2Assessment.model_validate(payload["data"])
        assert assessment.gate_hukum.tahapan.intensitas is None

    def test_tahapan_intensitas_lama_tetap_diterima(self):
        payload = json.loads((FIXTURES_DIR / "l2_sample_lolos.json").read_text(encoding="utf-8"))
        assessment = L2Assessment.model_validate(payload["data"])
        assert assessment.gate_hukum.tahapan.intensitas is not None

    def test_rdtr_subzone_diterima(self):
        # APP-2026-8090: field baru dari BE (permintaan sebelumnya) — kode sub-zona presisi.
        payload = json.loads((FIXTURES_DIR / "l2_sample_amplop_8090.json").read_text(encoding="utf-8"))
        assessment = L2Assessment.model_validate(payload["data"])
        assert assessment.lokasi.rdtr_subzone == "P-1"
        assert assessment.lokasi.rdtr_subzone_source == "geojson_overlay"

    def test_rdtr_subzone_absen_tetap_none(self):
        # Fixture lama (pra-8090) tak punya field ini sama sekali -> None, bukan error.
        payload = json.loads((FIXTURES_DIR / "l2_sample_lolos.json").read_text(encoding="utf-8"))
        assessment = L2Assessment.model_validate(payload["data"])
        assert assessment.lokasi.rdtr_subzone is None
        assert assessment.lokasi.rdtr_subzone_source is None

    def test_c_coefficients_boleh_campur_label_string(self):
        # APP-2026-8913/-7012/-5397: back-end selipkan "Kelas_Atap": "<label string>" berdampingan
        # dgn koefisien numerik (mis. "C_Atap_KBLI": 0.7) dlm SATU dict `c_coefficients` yang sama —
        # `dict[str, float]` gagal validasi (422) begitu ada key label non-numerik.
        payload = json.loads((FIXTURES_DIR / "l2_sample_amplop_8913.json").read_text(encoding="utf-8"))
        assessment = L2Assessment.model_validate(payload["data"])
        koef = assessment.impact_assessment.c_coefficients
        assert koef["Kelas_Atap"] == "Perdagangan Sekeliling Pusat Kota"
        assert koef["C_Atap_KBLI"] == 0.7


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
        catatan_global=[],
        low_confidence_keseluruhan=False,
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
