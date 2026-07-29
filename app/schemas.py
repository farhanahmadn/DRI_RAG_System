"""Skema Pydantic — kontrak input (gate_hukum + impact_assessment dari back-end/L2) & output (JSON ke web).

Lihat CLAUDE.md § Kontrak Input / Kontrak Output. Angka & status final selalu berasal dari back-end/
kalkulator deterministik — LLM tidak pernah mengisi field numerik atau menentukan status/rekomendasi.
"""

from typing import Any, Literal

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Input: L2Assessment (gate_hukum + impact_assessment dari back-end)
# ---------------------------------------------------------------------------


class Koordinat(BaseModel):
    lat: float
    lon: float


class GeoJSONPoint(BaseModel):
    type: str
    coordinates: list[float]


class Lokasi(BaseModel):
    koordinat: Koordinat
    geojson: GeoJSONPoint
    rdtr_zone: str
    luas_lahan_m2: float


class DasarHukum(BaseModel):
    dokumen: str
    pasal: str
    kutipan: str


class ItbxTahap(BaseModel):
    status: Literal["I", "T", "B", "TB", "X"]
    lolos: bool
    kbli_diusulkan: str | None = None
    kegiatan_diusulkan: str | None = None
    kegiatan_diizinkan: list[str] = []
    kegiatan_terbatas: list[str] = []
    kegiatan_bersyarat: list[str] = []
    kegiatan_terbatas_bersyarat: list[str] = []
    keterangan_ketentuan: list[str] = []
    reason: str
    dasar_hukum: list[DasarHukum] = []


class ParameterIntensitas(BaseModel):
    usulan: float
    ambang_maks: float | None = None
    ambang_min: float | None = None
    memenuhi: bool
    satuan: str


class IntensitasTahap(BaseModel):
    status: Literal["MEMENUHI_SYARAT", "MELAMPAUI_BATAS"]
    # `lolos` & `reason` disimpan apa adanya (faithful ke payload asli) TAPI TIDAK dipakai sebagai
    # sumber kebenaran oleh adapter — terbukti bisa stale/kontradiktif dgn `status`/`parameter.*.memenuhi`
    # (lihat tests/fixtures/l2_sample_lolos_bersyarat.json: lolos=True padahal KDB melampaui ambang).
    lolos: bool
    parameter: dict[str, ParameterIntensitas]
    luas_tapak_m2: float | None = None
    jumlah_lantai: int | None = None
    luas_rth_usulan_m2: float | None = None
    reason: str
    dasar_hukum: list[DasarHukum] = []


class Tahapan(BaseModel):
    itbx: ItbxTahap
    intensitas: IntensitasTahap


class GateHukum(BaseModel):
    final_gate_status: Literal["Lolos", "Lolos Bersyarat", "Tidak Lolos"]
    # Rekomendasi versi back-end — informational only. Kita turunkan rekomendasi_sistem sendiri secara
    # deterministik (lihat app/reasoning/rekomendasi.py), TIDAK memakai field ini langsung.
    rekomendasi_sistem: str | None = None
    decisive_stage: str | None = None
    tahapan: Tahapan


class ImpactAssessment(BaseModel):
    dinilai: bool
    c_before: float | None = None
    c_after: float | None = None
    runoff_change_index: float | None = None
    impact_score: float | None = None
    impact_category: str | None = None
    calculation_details: str | None = None
    threshold_bands: dict[str, str] | None = None
    existing_surface_details: dict[str, Any] | None = None
    proposed_surface_details: dict[str, float] | None = None
    c_coefficients: dict[str, float] | None = None
    luas_lahan_m2: float | None = None
    data_confidence: str | None = None
    limitations: str | None = None


class MetaL2(BaseModel):
    data_confidence_keseluruhan: str | None = None
    # Diminta di kontrak, belum pernah muncul di fixture nyata — default None, tidak ditebak isinya.
    caveats: list[str] | None = None


class L2Assessment(BaseModel):
    application_id: int | None = None
    application_number: str | None = None
    timestamp: str | None = None
    lokasi: Lokasi
    gate_hukum: GateHukum
    impact_assessment: ImpactAssessment
    meta: MetaL2 | None = None


# ---------------------------------------------------------------------------
# Poin internal — jembatan adapter -> generator (bukan output LLM)
# ---------------------------------------------------------------------------


class PoinKonteks(BaseModel):
    poin_id: str
    kategori: str
    tipe_rekomendasi: Literal["kategorikal", "numerik", "numerik-mitigasi"]
    status: str
    fakta: dict[str, Any]
    dasar_hukum: list[DasarHukum] = []


class AdapterResult(BaseModel):
    poin: list[PoinKonteks]
    rekomendasi_sistem: str


# ---------------------------------------------------------------------------
# Output: JSON ke web
# ---------------------------------------------------------------------------


class SitasiOutput(BaseModel):
    citation_id: str
    dokumen: str
    pasal: str
    halaman: int
    kutipan: str
    terverifikasi: bool


class RekomendasiOutput(BaseModel):
    tipe: Literal["kategorikal", "numerik", "numerik-mitigasi"]
    target: float | str | None = None
    saran: str
    disclaimer: str | None = None


class PoinOutput(BaseModel):
    poin_id: str
    kategori: str
    status: str
    reasoning_pendek: str
    reasoning_panjang: str
    sitasi: list[SitasiOutput]
    rekomendasi: RekomendasiOutput
    low_confidence: bool = False


class RingkasanGateOutput(BaseModel):
    final_gate_status: Literal["Lolos", "Lolos Bersyarat", "Tidak Lolos"]
    kalimat: str


class RingkasanDampakOutput(BaseModel):
    impact_category: str | None = None
    impact_score: float | None = None
    kalimat: str


class KesimpulanOutput(BaseModel):
    langkah_berdampak: list[str]
    catatan_lokasi: str | None = None


class OutputL3(BaseModel):
    ringkasan_gate: RingkasanGateOutput
    ringkasan_dampak: RingkasanDampakOutput
    poin: list[PoinOutput]
    rekomendasi_sistem: str
    kesimpulan: KesimpulanOutput
