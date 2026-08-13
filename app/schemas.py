"""Skema Pydantic — kontrak input (gate_hukum + impact_assessment dari back-end/L2) & output (JSON ke web).

Lihat CLAUDE.md § Kontrak Input / Kontrak Output. Angka & status final selalu berasal dari back-end/
kalkulator deterministik — LLM tidak pernah mengisi field numerik atau menentukan status/rekomendasi.
"""

from typing import Any, Literal

from pydantic import BaseModel, model_validator

# ---------------------------------------------------------------------------
# Input: L2Assessment (gate_hukum + impact_assessment dari back-end)
# ---------------------------------------------------------------------------


class Koordinat(BaseModel):
    lat: float
    lon: float


class GeoJSONPoint(BaseModel):
    type: str
    # `Any`, BUKAN `list[float]` — GeoJSON "coordinates" berbeda kedalaman nesting per geometry
    # (Point: [lon,lat] datar; Polygon: [[[lon,lat], ...]] 3 level; dst). Back-end pernah kirim
    # Polygon (APP-2026-8376) padahal kontrak awal cuma Point -> `list[float]` gagal validasi
    # (422) utk SEMUA geometry selain Point. Field ini TIDAK PERNAH dibaca oleh kode reasoning
    # (cuma lewat, lihat app/sanitize.py — koordinat presisi memang sengaja tak pernah sampai ke
    # LLM/query retrieval), jadi longgarkan validasi struktur di sini aman.
    coordinates: Any


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
    # sumber kebenaran oleh adapter — pernah teramati stale/kontradiktif dgn `status`/
    # `parameter.*.memenuhi` di data back-end nyata (lolos=True padahal KDB melampaui ambang).
    lolos: bool
    parameter: dict[str, ParameterIntensitas]
    luas_tapak_m2: float | None = None
    jumlah_lantai: int | None = None
    luas_rth_usulan_m2: float | None = None
    # Nullable: back-end kirim null saat intensitas BUKAN decisive_stage (gate sudah short-circuit
    # di ITBX duluan, lihat tests/fixtures/l2_sample_tidak_lolos.json).
    reason: str | None = None
    dasar_hukum: list[DasarHukum] = []


class Tahapan(BaseModel):
    itbx: ItbxTahap
    # Nullable (APP-2026-003): back-end SEBELUMNYA selalu kirim `intensitas` penuh (walau
    # `decisive_stage="itbx"`, lihat tests/fixtures/l2_sample_tidak_lolos.json — reason=null tapi
    # objek lengkap ada). Payload baru (itbx status "X", gate berhenti total di ITBX) OMIT key
    # `intensitas` sama sekali -> field wajib gagal validasi (422) utk kasus "Tidak Lolos" murni
    # ITBX. Field ini TIDAK PERNAH benar-benar kosong kalau gate lolos sampai tahap intensitas
    # dievaluasi (lihat app/adapter.py::_bangun_poin_intensitas utk penanganan None -> poin
    # "Tidak Dinilai", pola sama seperti impact_assessment.dinilai=False).
    intensitas: IntensitasTahap | None = None


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
    # `Any`, BUKAN `float` (APP-2026-8913/-7012/-5397): back-end mulai selipkan field label
    # string di sini juga (mis. "Kelas_Atap": "Perdagangan Sekeliling Pusat Kota") berdampingan dgn
    # koefisien numerik (mis. "C_Atap_KBLI": 0.7) dalam SATU dict yang sama -> `dict[str, float]`
    # gagal validasi (422) begitu ada key label. Field ini cuma diteruskan apa adanya ke
    # fakta["mitigasi"]["c_coefficients_referensi"] (calculator.py) & prompt (tak pernah dihitung
    # ulang secara aritmatika di kode kita), jadi longgarkan tipe value-nya aman.
    c_coefficients: dict[str, Any] | None = None
    luas_lahan_m2: float | None = None
    data_confidence: str | None = None
    limitations: str | None = None


class MetaL2(BaseModel):
    data_confidence_keseluruhan: str | None = None
    # Diminta di kontrak, belum pernah muncul di fixture nyata — default None, tidak ditebak isinya.
    caveats: list[str] | None = None


class L2Assessment(BaseModel):
    # int (fixture lama, mis. 21) ATAU string UUID (back-end baru, APP-2026-8376: "3a4a6f3a-...").
    # Field ini identitas murni — tidak pernah dipakai reasoning/kalkulasi apa pun (sengaja
    # dikecualikan dari fakta yang sampai ke LLM, lihat app/sanitize.py) — longgarkan tipe,
    # bukan pilih salah satu format & tolak yang lain.
    application_id: int | str | None = None
    application_number: str | None = None
    timestamp: str | None = None
    lokasi: Lokasi
    gate_hukum: GateHukum
    impact_assessment: ImpactAssessment
    meta: MetaL2 | None = None


class L2Envelope(BaseModel):
    """Terima L2Assessment polos ATAU amplop back-end {statusCode, message, data} — bentuk asli
    respons L2 Spatial Risk Assessment sungguhan (lihat tests/fixtures/l2_sample_*.json). Endpoint
    /reasoning pakai model ini sebagai body, lalu teruskan `.data` (lihat app/api/main.py)."""

    statusCode: int | None = None
    message: str | None = None
    data: L2Assessment

    @model_validator(mode="before")
    @classmethod
    def _bungkus_kalau_polos(cls, v):
        # Sudah ber-amplop (ada 'data') -> pakai apa adanya.
        # Payload polos (langsung lokasi/gate_hukum/...) -> bungkus jadi {'data': v}.
        if isinstance(v, dict) and "data" in v:
            return v
        return {"data": v}


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
    # Nama zona pemohon APA ADANYA dari `assessment.lokasi.rdtr_zone` (mis. "Zona Perumahan") — BUKAN
    # kode sub-zona presisi (back-end tak menyediakannya). Dipakai app/reasoning/generator.py utk
    # filter KELUARGA zona saat retrieval fallback (cegah kontaminasi lintas-zona, lihat APP-2026-6191).
    zona: str | None = None


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
    decisive_stage: str | None = None
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
    catatan_global: list[str] = []
    low_confidence_keseluruhan: bool
