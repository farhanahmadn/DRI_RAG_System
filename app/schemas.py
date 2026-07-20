"""Skema Pydantic — kontrak input (jejak aturan dari back-end) & output (JSON ke web).

Lihat CLAUDE.md § Kontrak Input / Kontrak Output. Angka (skor, kontribusi, target rekomendasi)
selalu berasal dari back-end/kalkulator deterministik — LLM tidak pernah mengisi field numerik ini.
"""

from typing import Literal

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Input: jejak aturan dari back-end
# ---------------------------------------------------------------------------


class FaktaSpasial(BaseModel):
    in_lp2b: bool | None = None
    banjir: bool | None = None
    tingkat_banjir: Literal["Tinggi", "Sedang", "Rendah"] | None = None
    resapan: bool | None = None
    in_sempadan: bool | None = None
    jarak_sungai_m: float | None = None
    nama_sungai: str | None = None
    arah: str | None = None


class IndikatorJejak(BaseModel):
    poin_id: str
    kategori: str
    bobot: float
    skor: float
    kontribusi: float
    nilai_input: float | str
    ambang: float | str
    operator: str
    formula: str
    zona: str | None = None
    luas_lahan: float | None = None
    referensi_hukum: list[str] = []
    fakta_spasial: FaktaSpasial | None = None
    target_rekomendasi: dict[str, float] | None = None


class JejakAturanRequest(BaseModel):
    skor_total: float
    level: str | None = None
    zona: str | None = None
    indikator: list[IndikatorJejak]


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
    tipe: Literal["numerik", "kegiatan", "lokasional"]
    target: float | str | None = None
    saran: str
    disclaimer: str | None = None


class PoinOutput(BaseModel):
    poin_id: str
    kategori: str
    status: str
    kontribusi: float
    reasoning_pendek: str
    reasoning_panjang: str
    sitasi: list[SitasiOutput]
    rekomendasi: RekomendasiOutput
    low_confidence: bool = False


class RingkasanOutput(BaseModel):
    skor_total: float
    level: str
    kalimat: str


class KesimpulanOutput(BaseModel):
    langkah_berdampak: list[str]
    catatan_lokasi: str | None = None


class OutputPreCheck(BaseModel):
    ringkasan: RingkasanOutput
    poin: list[PoinOutput]
    kesimpulan: KesimpulanOutput
