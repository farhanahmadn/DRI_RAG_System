"""Tipe bangunan/kegiatan yang diusulkan, untuk gate dampak hidrologi.

Permintaan (rapat 2026-10-16): gate dampak hidrologi harus membawa informasi TIPE BANGUNAN apa yang ingin
dibuat. Alasannya substantif: koefisien limpasan atap (C) pada perhitungan BE ditentukan oleh kelas
bangunan yang diturunkan dari KBLI kegiatan, jadi dua permohonan dengan luas sama tetapi kegiatan berbeda
(rumah tinggal vs pergudangan) menghasilkan indeks limpasan berbeda.

Sumber data pada payload BE: nama kegiatan dan KBLI hanya ada di gate ITBX; kelas bangunan ada di
`impact_assessment.c_coefficients.Kelas_Atap` (atau `detailed_surface_breakdown["Bangunan/Atap"].kelas`).
`impact_assessment` sendiri belum membawa nama kegiatan secara eksplisit (diminta ke BE).
"""

from __future__ import annotations

from app.schemas import L2Assessment


def ambil_tipe_bangunan(assessment: L2Assessment) -> dict:
    """{"kegiatan", "kbli", "kelas_atap", "jumlah_lantai"} — hanya kunci yang terisi."""
    itbx = assessment.gate_hukum.tahapan.itbx
    impact = assessment.impact_assessment
    kelas = (impact.c_coefficients or {}).get("Kelas_Atap")
    if not kelas:
        atap = (impact.detailed_surface_breakdown or {}).get("Bangunan/Atap")
        kelas = atap.get("kelas") if isinstance(atap, dict) else None
    intensitas = assessment.gate_hukum.tahapan.intensitas
    hasil = {
        "kegiatan": itbx.kegiatan_diusulkan,
        "kbli": itbx.kbli_diusulkan,
        "kelas_atap": kelas,
        "jumlah_lantai": intensitas.jumlah_lantai if intensitas is not None else None,
    }
    return {k: v for k, v in hasil.items() if v}


def frasa_tipe_bangunan(tipe: dict) -> str:
    """" untuk kegiatan Rumah Tinggal (kelas bangunan: Perumahan Tunggal)" atau "" bila tak ada data."""
    if not tipe.get("kegiatan") and not tipe.get("kelas_atap"):
        return ""
    frasa = f" untuk kegiatan {tipe['kegiatan']}" if tipe.get("kegiatan") else " untuk bangunan"
    if tipe.get("kelas_atap"):
        frasa += f" (kelas bangunan: {tipe['kelas_atap']})"
    return frasa
