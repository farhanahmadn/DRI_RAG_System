"""Kalkulator target intensitas & arah mitigasi dampak — murni deterministik, TIDAK ADA panggilan
LLM di sini. CLAUDE.md / docs/Blueprint-Adaptasi-Model-L2-Gate-Impact.md §5: angka & verdict dari
back-end/kode, LLM hanya pelapis.

Fungsi di sini TIDAK PERNAH menyentuh `status`/`memenuhi`/`final_gate_status` (ground truth
back-end) — murni menghitung angka target ilustratif untuk rekomendasi. Ambang yang null ditangani
dengan mengembalikan `None` (parameter dilewati), BUKAN raise — kontrak L2 eksplisit bilang ambang
bisa null sementara.
"""

from app.reasoning.rekomendasi import KATEGORI_DAMPAK_BERSYARAT
from app.schemas import ImpactAssessment, IntensitasTahap, ParameterIntensitas


def _ke_fraksi(nilai: float, satuan: str) -> float:
    """"persen" -> nilai/100, "rasio" (atau satuan lain) -> nilai apa adanya."""
    return nilai / 100 if satuan == "persen" else nilai


def hitung_target_kdb(param: ParameterIntensitas, luas_lahan_m2: float | None) -> dict[str, float] | None:
    if param.ambang_maks is None or param.usulan <= param.ambang_maks:
        return None
    selisih = param.usulan - param.ambang_maks
    hasil = {"target_kdb": param.ambang_maks, "selisih": selisih}
    if luas_lahan_m2 is not None:
        hasil["footprint_maks_m2"] = luas_lahan_m2 * _ke_fraksi(param.ambang_maks, param.satuan)
    return hasil


def hitung_target_klb(param: ParameterIntensitas, luas_lahan_m2: float | None) -> dict[str, float] | None:
    if param.ambang_maks is None or param.usulan <= param.ambang_maks:
        return None
    selisih = param.usulan - param.ambang_maks
    hasil = {"target_klb": param.ambang_maks, "selisih": selisih}
    if luas_lahan_m2 is not None:
        hasil["luas_lantai_maks_m2"] = luas_lahan_m2 * _ke_fraksi(param.ambang_maks, param.satuan)
    return hasil


def hitung_target_kdh(
    param: ParameterIntensitas, luas_lahan_m2: float | None, luas_rth_usulan_m2: float | None
) -> dict[str, float] | None:
    if param.ambang_min is None or param.usulan >= param.ambang_min:
        return None
    selisih = param.ambang_min - param.usulan  # positif = kurang
    hasil = {"target_kdh": param.ambang_min, "selisih": selisih}
    if luas_lahan_m2 is not None:
        rth_dibutuhkan = luas_lahan_m2 * _ke_fraksi(param.ambang_min, param.satuan)
        hasil["rth_dibutuhkan_m2"] = rth_dibutuhkan
        if luas_rth_usulan_m2 is not None:
            hasil["rth_kurang_m2"] = max(rth_dibutuhkan - luas_rth_usulan_m2, 0.0)
    return hasil


_HANDLER_PARAMETER = {"kdb": hitung_target_kdb, "klb": hitung_target_klb, "kdh": hitung_target_kdh}


def hitung_target_intensitas(intensitas: IntensitasTahap, luas_lahan_m2: float | None) -> dict[str, dict]:
    """{nama_parameter: target_dict} HANYA untuk parameter yang melampaui ambang. Parameter yang
    patuh atau datanya tak lengkap (ambang null) dilewati (bukan error) — status/memenuhi back-end
    TIDAK disentuh di sini, ini murni angka target ilustratif untuk rekomendasi.
    """
    hasil: dict[str, dict] = {}
    for nama, param in intensitas.parameter.items():
        if nama == "kdh":
            target = hitung_target_kdh(param, luas_lahan_m2, intensitas.luas_rth_usulan_m2)
        else:
            handler = _HANDLER_PARAMETER.get(nama)
            target = handler(param, luas_lahan_m2) if handler else None
        if target is not None:
            hasil[nama] = target
    return hasil


def normalisasi_kategori_dampak(raw: str | None) -> str | None:
    """"SEDANG" -> "Sedang", "SANGAT TINGGI" -> "Sangat Tinggi", None -> None."""
    return raw.strip().title() if raw else None


def sarankan_arah_mitigasi_dampak(impact: ImpactAssessment) -> dict:
    """Arah mitigasi KUALITATIF. TIDAK menghitung ulang c_after/runoff_change_index/impact_score —
    rumus C masih volatile di back-end (sudah direvisi berkali-kali), DILARANG di-hardcode di sini.
    Angka di `*_referensi` HANYA echo apa adanya dari back-end, bukan dihitung/didefinisikan ulang.
    """
    if not impact.dinilai:
        return {"perlu_mitigasi": False, "arah": []}

    kategori = normalisasi_kategori_dampak(impact.impact_category)
    if kategori not in KATEGORI_DAMPAK_BERSYARAT:
        return {"perlu_mitigasi": False, "arah": []}

    hasil: dict = {
        "perlu_mitigasi": True,
        "arah": [
            "turunkan KDB (proporsi tapak/bangunan)",
            "naikkan KDH/RTH",
            "sumur resapan",
            "kolam retensi",
        ],
    }
    if impact.c_coefficients:
        hasil["c_coefficients_referensi"] = impact.c_coefficients
    if impact.calculation_details:
        hasil["calculation_details_referensi"] = impact.calculation_details
    return hasil
