"""Kalkulator target rekomendasi — murni deterministik, TIDAK ADA panggilan LLM di sini.

CLAUDE.md § Konvensi kode: "Fungsi yang menghasilkan angka = deterministik & ada unit test.
LLM tidak boleh menyentuh angka final." Kalau back-end sudah mengirim `target_rekomendasi`,
fungsi di sini TIDAK menghitung ulang — cuma dipakai sebagai fallback.
"""

from typing import Literal

from app.schemas import IndikatorJejak

_NUMERIK_TOKENS = {"kdb", "klb", "kdh"}
_LOKASIONAL_TOKENS = {"banjir", "resapan", "sempadan", "lp2b"}

_OPERATOR_ALIASES = {
    "<=": "le",
    "le": "le",
    "<": "lt",
    "lt": "lt",
    ">=": "ge",
    "ge": "ge",
    ">": "gt",
    "gt": "gt",
}


def klasifikasi_tipe_rekomendasi(kategori: str) -> Literal["numerik", "kegiatan", "lokasional"]:
    """Klasifikasi taksonomi rekomendasi (CLAUDE.md § Kontrak Output) dari nama kategori indikator."""
    k = kategori.lower()
    if any(tok in k for tok in _NUMERIK_TOKENS):
        return "numerik"
    if any(tok in k for tok in _LOKASIONAL_TOKENS):
        return "lokasional"
    return "kegiatan"


def hitung_target_rekomendasi(indikator: IndikatorJejak) -> dict[str, float] | None:
    """Hitung target rekomendasi untuk indikator NUMERIK (KDB/KLB/KDH).

    - Kalau back-end sudah mengirim `target_rekomendasi`, dikembalikan apa adanya (tidak dihitung ulang).
    - Kalau indikator bukan numerik (kegiatan/lokasional), kembalikan `None`.
    - Kalau numerik dan `luas_lahan` tersedia: `target_maks = luas_lahan * ambang` (mis. footprint_maks,
      luas lantai maks, luas hijau min), `nilai_input` diperlakukan sebagai nilai aktual absolut (m2)
      yang sepadan.
    - Kalau numerik tanpa `luas_lahan`: fallback generik, `target = ambang` (nilai koefisien itu
      sendiri), dibandingkan langsung dengan `nilai_input` (rasio).
    - `selisih` positif SELALU berarti ada masalah (melebihi batas maks, atau kurang dari batas
      minimal); negatif/nol berarti patuh dengan margin.
    """
    if indikator.target_rekomendasi is not None:
        return indikator.target_rekomendasi

    if klasifikasi_tipe_rekomendasi(indikator.kategori) != "numerik":
        return None

    try:
        nilai_input = float(indikator.nilai_input)
        ambang = float(indikator.ambang)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Indikator numerik {indikator.poin_id!r} butuh nilai_input & ambang numerik, "
            f"didapat nilai_input={indikator.nilai_input!r} ambang={indikator.ambang!r}."
        ) from exc

    arah = _OPERATOR_ALIASES.get(indikator.operator)
    if arah is None:
        raise ValueError(
            f"Operator {indikator.operator!r} pada indikator numerik {indikator.poin_id!r} tidak "
            "didukung untuk penghitungan target otomatis — minta back-end kirim target_rekomendasi."
        )

    if indikator.luas_lahan is not None:
        target_efektif = indikator.luas_lahan * ambang
        key = "target_maks"
    else:
        target_efektif = ambang
        key = "target"

    if arah in ("le", "lt"):
        selisih = nilai_input - target_efektif
    else:  # ge, gt
        selisih = target_efektif - nilai_input

    return {key: target_efektif, "selisih": selisih}


def rakit_status_numerik(indikator: IndikatorJejak) -> str | None:
    """Rakit FAKTA verdict perbandingan (mis. "STATUS: MELEBIHI...") untuk indikator numerik.

    Dihitung JUJUR dari angka (bukan diasumsikan selalu "melanggar"), lalu disuntik ke prompt LLM
    supaya LLM tidak perlu (dan tidak boleh) menyimpulkan sendiri arah pelanggaran — pelajaran dari
    bug LLM salah simpul arah (mis. notebook Sempadan). `None` untuk indikator non-numerik atau
    yang datanya tidak bisa dibandingkan.
    """
    if klasifikasi_tipe_rekomendasi(indikator.kategori) != "numerik":
        return None

    try:
        target = hitung_target_rekomendasi(indikator)
    except (TypeError, ValueError):
        return None

    batas = pilih_target_utama(target)
    if batas is None:
        return None

    try:
        nilai_input = float(indikator.nilai_input)
    except (TypeError, ValueError):
        return None

    arah = _OPERATOR_ALIASES.get(indikator.operator)
    if arah is None:
        return None

    if arah in ("le", "lt"):
        if nilai_input > batas:
            return f"STATUS: MELEBIHI. Nilai aktual ({nilai_input}) > batas ({batas})."
        return f"STATUS: SESUAI. Nilai aktual ({nilai_input}) <= batas ({batas})."
    else:  # ge, gt
        if nilai_input < batas:
            return f"STATUS: KURANG. Nilai aktual ({nilai_input}) < batas ({batas})."
        return f"STATUS: SESUAI. Nilai aktual ({nilai_input}) >= batas ({batas})."


def pilih_target_utama(target: dict[str, float] | None) -> float | None:
    """Pilih satu angka representatif dari hasil `hitung_target_rekomendasi` untuk `RekomendasiOutput.target`."""
    if target is None:
        return None
    if "target_maks" in target:
        return target["target_maks"]
    return target.get("target")
