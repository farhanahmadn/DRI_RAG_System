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


_ITBX_LABEL = {
    "I": "Diizinkan",
    "T": "Terbatas",
    "B": "Bersyarat",
    "X": "Dilarang",
}


def rakit_status_kegiatan(indikator: IndikatorJejak) -> str | None:
    """Rakit FAKTA klasifikasi ITBX (mis. "KLASIFIKASI: X...") untuk indikator kegiatan.

    Sama semangat dengan `rakit_status_numerik`: klasifikasi (I/T/B/X) sudah ada di jejak
    (`nilai_input`), disuntik ke prompt sebagai fakta supaya LLM tidak perlu (dan tidak boleh)
    menyimpulkan sendiri klasifikasinya. `None` untuk indikator non-kegiatan atau klasifikasi yang
    tidak dikenal.
    """
    if klasifikasi_tipe_rekomendasi(indikator.kategori) != "kegiatan":
        return None

    klasifikasi = str(indikator.nilai_input).strip().upper()
    label = _ITBX_LABEL.get(klasifikasi)
    if label is None:
        return None

    zona = f" di zona {indikator.zona}" if indikator.zona else ""
    return (
        f"KLASIFIKASI: {klasifikasi} ({label}). Kegiatan ini{zona} tergolong {label} "
        "menurut matriks ITBX RDTR."
    )


def rakit_status_banjir(indikator: IndikatorJejak) -> str | None:
    """Rakit FAKTA klasifikasi tingkat risiko banjir (mis. "KLASIFIKASI: TINGGI...").

    Tingkat (Tinggi/Sedang/Rendah) sudah ditentukan GIS, disuntik ke prompt sebagai fakta supaya
    LLM tidak menyimpulkan sendiri tingkat risikonya. `None` kalau kategori bukan banjir atau
    `fakta_spasial.tingkat_banjir` tidak ada.
    """
    if "banjir" not in indikator.kategori.lower():
        return None
    if indikator.fakta_spasial is None or indikator.fakta_spasial.tingkat_banjir is None:
        return None

    tingkat = indikator.fakta_spasial.tingkat_banjir
    return (
        f"KLASIFIKASI: {tingkat.upper()}. Lokasi ini tergolong risiko banjir tingkat {tingkat} "
        "berdasarkan data spasial (GIS)."
    )


def rakit_status_sempadan(indikator: IndikatorJejak) -> str | None:
    """Rakit FAKTA verdict jarak sempadan sungai (mis. "STATUS: KURANG DARI MINIMUM...").

    KASUS BUG notebook lama: jarak ke sungai yang KURANG dari sempadan minimum berarti MELANGGAR
    (bukan "melebihi"). Dihitung jujur dari angka (nilai_input=jarak aktual, ambang=jarak minimum,
    operator ge/gt), disuntik sebagai fakta supaya LLM tidak menyimpulkan arah sendiri. `None`
    kalau kategori bukan sempadan atau datanya tidak bisa dibandingkan.
    """
    if "sempadan" not in indikator.kategori.lower():
        return None

    try:
        jarak = float(indikator.nilai_input)
        minimum = float(indikator.ambang)
    except (TypeError, ValueError):
        return None

    arah = _OPERATOR_ALIASES.get(indikator.operator)
    if arah is None:
        return None

    if arah in ("ge", "gt"):
        if jarak < minimum:
            return (
                f"STATUS: KURANG DARI MINIMUM (MELANGGAR SEMPADAN). Jarak ke sungai ({jarak} m) "
                f"< sempadan minimum ({minimum} m)."
            )
        return f"STATUS: SESUAI. Jarak ke sungai ({jarak} m) memenuhi sempadan minimum ({minimum} m)."
    else:  # le, lt — jarang dipakai utk sempadan, tapi tetap dihitung jujur
        if jarak > minimum:
            return f"STATUS: MELEBIHI. Jarak ke sungai ({jarak} m) > batas ({minimum} m)."
        return f"STATUS: SESUAI. Jarak ke sungai ({jarak} m) <= batas ({minimum} m)."


def rakit_fakta_verdict(indikator: IndikatorJejak) -> str | None:
    """Dispatcher fakta verdict — numerik/kegiatan/sebagian lokasional punya fakta khusus."""
    tipe = klasifikasi_tipe_rekomendasi(indikator.kategori)
    if tipe == "numerik":
        return rakit_status_numerik(indikator)
    if tipe == "kegiatan":
        return rakit_status_kegiatan(indikator)
    if tipe == "lokasional":
        k = indikator.kategori.lower()
        if "banjir" in k:
            return rakit_status_banjir(indikator)
        if "sempadan" in k:
            return rakit_status_sempadan(indikator)
        return None  # LP2B, resapan: fakta_spasial sudah cukup sbg konteks
    return None


def pilih_target_utama(target: dict[str, float] | None) -> float | None:
    """Pilih satu angka representatif dari hasil `hitung_target_rekomendasi` untuk `RekomendasiOutput.target`."""
    if target is None:
        return None
    if "target_maks" in target:
        return target["target_maks"]
    return target.get("target")
