"""Kalkulator target intensitas & arah mitigasi dampak — murni deterministik, TIDAK ADA panggilan
LLM di sini. CLAUDE.md / docs/Blueprint-Adaptasi-Model-L2-Gate-Impact.md §5: angka & verdict dari
back-end/kode, LLM hanya pelapis.

Fungsi di sini TIDAK PERNAH menyentuh `status`/`memenuhi`/`final_gate_status` (ground truth
back-end) — murni menghitung angka target ilustratif untuk rekomendasi. Ambang yang null ditangani
dengan mengembalikan `None` (parameter dilewati), BUKAN raise — kontrak L2 eksplisit bilang ambang
bisa null sementara.
"""

import re

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


def pilih_target_utama_intensitas(target_map: dict[str, dict]) -> float | None:
    """Satu angka representatif dari hasil `hitung_target_intensitas()` untuk `RekomendasiOutput.target`.

    Kalau lebih dari satu parameter melanggar, ambil yang pertama ditemukan (representatif, bukan
    klaim "paling penting") — narasi lengkap tetap ada di fakta lengkap, ini cuma satu angka ringkas.
    """
    for info in target_map.values():
        for key in ("target_kdb", "target_klb", "target_kdh"):
            if key in info:
                return info[key]
    return None


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


# ---------------------------------------------------------------------------
# Target mitigasi dampak — angka ambang (BUKAN rumus C/index dihitung ulang).
# ---------------------------------------------------------------------------

# Urutan RINGAN -> BERAT. Dipakai HANYA utk mencari kategori "satu tingkat lebih ringan" dari
# threshold_bands yang SUDAH DIBERI back-end — bukan definisi baru, cuma label yang sudah dipakai
# di seluruh sistem (app/reasoning/prompts.py, rekomendasi.py::KATEGORI_DAMPAK_BERSYARAT).
_URUTAN_KATEGORI_DAMPAK = ("Rendah", "Sedang", "Tinggi", "Sangat Tinggi")

# Pola sama seperti app/reasoning/guardrail.py::_RE_BAND (dipakai utk cek konsistensi, BUKAN target)
# — sengaja diduplikasi kecil di sini alih-alih di-impor, supaya calculator.py tidak balik bergantung
# ke guardrail.py (arah dependency yang ada sekarang: guardrail -> calculator, bukan sebaliknya).
_RE_BAND_DAMPAK = re.compile(r"(?P<op>[<>])?\s*(?P<n1>\d+(?:\.\d+)?)\s*(?:-\s*(?P<n2>\d+(?:\.\d+)?))?")


def hitung_target_mitigasi_dampak(impact: ImpactAssessment) -> dict:
    """Ambang `runoff_change_index` yang perlu dicapai (turun DI BAWAH ambang ini) supaya kategori
    dampak pindah ke SATU TINGKAT LEBIH RINGAN — murni aritmatika ambang atas `threshold_bands`
    (SUDAH diberi back-end) vs `runoff_change_index` (SUDAH diberi back-end). TIDAK menghitung ulang
    rumus C/index sama sekali (itu wewenang back-end, lihat docstring `sarankan_arah_mitigasi_dampak`
    di atas — rumus itu masih volatile & DILARANG di-hardcode di sini).

    Bug ditemukan live (APP-2026-8376, kategori "Sedang"): gate di sini SEBELUMNYA cuma "ada
    kategori lebih ringan?" (posisi != 0) — beda dari gate `sarankan_arah_mitigasi_dampak`
    (`KATEGORI_DAMPAK_BERSYARAT` = Tinggi/Sangat Tinggi saja). Akibatnya poin dampak "Sedang"
    (aman, `mitigasi.perlu_mitigasi=False`, saran="Tidak diperlukan tindakan khusus...") tetap dapat
    `target=1.5` — kontradiktif dgn narasinya sendiri. Gate DISAMAKAN persis dgn
    `sarankan_arah_mitigasi_dampak` — SATU sumber kebenaran "kapan mitigasi & targetnya berlaku",
    bukan dua gerbang independen yang bisa berbeda pendapat.

    Return {} (bukan angka) kalau: index/threshold_bands tidak ada, kategori BUKAN Tinggi/Sangat
    Tinggi (selaras `KATEGORI_DAMPAK_BERSYARAT`), band kategori target tak bisa di-parse, atau band
    target berbentuk "> n" (tak ada batas atas terhingga utk dijadikan target).
    """
    index = impact.runoff_change_index
    bands = impact.threshold_bands
    kategori = normalisasi_kategori_dampak(impact.impact_category)
    if index is None or not bands or kategori not in KATEGORI_DAMPAK_BERSYARAT:
        return {}

    posisi = _URUTAN_KATEGORI_DAMPAK.index(kategori)
    kategori_target = _URUTAN_KATEGORI_DAMPAK[posisi - 1]
    rentang_target = bands.get(kategori_target)
    if not rentang_target:
        return {}

    m = _RE_BAND_DAMPAK.search(rentang_target)
    if not m:
        return {}
    op, n1_str, n2_str = m.group("op"), m.group("n1"), m.group("n2")
    if n2_str is not None:
        ambang = float(n2_str)  # rentang "a-b" -> ambang atas = b
    elif op == "<":
        ambang = float(n1_str)  # "< n" -> ambang atas = n
    else:
        return {}  # "> n" (band target tak berbatas atas) -> tak ada target berarti

    return {
        "runoff_change_index_maks": ambang,
        "kategori_target": kategori_target,
        "index_saat_ini": index,
    }


def pilih_target_mitigasi_dampak(target_mitigasi: dict) -> float | None:
    """Satu angka representatif dari `hitung_target_mitigasi_dampak()` utk `RekomendasiOutput.target`
    — pola sama seperti `pilih_target_utama_intensitas`."""
    return target_mitigasi.get("runoff_change_index_maks")


# ---------------------------------------------------------------------------
# Presentasi target intensitas/mitigasi — kalimat siap-kutip + langkah_konkret terstruktur
# (RekomendasiOutput.langkah_konkret, schemas.py, additive 2026-08-18).
# ---------------------------------------------------------------------------


def format_target_parameter(info: dict) -> str:
    """Ubah dict target mentah dari `hitung_target_kdb/klb/kdh` di atas (mis. {'target_kdb': 10,
    'selisih': 30, 'footprint_maks_m2': 85.0}) jadi SATU kalimat siap-kutip, BUKAN dict Python
    mentah. Dipakai app/reasoning/prompts.py (fakta ke LLM) & bangun_langkah_konkret_intensitas
    di bawah (field terstruktur) — satu sumber kebenaran teks, tak digandakan di dua tempat.

    Kenapa kalimat, bukan dict mentah: LLM jauh lebih mudah menyalin kalimat lengkap drpd
    mem-parse repr dict & memilih field yang relevan sendiri — dict mentah terbukti live bikin LLM
    cenderung cuma sebut angka ambang persen (mis. "KDB harus 10%"), jarang angka fisik m² yang
    justru paling actionable buat reviewer (lihat SYSTEM_PROMPT aturan #11, prompts.py).
    """
    if "target_kdb" in info:
        kalimat = f"KDB harus turun ke maksimal {info['target_kdb']}% (selisih {info['selisih']} poin dari usulan)"
        if "footprint_maks_m2" in info:
            kalimat += f" → luas lantai dasar bangunan maksimal {info['footprint_maks_m2']:.1f} m²"
        return kalimat
    if "target_klb" in info:
        kalimat = f"KLB harus turun ke maksimal {info['target_klb']} (selisih {info['selisih']} dari usulan)"
        if "luas_lantai_maks_m2" in info:
            kalimat += f" → luas total lantai bangunan maksimal {info['luas_lantai_maks_m2']:.1f} m²"
        return kalimat
    if "target_kdh" in info:
        kalimat = f"KDH harus naik ke minimal {info['target_kdh']}% (kurang {info['selisih']} poin dari usulan)"
        if "rth_dibutuhkan_m2" in info:
            kalimat += f" → RTH dibutuhkan minimal {info['rth_dibutuhkan_m2']:.1f} m²"
        if info.get("rth_kurang_m2"):
            kalimat += f" (RTH yang sudah diusulkan pemohon masih kurang {info['rth_kurang_m2']:.1f} m²)"
        return kalimat
    # Jaring pengaman — harusnya tak pernah kena selama fungsi hitung_target_* di atas konsisten
    # dgn 3 bentuk di atas (target_kdb/target_klb/target_kdh), tapi jangan diam-diam sembunyikan
    # data kalau meleset.
    return str(info)


def _nilai_target_dari_info(info: dict) -> float | None:
    for kunci in ("target_kdb", "target_klb", "target_kdh"):
        if kunci in info:
            return info[kunci]
    return None


def bangun_langkah_konkret_intensitas(parameter: dict, target_map: dict) -> list[dict]:
    """Daftar SEMUA parameter intensitas yang melanggar (bukan cuma 1 representatif spt
    `pilih_target_utama_intensitas`) — dirakit deterministik, siap jadi
    `RekomendasiOutput.langkah_konkret` (schemas.py). `parameter`/`target_map` di sini SUDAH dict
    biasa (bukan objek Pydantic `ParameterIntensitas`) — bentuk yang sama seperti `poin.fakta`
    (app/adapter.py sudah meng-konversi sebelum masuk PoinKonteks).
    """
    langkah: list[dict] = []
    for nama, info in target_map.items():
        p = parameter.get(nama) or {}
        langkah.append({
            "parameter": nama.upper(),
            "deskripsi": format_target_parameter(info),
            "nilai_saat_ini": p.get("usulan"),
            "nilai_target": _nilai_target_dari_info(info),
            "satuan": p.get("satuan"),
        })
    return langkah


def bangun_langkah_konkret_dampak(target_mitigasi: dict) -> list[dict]:
    """Satu item langkah_konkret utk mitigasi dampak (runoff_change_index) — kosong kalau tak ada
    target (poin aman/kategori tak bersyarat, lihat `hitung_target_mitigasi_dampak` di atas)."""
    if not target_mitigasi or "runoff_change_index_maks" not in target_mitigasi:
        return []
    ambang = target_mitigasi["runoff_change_index_maks"]
    return [{
        "parameter": "runoff_change_index",
        "deskripsi": (
            f"Indikator limpasan air (runoff) perlu ditekan hingga di bawah {ambang} (saat ini "
            f"{target_mitigasi.get('index_saat_ini')}) supaya kategori dampak turun ke "
            f"{target_mitigasi.get('kategori_target')}."
        ),
        "nilai_saat_ini": target_mitigasi.get("index_saat_ini"),
        "nilai_target": ambang,
        "satuan": None,
    }]
