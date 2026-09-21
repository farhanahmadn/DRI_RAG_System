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


def _bulat2(nilai: float | int | None) -> float | int | None:
    """Bulatkan angka TAMPILAN (bukan fakta mentah) ke maks 2 desimal — presisi panjang dari BE
    (mis. usulan 96.09509778650137) tak berarti apa pun bagi reviewer & bikin kalimat/langkah_konkret
    terlihat 'kotor'. Dipakai HANYA di titik rakit dict target/langkah_konkret (nilai yang dipakai
    utk DITAMPILKAN, bukan dibandingkan lagi) — guardrail._dekat_dgn_pembulatan sudah toleran
    membandingkan versi bulat ini dgn fakta presisi penuh, jadi tidak mengganggu cek provenance.
    int/None/bool diteruskan apa adanya (bulat tak berarti apa pun utknya)."""
    if nilai is None or isinstance(nilai, bool) or isinstance(nilai, int):
        return nilai
    return round(nilai, 2)


def _fmt_angka(nilai: float | int | None) -> str:
    """Tampilkan angka TANPA nol berlebihan di belakang koma — "60.0" -> "60", "51.98" tetap
    "51.98" (bukan "51.98000..."). Reviewer non-teknis: presisi >2 desimal tak berarti apa pun,
    tapi "60.00%" juga terlihat janggal utk bilangan bulat — dipakai di titik render kalimat
    (nilai di sini SUDAH dibulatkan `_bulat2` sebelumnya, ini murni kosmetik tampilan)."""
    if nilai is None:
        return "-"
    if isinstance(nilai, bool):
        return str(nilai)
    if isinstance(nilai, int) or float(nilai).is_integer():
        return str(int(nilai))
    return f"{nilai:.2f}".rstrip("0").rstrip(".")


def hitung_target_kdb(param: ParameterIntensitas, luas_lahan_m2: float | None) -> dict[str, float] | None:
    if param.ambang_maks is None or param.usulan <= param.ambang_maks:
        return None
    selisih = param.usulan - param.ambang_maks
    hasil = {"target_kdb": _bulat2(param.ambang_maks), "selisih": _bulat2(selisih)}
    if luas_lahan_m2 is not None:
        hasil["footprint_maks_m2"] = _bulat2(luas_lahan_m2 * _ke_fraksi(param.ambang_maks, param.satuan))
    return hasil


def hitung_target_klb(param: ParameterIntensitas, luas_lahan_m2: float | None) -> dict[str, float] | None:
    if param.ambang_maks is None or param.usulan <= param.ambang_maks:
        return None
    selisih = param.usulan - param.ambang_maks
    hasil = {"target_klb": _bulat2(param.ambang_maks), "selisih": _bulat2(selisih)}
    if luas_lahan_m2 is not None:
        hasil["luas_lantai_maks_m2"] = _bulat2(luas_lahan_m2 * _ke_fraksi(param.ambang_maks, param.satuan))
    return hasil


def hitung_target_kdh(
    param: ParameterIntensitas, luas_lahan_m2: float | None, luas_rth_usulan_m2: float | None
) -> dict[str, float] | None:
    if param.ambang_min is None or param.usulan >= param.ambang_min:
        return None
    selisih = param.ambang_min - param.usulan  # positif = kurang
    hasil = {"target_kdh": _bulat2(param.ambang_min), "selisih": _bulat2(selisih)}
    if luas_lahan_m2 is not None:
        rth_dibutuhkan = luas_lahan_m2 * _ke_fraksi(param.ambang_min, param.satuan)
        hasil["rth_dibutuhkan_m2"] = _bulat2(rth_dibutuhkan)
        if luas_rth_usulan_m2 is not None:
            hasil["rth_kurang_m2"] = _bulat2(max(rth_dibutuhkan - luas_rth_usulan_m2, 0.0))
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

    APP-2026-8025/-5067: BE kini (kadang) kirim `impact.rekomendasi_mitigasi` — target indeks +
    rincian penyesuaian lahan (m²/KDB%/KDH% KONKRET) + dimensi minimum sumur/kolam resapan + narasi
    `saran` SUDAH DIHITUNG PENUH di sisi BE. Diprioritaskan kalau ada (echo APA ADANYA, bukan
    dihitung ulang — prinsip sama dgn limitations/luas_usulan_melebihi_persil): jauh lebih presisi &
    actionable drpd rekonstruksi kita dari threshold_bands (band cuma kasih ambang KATEGORI, tak
    pernah kasih angka fisik lahan). Fallback ke logika band lama di bawah kalau field ini absen
    (fixture lama / BE belum kirim untuk kasus ini) — backward-compatible.
    """
    rekom = impact.rekomendasi_mitigasi
    if rekom and rekom.get("target_indeks_maks") is not None:
        hasil: dict = {
            "runoff_change_index_maks": rekom["target_indeks_maks"],
            "kategori_target": normalisasi_kategori_dampak(rekom.get("target_kategori")) or "",
            "index_saat_ini": impact.runoff_change_index,
        }
        penyesuaian = rekom.get("rekomendasi_penyesuaian_lahan")
        if penyesuaian:
            hasil["penyesuaian_lahan"] = penyesuaian
            # "nilai_saat_ini" (luas bangunan/RTH TERUSULKAN, sebelum penyesuaian) — dari
            # `detailed_surface_breakdown` (echo apa adanya, bukan dihitung ulang) kalau BE
            # menyertakannya; None kalau tak ada (langkah_konkret tetap tampil, cuma tanpa
            # perbandingan "saat ini vs target").
            breakdown = impact.detailed_surface_breakdown or {}
            bangunan = breakdown.get("Bangunan/Atap")
            if isinstance(bangunan, dict) and bangunan.get("luas_m2") is not None:
                hasil["luas_bangunan_saat_ini_m2"] = bangunan["luas_m2"]
            rth = breakdown.get("Taman/RTH")
            if isinstance(rth, dict) and rth.get("luas_m2") is not None:
                hasil["luas_rth_saat_ini_m2"] = rth["luas_m2"]
        dimensi = rekom.get("dimensi_minimum")
        if dimensi:
            hasil["dimensi_minimum_resapan"] = dimensi
        saran_be = rekom.get("saran")
        if saran_be:
            hasil["saran_be"] = saran_be
        return hasil

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
    — pola sama seperti `pilih_target_utama_intensitas`. Dibulatkan (`_bulat2`) — field ini tampil
    langsung di JSON keluaran (`rekomendasi.target`)."""
    return _bulat2(target_mitigasi.get("runoff_change_index_maks"))


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
        kalimat = f"KDB harus turun ke maksimal {_fmt_angka(info['target_kdb'])}% (selisih {_fmt_angka(info['selisih'])} poin dari usulan)"
        if "footprint_maks_m2" in info:
            kalimat += f" → luas lantai dasar bangunan maksimal {_fmt_angka(info['footprint_maks_m2'])} m²"
        return kalimat
    if "target_klb" in info:
        kalimat = f"KLB harus turun ke maksimal {_fmt_angka(info['target_klb'])} (selisih {_fmt_angka(info['selisih'])} dari usulan)"
        if "luas_lantai_maks_m2" in info:
            kalimat += f" → luas total lantai bangunan maksimal {_fmt_angka(info['luas_lantai_maks_m2'])} m²"
        return kalimat
    if "target_kdh" in info:
        kalimat = f"KDH harus naik ke minimal {_fmt_angka(info['target_kdh'])}% (kurang {_fmt_angka(info['selisih'])} poin dari usulan)"
        if "rth_dibutuhkan_m2" in info:
            kalimat += f" → RTH dibutuhkan minimal {_fmt_angka(info['rth_dibutuhkan_m2'])} m²"
        if info.get("rth_kurang_m2"):
            kalimat += f" (RTH yang sudah diusulkan pemohon masih kurang {_fmt_angka(info['rth_kurang_m2'])} m²)"
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
            "nilai_saat_ini": _bulat2(p.get("usulan")),
            "nilai_target": _nilai_target_dari_info(info),
            "satuan": p.get("satuan"),
        })
    return langkah


def bangun_langkah_konkret_dampak(target_mitigasi: dict) -> list[dict]:
    """langkah_konkret utk mitigasi dampak — kosong kalau tak ada target (poin aman/kategori tak
    bersyarat, lihat `hitung_target_mitigasi_dampak` di atas).

    APP-2026-8025/-5067: kalau `target_mitigasi` bawa rincian dari BE (`penyesuaian_lahan`/
    `dimensi_minimum_resapan`, lihat `hitung_target_mitigasi_dampak`) — pecah jadi HINGGA 3 item
    KONKRET (luas bangunan m², luas RTH m², dimensi sumur/kolam resapan) alih-alih 1 item abstrak
    "runoff_change_index" — jauh lebih actionable bagi reviewer/pemohon (bisa langsung dibandingkan
    dgn gambar kerja, bukan angka indeks tanpa satuan fisik). Fallback ke item generik lama kalau
    BE tak kirim rincian ini (fixture lama/provider lama) — backward-compatible.
    """
    if not target_mitigasi or "runoff_change_index_maks" not in target_mitigasi:
        return []

    penyesuaian = target_mitigasi.get("penyesuaian_lahan")
    dimensi = target_mitigasi.get("dimensi_minimum_resapan")
    if penyesuaian or dimensi:
        langkah_rinci: list[dict] = []
        penyesuaian = penyesuaian or {}
        kategori_target = target_mitigasi.get("kategori_target") or ""
        luas_bangunan_maks = penyesuaian.get("luas_bangunan_maks_m2")
        if luas_bangunan_maks is not None:
            kdb_maks = penyesuaian.get("kdb_maks_persen")
            langkah_rinci.append({
                "parameter": "Luas Bangunan (Atap)",
                "deskripsi": (
                    f"Turunkan luas lantai dasar bangunan ke maksimal {_fmt_angka(luas_bangunan_maks)} m²"
                    + (f" (KDB maks {_fmt_angka(kdb_maks)}%)" if kdb_maks is not None else "")
                    + f" agar indeks limpasan turun ke kategori {kategori_target}."
                ),
                "nilai_saat_ini": _bulat2(target_mitigasi.get("luas_bangunan_saat_ini_m2")),
                "nilai_target": _bulat2(luas_bangunan_maks),
                "satuan": "m²",
            })
        luas_rth_min = penyesuaian.get("luas_rth_min_m2")
        if luas_rth_min is not None:
            kdh_min = penyesuaian.get("kdh_min_persen")
            langkah_rinci.append({
                "parameter": "Luas RTH",
                "deskripsi": (
                    f"Tingkatkan luas RTH ke minimal {_fmt_angka(luas_rth_min)} m²"
                    + (f" (KDH min {_fmt_angka(kdh_min)}%)" if kdh_min is not None else "")
                    + " untuk memperluas daerah peresapan air alami."
                ),
                "nilai_saat_ini": _bulat2(target_mitigasi.get("luas_rth_saat_ini_m2")),
                "nilai_target": _bulat2(luas_rth_min),
                "satuan": "m²",
            })
        if dimensi and dimensi.get("nilai") is not None:
            satuan_dimensi = dimensi.get("satuan") or ""
            langkah_rinci.append({
                "parameter": "Dimensi Minimum Sumur Resapan",
                "deskripsi": (
                    f"Alternatif: sediakan sumur atau kolam resapan dengan dimensi minimum "
                    f"{_fmt_angka(dimensi['nilai'])} {satuan_dimensi} berdasarkan rumus mitigasi hidrologi."
                ),
                "nilai_saat_ini": 0,
                "nilai_target": _bulat2(dimensi["nilai"]),
                "satuan": satuan_dimensi or None,
            })
        if langkah_rinci:
            return langkah_rinci
        # penyesuaian/dimensi ADA tapi tak satu pun angkanya valid -> jatuh ke item generik di bawah.

    # "Indeks Limpasan (Runoff)" — BUKAN nama variabel kode "runoff_change_index" apa adanya
    # (item #5 permintaan user, 2026-09-21): field ini tampil langsung di JSON keluaran sistem
    # (`RekomendasiOutput.langkah_konkret[].parameter`), nama variabel snake_case membingungkan
    # reviewer non-teknis yang membaca hasilnya.
    ambang = _bulat2(target_mitigasi["runoff_change_index_maks"])
    return [{
        "parameter": "Indeks Limpasan (Runoff)",
        "deskripsi": (
            f"Indikator limpasan air (runoff) perlu ditekan hingga di bawah {_fmt_angka(ambang)} (saat ini "
            f"{_fmt_angka(target_mitigasi.get('index_saat_ini'))}) supaya kategori dampak turun ke "
            f"{target_mitigasi.get('kategori_target')}."
        ),
        "nilai_saat_ini": _bulat2(target_mitigasi.get("index_saat_ini")),
        "nilai_target": ambang,
        "satuan": None,
    }]
