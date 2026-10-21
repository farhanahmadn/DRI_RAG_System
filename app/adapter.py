"""Adapter: L2Assessment (payload back-end mentah) -> poin internal + rekomendasi_sistem.

Murni transformasi data deterministik — tidak ada LLM/RAG di sini (itu tugas generator.py, langkah
berikutnya). Lihat CLAUDE.md § Kontrak Input untuk kontrak gate_hukum/impact_assessment.
"""

import os
import re

from app.reasoning.calculator import (
    hitung_target_intensitas,
    hitung_target_mitigasi_dampak,
    normalisasi_kategori_dampak,
    sarankan_arah_mitigasi_dampak,
)
from app.ketentuan_khusus import tentukan_kawasan_resapan, tentukan_rawan_kekeringan
from app.tipe_bangunan import ambil_tipe_bangunan
from app.reasoning.rekomendasi import gate_efektif, turunkan_rekomendasi
from app.schemas import (
    AdapterResult,
    IntensitasTahap,
    IntensitasZona,
    ItbxZona,
    L2Assessment,
    Lokasi,
    ParameterIntensitas,
    PoinKonteks,
    ZonaInfo,
)

# Kata kunci heuristik deteksi fallback ITBX (Blueprint §5.2: "kolom matriks RDTR kosong/otomatis").
# TIDAK ADA contoh nyata kasus fallback di fixture kita saat ini (keduanya status "I" dgn reason
# normal) — ini tebakan terbaik dari deskripsi Blueprint, gampang direvisi kalau contoh nyata muncul.
_KATA_KUNCI_FALLBACK_ITBX = ("kosong", "otomatis", "tidak ditemukan", "default")


def deteksi_fallback_itbx(reason: str) -> bool:
    reason_lower = reason.lower()
    return any(kata in reason_lower for kata in _KATA_KUNCI_FALLBACK_ITBX)


def _bbox_cakupan() -> tuple[float, float, float, float] | None:
    """Baca `CAKUPAN_BBOX` ("lat_min,lat_max,lon_min,lon_max") dari env. None = guard NONAKTIF.

    Dibaca per-panggilan (bukan konstanta modul) supaya test bisa set/unset env tanpa reload modul,
    pola yang sama dgn os.getenv di app/api/dependencies.py. Env kosong/cacat -> None, BUKAN raise:
    salah ketik konfigurasi tidak boleh menjatuhkan permohonan, dan diam jauh lebih baik daripada
    memberi peringatan palsu ke petugas.
    """
    mentah = os.getenv("CAKUPAN_BBOX", "").strip()
    if not mentah:
        return None
    bagian = mentah.split(",")
    if len(bagian) != 4:
        return None
    try:
        lat_min, lat_max, lon_min, lon_max = (float(b) for b in bagian)
    except ValueError:
        return None
    if lat_min > lat_max or lon_min > lon_max:
        return None
    return lat_min, lat_max, lon_min, lon_max


def di_luar_cakupan(lokasi: Lokasi) -> bool:
    """True kalau koordinat permohonan DIPASTIKAN di luar wilayah RDTR yang dilayani.

    Bukti kenapa perlu (2026-09-07): retrieval dikunci ke satu wilayah lewat `RETRIEVER_WILAYAH`,
    tapi tidak ada satu pun titik di sistem yang memeriksa apakah lokasi permohonan memang ada di
    wilayah itu. Terbukti di `logs/precheck.jsonl`: 3 permohonan (APP-2026-6191/-3335/-3468) di
    koordinat (-7.78329, 110.47835) — di luar batas bujur Sleman Tengah — tetap dijawab & disitasi
    `rdtr-sleman-tengah-*` tanpa peringatan apa pun.

    ARAH PEMERIKSAAN SENGAJA SATU SISI. Bounding box bukan poligon: titik di LUAR bbox pasti di luar
    wilayah (aman disimpulkan), tapi titik di DALAM bbox BELUM TENTU di dalam wilayah (bbox selalu
    lebih besar dari poligon aslinya). Jadi fungsi ini hanya boleh dipakai utk menandai "pasti di
    luar" — `False` TIDAK berarti "terverifikasi berada di dalam delineasi", dan kalimat caveat di
    app/reasoning/assemble.py sengaja tidak mengklaim itu.
    """
    bbox = _bbox_cakupan()
    if bbox is None:
        return False
    lat_min, lat_max, lon_min, lon_max = bbox
    lat = lokasi.koordinat.lat
    lon = lokasi.koordinat.lon
    return not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max)


def _bangun_poin_itbx(assessment: L2Assessment) -> PoinKonteks:
    itbx = assessment.gate_hukum.tahapan.itbx
    return PoinKonteks(
        poin_id="itbx",
        kategori="Klasifikasi Kegiatan (ITBX)",
        tipe_rekomendasi="kategorikal",
        status=itbx.status,
        fakta={
            "lolos": itbx.lolos,
            "kbli_diusulkan": itbx.kbli_diusulkan,
            "kegiatan_diusulkan": itbx.kegiatan_diusulkan,
            "kegiatan_diizinkan": itbx.kegiatan_diizinkan,
            "kegiatan_terbatas": itbx.kegiatan_terbatas,
            "kegiatan_bersyarat": itbx.kegiatan_bersyarat,
            "kegiatan_terbatas_bersyarat": itbx.kegiatan_terbatas_bersyarat,
            "keterangan_ketentuan": itbx.keterangan_ketentuan,
            "reason": itbx.reason,
            "fallback_data_kosong": deteksi_fallback_itbx(itbx.reason),
        },
        dasar_hukum=itbx.dasar_hukum,
        zona=assessment.lokasi.rdtr_zone,
        zona_subzone=assessment.lokasi.rdtr_subzone,
    )


def _zona_pemilik_ambang(assessment: L2Assessment) -> tuple[str | None, str | None]:
    """(nama_zona, kode_zona) zona yang MEMILIKI ambang pada `tahapan.intensitas`, atau
    (None, None) bila tak bisa dipastikan.

    APP-2026-0095: lahan menyilang tiga zona. `lokasi.rdtr_zone` = Zona Pertanian (P-1), tapi
    Pertanian berstatus X sehingga intensitasnya TIDAK_DIEVALUASI, dan ambang pada
    `tahapan.intensitas` (KDB maks 60%) milik Zona Perumahan. Memakai `lokasi.rdtr_zone` membuat
    narasi menulis "KDB melebihi 60% di Zona Pertanian P-1" dan retrieval mengambil Lampiran VI
    P-1 (KDB 10%) yang bertabrakan dgn angka BE.

    Penentu: kode zona di akhir `dasar_hukum[].pasal` ("... (R)") yang cocok dgn tepat satu entri
    `intensitas_zones`; kalau tak ada, entri tunggal yang status & ambangnya sama persis.
    """
    intensitas = assessment.gate_hukum.tahapan.intensitas
    # Permintaan BE 2026-10-15: kalau BE menyebut zona pemilik ambang secara eksplisit, itu yang
    # dipakai (tak perlu menebak dari intensitas_zones).
    if intensitas is not None and intensitas.nama_zona:
        return intensitas.nama_zona, intensitas.kode_zona
    zona_list = assessment.gate_hukum.intensitas_zones or []
    if intensitas is None or len(zona_list) < 2:
        return None, None

    kode_pasal = {
        m.group(1).strip().upper()
        for d in intensitas.dasar_hukum
        if (m := re.search(r"\(([^()]+)\)\s*$", d.pasal or ""))
    }
    cocok = [z for z in zona_list if z.kode_zona and z.kode_zona.strip().upper() in kode_pasal]
    if len(cocok) != 1:
        def _ambang(parameter: dict) -> dict:
            return {
                nama: (p.get("ambang_maks"), p.get("ambang_min")) if isinstance(p, dict)
                else (p.ambang_maks, p.ambang_min)
                for nama, p in parameter.items()
            }
        milik = _ambang(intensitas.parameter)
        cocok = [z for z in zona_list if z.status == intensitas.status and _ambang(z.parameter) == milik]
    if len(cocok) != 1:
        return None, None
    return cocok[0].nama_zona, cocok[0].kode_zona


def _bangun_poin_intensitas(assessment: L2Assessment) -> PoinKonteks:
    intensitas = assessment.gate_hukum.tahapan.intensitas
    if intensitas is None:
        # Gate berhenti di ITBX (mis. status "X") -> intensitas tidak pernah dievaluasi back-end.
        # Pola sama dgn impact_assessment.dinilai=False -> status "Tidak Dinilai" (lihat
        # _bangun_poin_dampak di bawah); generator.py/guardrail.py sudah aman thd fakta minim ini
        # (semua akses field pakai `.get(...)` dgn default, lihat app/reasoning/prompts.py::
        # _bangun_fakta_intensitas & app/reasoning/guardrail.py::_angka_fakta_poin).
        return PoinKonteks(
            poin_id="intensitas",
            kategori="Intensitas Bangunan (KDB/KLB/KDH)",
            tipe_rekomendasi="numerik",
            status="Tidak Dinilai",
            fakta={"dinilai": False},
            dasar_hukum=[],
            zona=assessment.lokasi.rdtr_zone,
        zona_subzone=assessment.lokasi.rdtr_subzone,
        )
    zona, zona_subzone = assessment.lokasi.rdtr_zone, assessment.lokasi.rdtr_subzone
    nama_pemilik, kode_pemilik = _zona_pemilik_ambang(assessment)
    if nama_pemilik and nama_pemilik.strip().lower() != (zona or "").strip().lower():
        # Sub-zona lokasi (mis. P-1) milik zona LAIN, bukan pemilik ambang -> jangan dibawa.
        # Sub-zona pemilik ambang hanya dipakai bila BE mengirim kodenya berbentuk sub-zona ("R-3").
        zona = nama_pemilik
        zona_subzone = kode_pemilik if kode_pemilik and re.match(r"^[A-Za-z]+-\d", kode_pemilik) else None
    return _poin_dari_intensitas(intensitas, assessment, zona, zona_subzone)


def _poin_dari_intensitas(
    intensitas: IntensitasTahap,
    assessment: L2Assessment,
    zona: str | None,
    zona_subzone: str | None,
    zona_info: ZonaInfo | None = None,
) -> PoinKonteks:
    """Poin intensitas dari satu objek `IntensitasTahap` — dipakai jalur satu zona (`tahapan.intensitas`)
    dan jalur multi-zona (satu per entri `intensitas_zones`)."""
    return PoinKonteks(
        poin_id="intensitas",
        kategori="Intensitas Bangunan (KDB/KLB/KDH)",
        tipe_rekomendasi="numerik",
        # `status` dikonsumsi APA ADANYA dari back-end (ground truth) — BUKAN dari `.lolos` (terbukti
        # bisa stale/kontradiktif di data nyata) dan BUKAN diturunkan/divalidasi ulang dari
        # `parameter.*.memenuhi`. `parameter.*.memenuhi` di bawah HANYA fakta pendukung di `fakta[]`
        # untuk generator/prompt nanti — cek silang status/parameter/final_gate_status adalah tugas
        # `cek_konsistensi_intensitas` (dipanggil dari app/reasoning/guardrail.py::_paksa_field_wajib,
        # bukan dari adaptasi() di sini).
        status=intensitas.status,
        fakta={
            "parameter": {
                nama: {
                    "usulan": p.usulan,
                    "ambang_maks": p.ambang_maks,
                    "ambang_min": p.ambang_min,
                    "memenuhi": p.memenuhi,
                    "satuan": p.satuan,
                }
                for nama, p in intensitas.parameter.items()
            },
            "dasar_ambang": {
                k: v for k, v in {
                    "nama_zona": intensitas.nama_zona,
                    "kode_zona": intensitas.kode_zona,
                    "hierarki_jalan": intensitas.hierarki_jalan,
                    "kawasan_resapan": (intensitas.kawasan_resapan if intensitas.kawasan_resapan is not None
                                        else tentukan_kawasan_resapan(assessment)),
                }.items() if v is not None
            },
            "luas_tapak_m2": intensitas.luas_tapak_m2,
            "jumlah_lantai": intensitas.jumlah_lantai,
            "luas_rth_usulan_m2": intensitas.luas_rth_usulan_m2,
            # Angka target PATUH (mis. footprint_maks_m2) — dihitung app/reasoning/calculator.py,
            # dict kosong kalau semua parameter patuh. Tidak mengubah `status` di atas.
            "target": hitung_target_intensitas(intensitas, assessment.lokasi.luas_lahan_m2),
        },
        dasar_hukum=intensitas.dasar_hukum,
        zona=zona,
        zona_subzone=zona_subzone,
        zona_info=zona_info,
    )


# ---------------------------------------------------------------------------
# Lahan multi-zona (permintaan BE 2026-10-20): ITBX dan intensitas dinilai dan dijelaskan PER ZONA,
# hidrologi tetap satu.
# ---------------------------------------------------------------------------

_RE_SUBZONA = re.compile(r"^[A-Za-z]+-\d")


def _cocok_intensitas(zona_itbx: ItbxZona, daftar: list[IntensitasZona]) -> IntensitasZona | None:
    """Entri `intensitas_zones` untuk zona ITBX ini: nama zona sama, atau kode intensitas turunan kode ITBX
    ("P-1" untuk "P")."""
    nama = (zona_itbx.nama_zona or "").strip().lower()
    kode = (zona_itbx.kode_zona or "").strip().upper()
    for z in daftar:
        if nama and (z.nama_zona or "").strip().lower() == nama:
            return z
    for z in daftar:
        k = (z.kode_zona or "").strip().upper()
        if kode and (k == kode or k.startswith(kode + "-")):
            return z
    return None


def zona_terpisah(assessment: L2Assessment) -> list[tuple[ItbxZona, IntensitasZona | None]] | None:
    """Pasangan (zona ITBX, zona intensitas) bila permohonan menyilang >= 2 zona dan BE mengirim
    `itbx_zones`; None untuk lahan satu zona (jalur lama, tidak berubah)."""
    zona_itbx = assessment.gate_hukum.itbx_zones or []
    if len(zona_itbx) < 2:
        return None
    daftar = assessment.gate_hukum.intensitas_zones or []
    return [(z, _cocok_intensitas(z, daftar)) for z in zona_itbx]


def _info_zona(z: ItbxZona, inten: IntensitasZona | None) -> ZonaInfo:
    return ZonaInfo(
        nama_zona=z.nama_zona, kode_zona=z.kode_zona,
        luas_m2=z.luas_m2 if z.luas_m2 is not None else (inten.luas_irisan_m2 if inten else None),
        persentase=z.persentase if z.persentase is not None else (inten.persentase if inten else None),
    )


def _subzona_zona(z: ItbxZona, inten: IntensitasZona | None, assessment: L2Assessment) -> str | None:
    """Kode sub-zona presisi (mis. "P-1") untuk retrieval zona ini, bila diketahui."""
    if inten and inten.kode_zona and _RE_SUBZONA.match(inten.kode_zona):
        return inten.kode_zona
    lokasi_sub = assessment.lokasi.rdtr_subzone or ""
    kode = (z.kode_zona or "").strip().upper()
    if kode and lokasi_sub.upper().startswith(kode + "-"):
        return lokasi_sub
    return None


_RE_LUAS_DI_REASON = re.compile(r"\s+seluas\s+[\d.,]+\s*m\S*", re.IGNORECASE)


def _reason_tanpa_luas(reason: str) -> str:
    """Buang frasa "seluas 623,55 m" dari `reason` zona: luas irisan zona bukan luas kegiatan, dan LLM pernah menulis
    "luas kegiatan 623,55 m² berada dalam batas yang diizinkan" (APP-2026-0022). Luas zona tetap ada di `zona`."""
    return _RE_LUAS_DI_REASON.sub("", reason or "").strip()


def _bangun_poin_itbx_zona(z: ItbxZona, inten: IntensitasZona | None, assessment: L2Assessment) -> PoinKonteks:
    return PoinKonteks(
        poin_id="itbx",
        kategori="Klasifikasi Kegiatan (ITBX)",
        tipe_rekomendasi="kategorikal",
        status=z.status,
        fakta={
            "lolos": z.lolos,
            "kbli_diusulkan": z.kbli_diusulkan,
            "kegiatan_diusulkan": z.kegiatan_diusulkan,
            "kegiatan_diizinkan": z.kegiatan_diizinkan,
            "kegiatan_terbatas": z.kegiatan_terbatas,
            "kegiatan_bersyarat": z.kegiatan_bersyarat,
            "kegiatan_terbatas_bersyarat": z.kegiatan_terbatas_bersyarat,
            "keterangan_ketentuan": z.keterangan_ketentuan,
            "reason": _reason_tanpa_luas(z.reason),
            "fallback_data_kosong": deteksi_fallback_itbx(z.reason),
        },
        dasar_hukum=z.dasar_hukum,
        zona=z.nama_zona,
        zona_subzone=_subzona_zona(z, inten, assessment),
        zona_info=_info_zona(z, inten),
    )


def _bangun_poin_intensitas_zona(z: ItbxZona, inten: IntensitasZona | None, assessment: L2Assessment) -> PoinKonteks:
    info = _info_zona(z, inten)
    subzona = _subzona_zona(z, inten, assessment)
    tidak_dinilai = PoinKonteks(
        poin_id="intensitas", kategori="Intensitas Bangunan (KDB/KLB/KDH)", tipe_rekomendasi="numerik",
        status="Tidak Dinilai", fakta={"dinilai": False}, dasar_hukum=[],
        zona=z.nama_zona, zona_subzone=subzona, zona_info=info,
    )
    if inten is None or inten.status not in ("MEMENUHI_SYARAT", "MELAMPAUI_BATAS") or not inten.parameter:
        # Zona ini tidak dievaluasi intensitasnya (mis. ITBX X, atau BE tidak mengirim entri).
        return tidak_dinilai
    try:
        tahap = IntensitasTahap(
            status=inten.status,
            lolos=inten.status == "MEMENUHI_SYARAT",
            parameter={nama: ParameterIntensitas(**p) for nama, p in inten.parameter.items()},
            luas_tapak_m2=inten.luas_tapak_m2, jumlah_lantai=inten.jumlah_lantai,
            luas_rth_usulan_m2=inten.luas_rth_usulan_m2, reason=inten.reason, dasar_hukum=inten.dasar_hukum,
            nama_zona=inten.nama_zona, kode_zona=inten.kode_zona,
            hierarki_jalan=inten.hierarki_jalan, kawasan_resapan=inten.kawasan_resapan,
        )
    except Exception:  # noqa: BLE001 — parameter zona rusak: jangan menjatuhkan seluruh permohonan
        return tidak_dinilai
    return _poin_dari_intensitas(tahap, assessment, z.nama_zona, subzona, info)


def _bangun_poin_dampak(assessment: L2Assessment) -> PoinKonteks:
    impact = assessment.impact_assessment
    kawasan_resapan = tentukan_kawasan_resapan(assessment)
    kategori_dampak = normalisasi_kategori_dampak(impact.impact_category)
    return PoinKonteks(
        poin_id="dampak",
        # Item permintaan user 2026-09-21: "Dampak Tata Guna Lahan" (lama) -> label ini lebih
        # jelas mencerminkan substansi indikator (dampak HIDROLOGI/limpasan air akibat perubahan
        # tata guna lahan, bukan tata guna lahan itu sendiri). HANYA label tampilan di output —
        # TIDAK menyentuh `_QUERY_FALLBACK_PER_POIN["dampak"]`/`_EXPANSION` (generator.py/
        # retriever.py, string internal "dampak tata guna lahan" lowercase TERPISAH, dipakai
        # query RAG, sengaja TIDAK diubah supaya retrieval tak ikut berubah perilaku).
        kategori="Dampak Terhadap Lingkungan (Hidrologi)",
        tipe_rekomendasi="numerik-mitigasi",
        status=kategori_dampak if impact.dinilai else "Tidak Dinilai",
        fakta={
            "dinilai": impact.dinilai,
            "impact_score": impact.impact_score,
            "runoff_change_index": impact.runoff_change_index,
            "c_before": impact.c_before,
            "c_after": impact.c_after,
            "threshold_bands": impact.threshold_bands,
            "existing_surface_details": impact.existing_surface_details,
            "proposed_surface_details": impact.proposed_surface_details,
            "c_coefficients": impact.c_coefficients,
            "data_confidence": impact.data_confidence,
            "limitations": impact.limitations,
            "luas_usulan_melebihi_persil": impact.luas_usulan_melebihi_persil,
            "kawasan_resapan": kawasan_resapan,
            # Tipe bangunan/kegiatan yang diusulkan (permintaan rapat 2026-10-16): menentukan kelas
            # bangunan & koefisien limpasan atap pada perhitungan BE.
            "tipe_bangunan": ambil_tipe_bangunan(assessment),
            "kawasan_hidrologi": {
                k: v for k, v in {
                    "nama": impact.kawasan_hidrologi, "kode": impact.kode_kawasan,
                    "status_itbx": impact.status_kawasan,
                }.items() if v is not None
            },
            # Arah mitigasi kualitatif — app/reasoning/calculator.py, TIDAK menghitung ulang C/index.
            "mitigasi": sarankan_arah_mitigasi_dampak(impact, kawasan_resapan),
            # Target kuantitatif (ambang runoff_change_index utk turun 1 kategori) — murni aritmatika
            # threshold_bands vs index yang SUDAH diberi back-end, bukan hitung ulang rumus C.
            "target_mitigasi": hitung_target_mitigasi_dampak(impact, kawasan_resapan),
        },
        dasar_hukum=[],
        zona=assessment.lokasi.rdtr_zone,
        zona_subzone=assessment.lokasi.rdtr_subzone,
        kawasan_resapan=kawasan_resapan,
        rawan_kekeringan=tentukan_rawan_kekeringan(assessment),
    )


_URUTAN_KETAT_ITBX = {"X": 4, "TB": 3, "T": 3, "B": 2, "I": 1}


def _itbx_paling_ketat(assessment: L2Assessment) -> str | None:
    """Status ITBX paling ketat di seluruh zona (X > T/TB > B > I); satu zona = status `tahapan.itbx`."""
    zona = assessment.gate_hukum.itbx_zones or []
    daftar = [z.status for z in zona] if len(zona) >= 2 else [assessment.gate_hukum.tahapan.itbx.status]
    return max(daftar, key=lambda s: _URUTAN_KETAT_ITBX.get(s, 0)) if daftar else None


def status_gate_efektif(assessment: L2Assessment) -> str:
    """Status gate menurut spesifikasi BE bagian 4: ITBX X -> Tidak Lolos dan ITBX T/TB -> Lolos Terbatas
    mengalahkan label gate BE (BE kerap melabeli ITBX T sebagai Lolos Bersyarat atau Lolos)."""
    return gate_efektif(assessment.gate_hukum.final_gate_status, _itbx_paling_ketat(assessment))


def adaptasi(assessment: L2Assessment) -> AdapterResult:
    # final_gate_status dikonsumsi APA ADANYA (ground truth back-end) — tidak diturunkan ulang dari
    # status/parameter tahap-tahap di bawahnya.
    kategori_dampak = normalisasi_kategori_dampak(assessment.impact_assessment.impact_category)
    rekomendasi_sistem = turunkan_rekomendasi(
        assessment.gate_hukum.final_gate_status, kategori_dampak, _itbx_paling_ketat(assessment))
    pasangan = zona_terpisah(assessment)
    if pasangan:
        poin: list[PoinKonteks] = []
        for z, inten in pasangan:
            poin.append(_bangun_poin_itbx_zona(z, inten, assessment))
            poin.append(_bangun_poin_intensitas_zona(z, inten, assessment))
        poin.append(_bangun_poin_dampak(assessment))  # hidrologi tetap satu (digabung)
        return AdapterResult(poin=poin, rekomendasi_sistem=rekomendasi_sistem)
    return AdapterResult(
        poin=[
            _bangun_poin_itbx(assessment),
            _bangun_poin_intensitas(assessment),
            _bangun_poin_dampak(assessment),
        ],
        rekomendasi_sistem=rekomendasi_sistem,
    )


def cek_konsistensi_intensitas(assessment: L2Assessment) -> list[str]:
    """Deteksi ketidakkonsistenan status vs parameter.memenuhi vs final_gate_status.

    Dipanggil dari app/reasoning/guardrail.py::_paksa_field_wajib (Cek #4) untuk poin intensitas —
    BUKAN dari `adaptasi()` di sini, karena hasilnya cuma dipakai guardrail untuk menandai
    low_confidence, tidak untuk membentuk PoinKonteks.

    Kalau daftar hasil non-kosong, guardrail HARUS menandai low_confidence + log rinciannya — JANGAN
    menimpa/mengoreksi field back-end di sini atau di mana pun (prinsip Faithful, CLAUDE.md).
    """
    masalah: list[str] = []
    intensitas = assessment.gate_hukum.tahapan.intensitas
    if intensitas is None:
        # Tidak dievaluasi back-end (gate berhenti di ITBX) -> tidak ada apa pun utk dicek silang.
        return masalah
    ada_pelanggaran_param = any(not p.memenuhi for p in intensitas.parameter.values())

    if intensitas.status == "MELAMPAUI_BATAS" and not ada_pelanggaran_param:
        masalah.append("status=MELAMPAUI_BATAS tapi semua parameter.memenuhi=True")
    if intensitas.status == "MEMENUHI_SYARAT" and ada_pelanggaran_param:
        masalah.append("status=MEMENUHI_SYARAT tapi ada parameter.memenuhi=False")
    if intensitas.status == "MELAMPAUI_BATAS" and assessment.gate_hukum.final_gate_status == "Lolos":
        masalah.append("intensitas MELAMPAUI_BATAS tapi final_gate_status=Lolos (harusnya Lolos Bersyarat)")

    return masalah


def cek_konsistensi_poin_intensitas(poin: PoinKonteks) -> list[str]:
    """Cek silang status vs parameter.memenuhi untuk poin intensitas PER ZONA (multi-zona). Cek terhadap
    `final_gate_status` sengaja tidak dilakukan di sini: gate adalah keputusan seluruh permohonan, bukan satu zona."""
    parameter = (poin.fakta or {}).get("parameter") or {}
    if poin.status not in ("MEMENUHI_SYARAT", "MELAMPAUI_BATAS") or not parameter:
        return []
    ada_pelanggaran = any(not v.get("memenuhi", True) for v in parameter.values())
    if poin.status == "MELAMPAUI_BATAS" and not ada_pelanggaran:
        return ["status=MELAMPAUI_BATAS tapi semua parameter.memenuhi=True"]
    if poin.status == "MEMENUHI_SYARAT" and ada_pelanggaran:
        return ["status=MEMENUHI_SYARAT tapi ada parameter.memenuhi=False"]
    return []
