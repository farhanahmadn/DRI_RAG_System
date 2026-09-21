"""Adapter: L2Assessment (payload back-end mentah) -> poin internal + rekomendasi_sistem.

Murni transformasi data deterministik — tidak ada LLM/RAG di sini (itu tugas generator.py, langkah
berikutnya). Lihat CLAUDE.md § Kontrak Input untuk kontrak gate_hukum/impact_assessment.
"""

import os

from app.reasoning.calculator import (
    hitung_target_intensitas,
    hitung_target_mitigasi_dampak,
    normalisasi_kategori_dampak,
    sarankan_arah_mitigasi_dampak,
)
from app.reasoning.rekomendasi import turunkan_rekomendasi
from app.schemas import AdapterResult, L2Assessment, Lokasi, PoinKonteks

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
            "luas_tapak_m2": intensitas.luas_tapak_m2,
            "jumlah_lantai": intensitas.jumlah_lantai,
            "luas_rth_usulan_m2": intensitas.luas_rth_usulan_m2,
            # Angka target PATUH (mis. footprint_maks_m2) — dihitung app/reasoning/calculator.py,
            # dict kosong kalau semua parameter patuh. Tidak mengubah `status` di atas.
            "target": hitung_target_intensitas(intensitas, assessment.lokasi.luas_lahan_m2),
        },
        dasar_hukum=intensitas.dasar_hukum,
        zona=assessment.lokasi.rdtr_zone,
        zona_subzone=assessment.lokasi.rdtr_subzone,
    )


def _bangun_poin_dampak(assessment: L2Assessment) -> PoinKonteks:
    impact = assessment.impact_assessment
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
            # Arah mitigasi kualitatif — app/reasoning/calculator.py, TIDAK menghitung ulang C/index.
            "mitigasi": sarankan_arah_mitigasi_dampak(impact),
            # Target kuantitatif (ambang runoff_change_index utk turun 1 kategori) — murni aritmatika
            # threshold_bands vs index yang SUDAH diberi back-end, bukan hitung ulang rumus C.
            "target_mitigasi": hitung_target_mitigasi_dampak(impact),
        },
        dasar_hukum=[],
        zona=assessment.lokasi.rdtr_zone,
        zona_subzone=assessment.lokasi.rdtr_subzone,
    )


def adaptasi(assessment: L2Assessment) -> AdapterResult:
    # final_gate_status dikonsumsi APA ADANYA (ground truth back-end) — tidak diturunkan ulang dari
    # status/parameter tahap-tahap di bawahnya.
    kategori_dampak = normalisasi_kategori_dampak(assessment.impact_assessment.impact_category)
    rekomendasi_sistem = turunkan_rekomendasi(assessment.gate_hukum.final_gate_status, kategori_dampak)
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
