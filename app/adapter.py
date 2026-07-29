"""Adapter: L2Assessment (payload back-end mentah) -> poin internal + rekomendasi_sistem.

Murni transformasi data deterministik — tidak ada LLM/RAG di sini (itu tugas generator.py, langkah
berikutnya). Lihat CLAUDE.md § Kontrak Input untuk kontrak gate_hukum/impact_assessment.
"""

from app.reasoning.rekomendasi import turunkan_rekomendasi
from app.schemas import AdapterResult, L2Assessment, PoinKonteks


def _normalisasi_kategori_dampak(raw: str | None) -> str | None:
    """"SEDANG" -> "Sedang", "SANGAT TINGGI" -> "Sangat Tinggi", None -> None."""
    return raw.strip().title() if raw else None


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
        },
        dasar_hukum=itbx.dasar_hukum,
    )


def _bangun_poin_intensitas(assessment: L2Assessment) -> PoinKonteks:
    intensitas = assessment.gate_hukum.tahapan.intensitas
    return PoinKonteks(
        poin_id="intensitas",
        kategori="Intensitas Bangunan (KDB/KLB/KDH)",
        tipe_rekomendasi="numerik",
        # Status diturunkan dari `status` (MEMENUHI_SYARAT/MELAMPAUI_BATAS), BUKAN dari `.lolos` —
        # terbukti bisa stale/kontradiktif di data back-end nyata.
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
        },
        dasar_hukum=intensitas.dasar_hukum,
    )


def _bangun_poin_dampak(assessment: L2Assessment) -> PoinKonteks:
    impact = assessment.impact_assessment
    kategori_dampak = _normalisasi_kategori_dampak(impact.impact_category)
    return PoinKonteks(
        poin_id="dampak",
        kategori="Dampak Tata Guna Lahan",
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
        },
        dasar_hukum=[],
    )


def adaptasi(assessment: L2Assessment) -> AdapterResult:
    kategori_dampak = _normalisasi_kategori_dampak(assessment.impact_assessment.impact_category)
    rekomendasi_sistem = turunkan_rekomendasi(assessment.gate_hukum.final_gate_status, kategori_dampak)
    return AdapterResult(
        poin=[
            _bangun_poin_itbx(assessment),
            _bangun_poin_intensitas(assessment),
            _bangun_poin_dampak(assessment),
        ],
        rekomendasi_sistem=rekomendasi_sistem,
    )
