"""Perakit OutputL3 penuh — entrypoint terakhir sebelum JSON dikirim ke web.

Alur: `app.adapter.adaptasi()` (poin[] + rekomendasi_sistem, TIDAK dihitung ulang di sini) ->
`guardrail.generate_poin_dengan_guardrail()` per poin (PARALEL) -> rakit ringkasan_gate/
ringkasan_dampak (deterministik, tanpa LLM) + kesimpulan (SATU panggilan sintesis LLM, HANYA dari
ringkasan per-poin yang sudah lolos guardrail — bukan fakta mentah/angka).

Poin diproses PARALEL (ThreadPoolExecutor) — panggilan Groq itu I/O-bound, GIL dilepas saat
menunggu socket, jadi threading beri speedup nyata tanpa perlu menulis ulang seluruh chain jadi
async (yang akan memaksa ubah SEAM Retriever Protocol, kontrak dengan teman).
"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

from app.adapter import adaptasi, deteksi_fallback_itbx, di_luar_cakupan
from app.logging_util import log_precheck
from app.reasoning import llm_client, observability
from app.reasoning.calculator import normalisasi_kategori_dampak
from app.reasoning.guardrail import DiagnosaPoin, caveat_fallback_itbx, generate_poin_terdiagnosis
from app.reasoning.prompts import SYSTEM_PROMPT_KESIMPULAN, build_kesimpulan_prompt
from app.reasoning.templates import template_low_confidence
from app.retrieval.base import Retriever
from app.schemas import (
    KesimpulanOutput,
    L2Assessment,
    OutputL3,
    PoinKonteks,
    PoinOutput,
    RingkasanDampakOutput,
    RingkasanGateOutput,
)

logger = logging.getLogger(__name__)

_LLM_RESPONSE_SCHEMA_KESIMPULAN = {
    "type": "object",
    "properties": {
        "langkah_berdampak": {"type": "array", "items": {"type": "string"}},
        "catatan_lokasi": {"type": ["string", "null"]},
    },
    "required": ["langkah_berdampak", "catatan_lokasi"],
    "additionalProperties": False,
}
# Sama pola dgn app.reasoning.guardrail._cek_konsistensi_numerik (cek #6) — angka di narasi
# kesimpulan dilarang sama seperti di reasoning per-poin (SYSTEM_PROMPT_KESIMPULAN aturan #3).
_RE_ANGKA_MENCURIGAKAN = re.compile(r"\b\d+[.,]\d+\b|\b\d{2,}\b")

_LABEL_TAHAP = {"itbx": "klasifikasi kegiatan (ITBX)", "intensitas": "intensitas bangunan (KDB/KLB/KDH)"}

_MAX_WORKERS = int(os.getenv("REASONING_MAX_WORKERS", "8"))
_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="reasoning-poin")


def _generate_poin_defensif(
    poin: PoinKonteks, retriever: Retriever, assessment: L2Assessment
) -> tuple[PoinOutput, DiagnosaPoin]:
    """Lapis pertahanan tambahan: guardrail.py seharusnya tidak pernah raise, tapi kalau suatu
    saat ada bug tak terduga, batch tidak boleh gagal total gara-gara 1 poin.

    Ikut membawa keluar `DiagnosaPoin` (sebab poin ini berakhir spt itu) — dipakai `log_precheck`,
    TIDAK ikut ke `OutputL3`. Lihat guardrail.DiagnosaPoin utk kenapa ini perlu ada."""
    try:
        return generate_poin_terdiagnosis(poin, retriever, assessment)
    except Exception as exc:
        logger.exception(
            "generate_poin_terdiagnosis gagal tak terduga utk %s — fallback low_confidence.",
            poin.poin_id,
        )
        diagnosa = DiagnosaPoin(
            poin_id=poin.poin_id, berhasil=False, exception_terakhir=f"{type(exc).__name__}: {exc}"
        )
        return template_low_confidence(poin), diagnosa


_KALIMAT_FALLBACK_ITBX = (
    "klasifikasi kegiatan (ITBX) lolos secara otomatis karena data matriks RDTR belum tersedia — "
    "status ini belum terverifikasi dan perlu ditinjau manual"
)


def _rakit_kalimat_gate(final_gate_status: str, decisive_stage: str | None, itbx_fallback: bool) -> str:
    """Fungsi MURNI atas primitif (bukan L2Assessment) — supaya tiap cabang bisa diuji penuh tanpa
    perlu fabrikasi payload ITBX/Intensitas lengkap.

    Blueprint §5.2: status ITBX yang lolos via fallback (data matriks RDTR kosong/ambigu) BUKAN
    kepatuhan terverifikasi. `final_gate_status` TETAP apa adanya (TIDAK diubah) — yang disesuaikan
    HANYA narasi, supaya tidak overclaim ke arah mana pun (baik overclaim "patuh" utk Lolos, maupun
    overclaim "pelanggaran mutlak" utk Tidak Lolos — dibuktikan perlu oleh fixture nyata
    l2_sample_tidak_lolos.json yang statusnya X dgn reason ambigu).
    """
    tahap = _LABEL_TAHAP.get(decisive_stage, decisive_stage) if decisive_stage else None

    if final_gate_status == "Lolos":
        if itbx_fallback:
            return f"Permohonan lolos pemeriksaan gate hukum, namun {_KALIMAT_FALLBACK_ITBX}."
        return (
            "Permohonan lolos pemeriksaan gate hukum — kegiatan dan intensitas bangunan "
            "memenuhi seluruh ketentuan yang berlaku."
        )

    if final_gate_status == "Lolos Bersyarat":
        dasar = f" pada tahap {tahap}" if tahap else ""
        kalimat = f"Permohonan lolos bersyarat pemeriksaan gate hukum — terdapat catatan{dasar} yang perlu ditindaklanjuti."
        if itbx_fallback:
            kalimat += f" Selain itu, {_KALIMAT_FALLBACK_ITBX}."
        return kalimat

    dasar = f" pada tahap {tahap}" if tahap else ""
    if itbx_fallback:
        # APP-2026-3335: jangan gabung framing "mutlak" (kalimat dasar) dgn "bukan kepastian"
        # (caveat) sekaligus — kontradiktif. Kalau fallback berlaku, "mutlak" DIHAPUS dari kalimat
        # dasar, framing "perlu verifikasi/ditinjau manual" satu-satunya yang dipakai.
        return (
            f"Permohonan tidak lolos pemeriksaan gate hukum — terdapat pelanggaran{dasar}. "
            "Catatan: penentuan ini didasarkan pada data matriks RDTR yang belum lengkap — "
            "bukan kepastian pelanggaran, perlu ditinjau manual."
        )
    return f"Permohonan tidak lolos pemeriksaan gate hukum — terdapat pelanggaran{dasar} yang bersifat mutlak."


def _rakit_ringkasan_gate(assessment: L2Assessment, itbx_fallback: bool) -> RingkasanGateOutput:
    gate = assessment.gate_hukum
    return RingkasanGateOutput(
        final_gate_status=gate.final_gate_status,
        decisive_stage=gate.decisive_stage,
        kalimat=_rakit_kalimat_gate(gate.final_gate_status, gate.decisive_stage, itbx_fallback),
    )


def _rakit_ringkasan_dampak(assessment: L2Assessment) -> RingkasanDampakOutput:
    impact = assessment.impact_assessment
    if not impact.dinilai:
        return RingkasanDampakOutput(
            impact_category=None,
            impact_score=None,
            kalimat="Dampak tata guna lahan belum dinilai untuk permohonan ini.",
        )

    kategori = normalisasi_kategori_dampak(impact.impact_category)
    kalimat = f"Dampak tata guna lahan tergolong {kategori}"
    if impact.impact_score is not None:
        kalimat += " (skor dampak bersifat invers: semakin tinggi skor, semakin rendah dampaknya)."
    else:
        kalimat += "."

    return RingkasanDampakOutput(impact_category=kategori, impact_score=impact.impact_score, kalimat=kalimat)


def _rakit_kesimpulan_fallback(poin_list: list[PoinOutput]) -> KesimpulanOutput:
    """Dipakai kalau panggilan LLM sintesis gagal/melanggar aturan — deterministik, saran
    per-poin verbatim (sudah lolos guardrail, jadi aman ditampilkan apa adanya)."""
    langkah = [p.rekomendasi.saran for p in poin_list if p.rekomendasi.saran.strip()]
    return KesimpulanOutput(langkah_berdampak=langkah, catatan_lokasi=None)


def _rakit_kesimpulan(poin_list: list[PoinOutput], rekomendasi_sistem: str) -> KesimpulanOutput:
    """SATU panggilan LLM (bukan retry loop) — gagal atau melanggar aturan angka -> fallback
    deterministik dari saran per-poin."""
    try:
        prompt = build_kesimpulan_prompt(poin_list, rekomendasi_sistem)
        hasil = llm_client.generate(
            prompt,
            _LLM_RESPONSE_SCHEMA_KESIMPULAN,
            schema_name="kesimpulan",
            system=SYSTEM_PROMPT_KESIMPULAN,
        )
        langkah = hasil.get("langkah_berdampak") or []
        catatan = hasil.get("catatan_lokasi")
        gabungan = " ".join(langkah) + " " + (catatan or "")
        if _RE_ANGKA_MENCURIGAKAN.search(gabungan):
            raise ValueError("Kesimpulan LLM menyebutkan angka — dilarang (SYSTEM_PROMPT_KESIMPULAN aturan #3).")
        return KesimpulanOutput(langkah_berdampak=langkah, catatan_lokasi=catatan)
    except Exception:
        logger.exception("Sintesis kesimpulan via LLM gagal/melanggar aturan — fallback deterministik.")
        return _rakit_kesimpulan_fallback(poin_list)


CAVEAT_DI_LUAR_CAKUPAN = (
    "Koordinat permohonan berada di luar delineasi wilayah RDTR yang menjadi dasar seluruh sitasi "
    "di dokumen ini — dasar hukum yang dikutip kemungkinan tidak berlaku untuk lokasi ini dan wajib "
    "ditinjau manual."
)


def _rakit_catatan_global(
    assessment: L2Assessment,
    itbx_fallback: bool,
    poin_list: list[PoinOutput],
    di_luar_wilayah: bool = False,
) -> list[str]:
    """Blueprint §5.4: meta.caveats WAJIB muncul di output, tidak disembunyikan — plus caveat
    fallback ITBX (§5.2) kalau berlaku. Terpisah dari disclaimer per-poin (guardrail._paksa_field_wajib)
    supaya caveat level-permohonan tetap terlihat walau pemakai cuma baca ringkasan, bukan tiap poin.

    Tambahan: poin bisa jatuh ke low_confidence lewat jalur LAIN di luar fallback ITBX/meta.caveats
    (mis. guardrail kehabisan retry karena reasoning terus melanggar aturan teks — lihat investigasi
    ITBX APP-2026-6191) — kalau begitu, catatan_global tanpa ini akan tetap kosong padahal
    low_confidence_keseluruhan=True. Sebutkan poin mana yang perlu ditinjau manual, apa pun sebabnya.
    """
    catatan = list(assessment.meta.caveats) if assessment.meta and assessment.meta.caveats else []
    if itbx_fallback:
        caveat = caveat_fallback_itbx(assessment.gate_hukum.tahapan.itbx.status)
        catatan.append(caveat.capitalize() + ".")

    if di_luar_wilayah:
        catatan.append(CAVEAT_DI_LUAR_CAKUPAN)

    poin_low_confidence = [p.poin_id for p in poin_list if p.low_confidence]
    if poin_low_confidence:
        daftar = ", ".join(poin_low_confidence)
        catatan.append(
            f"Sebagian penjelasan (poin: {daftar}) tidak dapat dihasilkan otomatis dan perlu "
            "peninjauan manual."
        )

    return catatan


def jalankan_precheck(assessment: L2Assessment, retriever: Retriever) -> OutputL3:
    """Jalankan precheck penuh: generate tiap poin PARALEL (dengan guardrail), rakit output dua-jalur."""
    hasil_adaptasi = adaptasi(assessment)
    itbx_fallback = deteksi_fallback_itbx(assessment.gate_hukum.tahapan.itbx.reason)
    # Dihitung SEKALI di sini (bukan per poin): ini properti permohonan, bukan properti poin.
    # Retrieval dikunci ke satu wilayah lewat RETRIEVER_WILAYAH, jadi kalau lokasinya di luar
    # wilayah itu, SELURUH sitasi di output ini berpotensi tak berlaku — bukan cuma satu poin.
    di_luar_wilayah = di_luar_cakupan(assessment.lokasi)

    futures = [
        _executor.submit(_generate_poin_defensif, poin, retriever, assessment)
        for poin in hasil_adaptasi.poin
    ]
    hasil_poin = [f.result() for f in futures]  # urutan submit == urutan hasil (itbx, intensitas, dampak)
    poin_list = [p for p, _ in hasil_poin]
    diagnostik = [d for _, d in hasil_poin]

    output = OutputL3(
        ringkasan_gate=_rakit_ringkasan_gate(assessment, itbx_fallback),
        ringkasan_dampak=_rakit_ringkasan_dampak(assessment),
        poin=poin_list,
        rekomendasi_sistem=hasil_adaptasi.rekomendasi_sistem,
        kesimpulan=_rakit_kesimpulan(poin_list, hasil_adaptasi.rekomendasi_sistem),
        catatan_global=_rakit_catatan_global(assessment, itbx_fallback, poin_list, di_luar_wilayah),
        # Di luar cakupan = seluruh dasar hukum patut diragukan -> tandai low_confidence walau
        # ketiga poin sendiri lolos guardrail dgn mulus.
        low_confidence_keseluruhan=any(p.low_confidence for p in poin_list) or di_luar_wilayah,
    )

    try:
        log_precheck(assessment, output, diagnostik=[vars(d) | {"sebab": d.sebab()} for d in diagnostik])
    except Exception:
        logger.exception("Gagal menulis log precheck — melanjutkan tanpa menggagalkan respons.")

    observability.catat_precheck_trace(
        "jalankan_precheck", assessment.model_dump(mode="json"), output.model_dump(mode="json")
    )

    return output
