"""Perakit OutputL3 penuh — entrypoint terakhir sebelum JSON dikirim ke web.

Alur: `app.adapter.adaptasi()` (poin[] + rekomendasi_sistem, TIDAK dihitung ulang di sini) ->
`guardrail.generate_poin_dengan_guardrail()` per poin (PARALEL) -> rakit ringkasan_gate/
ringkasan_dampak (deterministik, tanpa LLM) + kesimpulan (SATU panggilan sintesis LLM, HANYA dari
ringkasan per-poin yang sudah lolos guardrail — bukan fakta mentah/angka).

Poin diproses PARALEL (ThreadPoolExecutor) — panggilan Groq itu I/O-bound, GIL dilepas saat
menunggu socket, jadi threading beri speedup nyata tanpa perlu menulis ulang seluruh chain jadi
async (yang akan memaksa ubah SEAM Retriever Protocol, kontrak dengan teman).
"""

import contextvars
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

from app.adapter import adaptasi, deteksi_fallback_itbx, di_luar_cakupan
from app.logging_util import log_precheck
from app import batas_waktu
from app.tipe_bangunan import ambil_tipe_bangunan, frasa_tipe_bangunan
from app.reasoning import llm_client, observability
from app.reasoning.calculator import _fmt_id, normalisasi_kategori_dampak
from app.reasoning.guardrail import (
    DiagnosaPoin,
    _dekat_dgn_pembulatan,
    _perbaiki_titik_koma,
    caveat_fallback_itbx,
    generate_poin_terdiagnosis,
)
from app.reasoning.prompts import (
    CAVEAT_SUBZONA_TAK_TERKONFIRMASI,
    SYSTEM_PROMPT_KESIMPULAN,
    SYSTEM_PROMPT_NARASI_REKOMENDASI,
    build_kesimpulan_prompt,
    build_narasi_rekomendasi_prompt,
)
from app.reasoning.templates import template_low_confidence
from app.retrieval.base import Retriever
from app.schemas import (
    KesimpulanOutput,
    L2Assessment,
    NarasiRekomendasiOutput,
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

_LLM_RESPONSE_SCHEMA_NARASI_REKOMENDASI = {
    "type": "object",
    "properties": {
        "paragraf_gate_intensitas": {"type": "string"},
        "paragraf_dampak": {"type": "string"},
    },
    "required": ["paragraf_gate_intensitas", "paragraf_dampak"],
    "additionalProperties": False,
}

_LABEL_TAHAP = {"itbx": "klasifikasi kegiatan (ITBX)", "intensitas": "intensitas bangunan (KDB/KLB/KDH)"}

_MAX_WORKERS = int(os.getenv("REASONING_MAX_WORKERS", "8"))
_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="reasoning-poin")


# Tambahan waktu tunggu (detik) atas sisa anggaran saat menunggu thread poin: thread yang taat anggaran
# berhenti sendiri, tunggu sebentar agar fallback mereka (yang lebih informatif) sempat terkumpul.
_TOLERANSI_TUNGGU_S = 4.0
# Sisa anggaran minimum (detik) agar satu panggilan sintesis LLM masih layak.
_MIN_SISA_SINTESIS_S = 10.0


def _poin_batas_waktu(poin: PoinKonteks) -> tuple[PoinOutput, DiagnosaPoin]:
    diagnosa = DiagnosaPoin(
        poin_id=poin.poin_id, berhasil=False,
        exception_terakhir="BatasWaktuTerlampaui: poin belum selesai saat anggaran waktu permintaan habis",
    )
    return template_low_confidence(poin), diagnosa


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


def _serialisasi_diagnostik(diagnostik: list[DiagnosaPoin]) -> list[dict]:
    """`DiagnosaPoin` -> dict siap-JSON, satu per poin, dgn label `sebab` ikut dihitung.

    Dipisah dari pemanggilan `log_precheck` supaya kegagalan pada SATU poin tidak menjatuhkan
    seluruh baris log. Sebelumnya perakitan ini jadi ekspresi argumen di dalam `try` pemanggil:
    satu `d.sebab()` yang raise (mis. `exception_terakhir` bertipe aneh) membuang request +
    response sekalian — padahal keduanya justru bukti yang paling sulit dikumpulkan ulang.
    Diagnostik adalah data operasional; ia boleh degradasi, log utamanya tidak.
    """
    keluar: list[dict] = []
    for d in diagnostik:
        try:
            keluar.append(vars(d) | {"sebab": d.sebab()})
        except Exception:
            logger.exception("Gagal menyerialkan diagnostik poin %r — dicatat sbg rusak.",
                             getattr(d, "poin_id", "?"))
            keluar.append({"poin_id": getattr(d, "poin_id", None), "sebab": "diagnostik_rusak"})
    return keluar


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
            kalimat="Dampak terhadap lingkungan (hidrologi) belum dinilai untuk permohonan ini.",
        )

    kategori = normalisasi_kategori_dampak(impact.impact_category)
    kalimat = (f"Dampak terhadap lingkungan (hidrologi) tergolong {kategori}"
               f"{frasa_tipe_bangunan(ambil_tipe_bangunan(assessment))}")
    if impact.impact_score is not None:
        kalimat += " (skor dampak bersifat invers: semakin tinggi skor, semakin rendah dampaknya)."
    else:
        kalimat += "."

    return RingkasanDampakOutput(impact_category=kategori, impact_score=impact.impact_score, kalimat=kalimat)


_TOLERANSI_M2 = 0.5


def _catatan_batas_berbeda(poin_konteks: list) -> str | None:
    """Catatan deterministik bila batas luas dari ketentuan INTENSITAS dan dari syarat penurunan
    kategori DAMPAK berbeda (APP-2026-UJI-4: lantai dasar maks 861,6 m² vs 1.077,00 m²). Keduanya
    angka yang benar menurut aturannya masing-masing, tapi pemohon yang membaca keduanya akan
    bingung mana yang harus diikuti. Yang diikuti = yang lebih ketat, supaya memenuhi keduanya."""
    by_id = {p.poin_id: p for p in poin_konteks}
    inten, damp = by_id.get("intensitas"), by_id.get("dampak")
    if not inten or not damp or "Tidak Dinilai" in (inten.status, damp.status):
        return None
    target = inten.fakta.get("target") or {}
    penyesuaian = (damp.fakta.get("target_mitigasi") or {}).get("penyesuaian_lahan") or {}
    b_int = (target.get("kdb") or {}).get("footprint_maks_m2")
    b_dam = penyesuaian.get("luas_bangunan_maks_m2")
    r_int = (target.get("kdh") or {}).get("rth_dibutuhkan_m2")
    r_dam = penyesuaian.get("luas_rth_min_m2")

    kalimat: list[str] = []
    if b_int is not None and b_dam is not None and abs(b_int - b_dam) > _TOLERANSI_M2:
        if b_dam > b_int:
            kalimat.append(
                f"Batas luas lantai dasar menurut ketentuan intensitas ruang ({_fmt_id(b_int)} m²) lebih ketat "
                f"daripada syarat penurunan kategori dampak ({_fmt_id(b_dam)} m²), sehingga yang harus diikuti "
                f"adalah {_fmt_id(b_int)} m².")
        else:
            kalimat.append(
                f"Syarat penurunan kategori dampak membatasi luas lantai dasar lebih ketat ({_fmt_id(b_dam)} m²) "
                f"daripada ketentuan intensitas ruang ({_fmt_id(b_int)} m²), sehingga untuk memenuhi keduanya "
                f"yang diikuti adalah {_fmt_id(b_dam)} m².")
    if r_int is not None and r_dam is not None and abs(r_int - r_dam) > _TOLERANSI_M2:
        if r_dam > r_int:
            kalimat.append(
                f"Syarat penurunan kategori dampak menuntut RTH lebih luas ({_fmt_id(r_dam)} m²) daripada "
                f"ketentuan intensitas ruang ({_fmt_id(r_int)} m²), sehingga untuk memenuhi keduanya yang diikuti "
                f"adalah {_fmt_id(r_dam)} m².")
        else:
            kalimat.append(
                f"Ketentuan intensitas ruang menuntut RTH lebih luas ({_fmt_id(r_int)} m²) daripada syarat "
                f"penurunan kategori dampak ({_fmt_id(r_dam)} m²), sehingga yang diikuti adalah {_fmt_id(r_int)} m².")
    return " ".join(kalimat) or None


def _terapkan_catatan_batas(poin_list: list[PoinOutput], poin_konteks: list) -> list[PoinOutput]:
    catatan = _catatan_batas_berbeda(poin_konteks)
    if not catatan:
        return poin_list
    hasil = []
    for p in poin_list:
        if p.poin_id == "dampak":
            lama = (p.rekomendasi.disclaimer or "").strip()
            gabung = f"{lama} {catatan}".strip()
            p = p.model_copy(update={"rekomendasi": p.rekomendasi.model_copy(update={"disclaimer": gabung})})
        hasil.append(p)
    return hasil


def _angka_tak_terlacak_ke_saran(teks: str, poin_list: list[PoinOutput]) -> str | None:
    """Angka pertama di `teks` yang TIDAK tercantum di saran/langkah_konkret per-poin, atau None.

    Larangan angka total di kesimpulan/narasi membuat LLM menulis "persentase tertentu" dan
    "dimensi minimum tertentu" (APP-2026-UJI-4) — tak berguna bagi reviewer. Angka itu sebenarnya
    sudah final di saran poin (dari kalkulator/BE), jadi mengutipnya persis AMAN. Yang tetap
    dilarang: angka yang tak bisa dilacak ke saran manapun (karangan/hitungan LLM)."""
    sumber = " ".join(p.rekomendasi.saran for p in poin_list)
    sumber += " " + " ".join(l.deskripsi for p in poin_list for l in p.rekomendasi.langkah_konkret)
    for m in _RE_ANGKA_MENCURIGAKAN.finditer(teks):
        angka = m.group(0)
        if re.search(rf"\b{re.escape(angka)}\b", sumber) is None and not _dekat_dgn_pembulatan(angka, sumber):
            return angka
    return None


def _rakit_kesimpulan_fallback(poin_list: list[PoinOutput]) -> KesimpulanOutput:
    """Dipakai kalau panggilan LLM sintesis gagal/melanggar aturan — deterministik, saran
    per-poin verbatim (sudah lolos guardrail, jadi aman ditampilkan apa adanya)."""
    langkah = [p.rekomendasi.saran for p in poin_list if p.rekomendasi.saran.strip()]
    return KesimpulanOutput(langkah_berdampak=langkah, catatan_lokasi=None)


def _alasan_fallback(exc: Exception) -> str:
    return f"fallback: {type(exc).__name__}: {str(exc)[:240]}"


def _rakit_kesimpulan(poin_list: list[PoinOutput], rekomendasi_sistem: str,
                      catatan_diag: dict | None = None) -> KesimpulanOutput:
    """SATU panggilan LLM (bukan retry loop) — gagal atau melanggar aturan angka -> fallback
    deterministik dari saran per-poin."""
    try:
        batas_waktu.periksa(_MIN_SISA_SINTESIS_S, "sintesis kesimpulan")
        prompt = build_kesimpulan_prompt(poin_list, rekomendasi_sistem)
        hasil = llm_client.generate(
            prompt,
            _LLM_RESPONSE_SCHEMA_KESIMPULAN,
            schema_name="kesimpulan",
            system=SYSTEM_PROMPT_KESIMPULAN,
        )
        # Nomor "1." di awal butir dibuang: FE sudah memberi penomoran (APP-2026-UJI-1: nomor terulang).
        langkah = [re.sub(r"^\s*\d+[.)]\s*", "", str(l)) for l in (hasil.get("langkah_berdampak") or [])]
        catatan = hasil.get("catatan_lokasi")
        gabungan = " ".join(langkah) + " " + (catatan or "")
        asing = _angka_tak_terlacak_ke_saran(gabungan, poin_list)
        if asing is not None:
            raise ValueError(
                f"Kesimpulan LLM menyebutkan angka {asing!r} yang tak ada di saran poin — dilarang "
                "(SYSTEM_PROMPT_KESIMPULAN aturan #3)."
            )
        if catatan_diag is not None:
            catatan_diag["kesimpulan"] = "llm"
        return KesimpulanOutput(langkah_berdampak=langkah, catatan_lokasi=catatan)
    except Exception as exc:
        logger.exception("Sintesis kesimpulan via LLM gagal/melanggar aturan — fallback deterministik.")
        if catatan_diag is not None:
            catatan_diag["kesimpulan"] = _alasan_fallback(exc)
        return _rakit_kesimpulan_fallback(poin_list)


def _rakit_narasi_rekomendasi_fallback(poin_list: list[PoinOutput]) -> NarasiRekomendasiOutput:
    """Dipakai kalau panggilan LLM narasi gagal/melanggar aturan — deterministik, gabungan saran
    per-poin verbatim (sudah lolos guardrail) per kelompok paragraf, pola sama dgn
    `_rakit_kesimpulan_fallback`."""
    by_id = {p.poin_id: p for p in poin_list}
    saran_gate = [
        by_id[pid].rekomendasi.saran for pid in ("itbx", "intensitas")
        if pid in by_id and by_id[pid].rekomendasi.saran.strip()
    ]
    poin_dampak = by_id.get("dampak")
    saran_dampak = poin_dampak.rekomendasi.saran if poin_dampak and poin_dampak.rekomendasi.saran.strip() else ""
    return NarasiRekomendasiOutput(
        paragraf_gate_intensitas=" ".join(saran_gate) or "Tidak ada catatan tambahan untuk ITBX dan intensitas bangunan.",
        paragraf_dampak=saran_dampak or "Tidak ada catatan tambahan untuk dampak terhadap lingkungan (hidrologi).",
    )


def _rakit_narasi_rekomendasi(poin_list: list[PoinOutput], rekomendasi_sistem: str,
                              catatan_diag: dict | None = None) -> NarasiRekomendasiOutput:
    """SATU panggilan LLM TAMBAHAN (item permintaan user 2026-09-21) — narasi 2 paragraf ringkas
    berdasar rekomendasi_sistem per-poin (paragraf 1: ITBX+Intensitas, paragraf 2: Dampak Terhadap
    Lingkungan/Hidrologi). Gagal/melanggar aturan angka -> fallback deterministik dari saran per-poin
    (pola identik `_rakit_kesimpulan` di atas — jaga konsistensi 2 panggilan sintesis ini)."""
    try:
        batas_waktu.periksa(_MIN_SISA_SINTESIS_S, "sintesis narasi")
        prompt = build_narasi_rekomendasi_prompt(poin_list, rekomendasi_sistem)
        hasil = llm_client.generate(
            prompt,
            _LLM_RESPONSE_SCHEMA_NARASI_REKOMENDASI,
            schema_name="narasi_rekomendasi",
            system=SYSTEM_PROMPT_NARASI_REKOMENDASI,
        )
        # Titik koma = gaya, bukan fakta: diperbaiki (bukan dibuang seluruh sintesis ke fallback).
        # Log VPS 2026-10-14: satu ';' membuang narasi LLM yang sudah jadi.
        paragraf_gate = _perbaiki_titik_koma(hasil.get("paragraf_gate_intensitas") or "")
        paragraf_dampak = _perbaiki_titik_koma(hasil.get("paragraf_dampak") or "")
        gabungan = f"{paragraf_gate} {paragraf_dampak}"
        asing = _angka_tak_terlacak_ke_saran(gabungan, poin_list)
        if asing is not None:
            raise ValueError(
                f"Narasi rekomendasi LLM menyebutkan angka {asing!r} yang tak ada di saran poin — "
                "dilarang (SYSTEM_PROMPT_NARASI_REKOMENDASI aturan #5)."
            )
        if ";" in gabungan:
            raise ValueError(
                "Narasi rekomendasi LLM memakai tanda titik koma — dilarang "
                "(SYSTEM_PROMPT_NARASI_REKOMENDASI aturan #6)."
            )
        if not paragraf_gate.strip() or not paragraf_dampak.strip():
            raise ValueError("Narasi rekomendasi LLM mengembalikan paragraf kosong.")
        if catatan_diag is not None:
            catatan_diag["narasi"] = "llm"
        return NarasiRekomendasiOutput(paragraf_gate_intensitas=paragraf_gate, paragraf_dampak=paragraf_dampak)
    except Exception as exc:
        logger.exception("Sintesis narasi rekomendasi via LLM gagal/melanggar aturan — fallback deterministik.")
        if catatan_diag is not None:
            catatan_diag["narasi"] = _alasan_fallback(exc)
        return _rakit_narasi_rekomendasi_fallback(poin_list)


CAVEAT_DI_LUAR_CAKUPAN = (
    "Koordinat permohonan berada di luar delineasi wilayah RDTR yang menjadi dasar seluruh sitasi "
    "di dokumen ini — dasar hukum yang dikutip kemungkinan tidak berlaku untuk lokasi ini dan wajib "
    "ditinjau manual."
)


def _intensitas_dinilai(poin: PoinOutput) -> bool:
    """Poin intensitas yang benar-benar menilai ambang KDB/KLB/KDH.

    Kalau back-end tak mengirim penilaian intensitas, adapter TETAP merakit poinnya dengan
    status "Tidak Dinilai" dan `fakta={"dinilai": False}` — tanpa parameter, tanpa ambang,
    tanpa tabel Lampiran VI yang dikutip. Caveat sub-zona di situ memperingatkan soal angka
    yang tidak ada, jadi ia disaring di sini (ditemukan lewat fixture
    l2_sample_itbx_x_tanpa_intensitas.json).
    """
    return poin.poin_id == "intensitas" and poin.status != "Tidak Dinilai"


def _rakit_catatan_global(
    assessment: L2Assessment,
    itbx_fallback: bool,
    poin_list: list[PoinOutput],
    di_luar_wilayah: bool = False,
    subzona_tak_pasti: bool = False,
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

    # Hanya kalau ada poin intensitas: tanpa poin itu, tak ada ambang KDB/KLB/KDH yang dikutip
    # dan caveat ini cuma jadi derau. Lapis ini DETERMINISTIK — prompt juga membawa caveat yang
    # sama (generator.caveat_subzona) supaya narasinya tak mengklaim lebih dari yang diketahui,
    # tapi narasi bergantung LLM; yang ini tidak, jadi pembaca tetap melihatnya walau narasi
    # jatuh ke template low_confidence.
    if subzona_tak_pasti and any(_intensitas_dinilai(p) for p in poin_list):
        catatan.append(CAVEAT_SUBZONA_TAK_TERKONFIRMASI)

    poin_low_confidence = [p.poin_id for p in poin_list if p.low_confidence]
    if poin_low_confidence:
        daftar = ", ".join(poin_low_confidence)
        catatan.append(
            f"Sebagian penjelasan (poin: {daftar}) tidak dapat dihasilkan otomatis dan perlu "
            "peninjauan manual."
        )

    return catatan


CATATAN_BATAS_WAKTU = (
    "Batas waktu pemrosesan tercapai, sehingga sebagian penjelasan tidak sempat dihasilkan otomatis dan "
    "perlu ditinjau manual atau permohonan diulang."
)


def jalankan_precheck(assessment: L2Assessment, retriever: Retriever) -> OutputL3:
    """Jalankan precheck penuh: generate tiap poin PARALEL (dengan guardrail), rakit output dua-jalur.

    Seluruh pemrosesan berjalan dalam anggaran waktu (`app/batas_waktu.py`, REASONING_DEADLINE_S, bawaan
    85 detik) supaya respons tiba sebelum batas 100 detik Cloudflare."""
    with batas_waktu.anggaran():
        return _jalankan_precheck_inti(assessment, retriever)


def _jalankan_precheck_inti(assessment: L2Assessment, retriever: Retriever) -> OutputL3:
    mulai = time.perf_counter()
    hasil_adaptasi = adaptasi(assessment)
    itbx_fallback = deteksi_fallback_itbx(assessment.gate_hukum.tahapan.itbx.reason)
    # Dihitung SEKALI di sini (bukan per poin): ini properti permohonan, bukan properti poin.
    # Retrieval dikunci ke satu wilayah lewat RETRIEVER_WILAYAH, jadi kalau lokasinya di luar
    # wilayah itu, SELURUH sitasi di output ini berpotensi tak berlaku — bukan cuma satu poin.
    di_luar_wilayah = di_luar_cakupan(assessment.lokasi)
    # Idem: sub-zona presisi adalah properti LOKASI, bukan properti poin. Kalau back-end tak
    # mengonfirmasinya, seluruh sitasi intensitas di output ini berasal dari tabel tingkat
    # keluarga zona (lihat generator._QUERY_INTENSITAS_TAJAM).
    subzona_tak_pasti = not (assessment.lokasi.rdtr_subzone or "").strip()

    # Konteks (anggaran waktu) disalin per thread: ContextVar tidak otomatis ikut ke thread pool.
    futures = [
        _executor.submit(contextvars.copy_context().run, _generate_poin_defensif, poin, retriever, assessment)
        for poin in hasil_adaptasi.poin
    ]
    hasil_poin = []
    for f, poin in zip(futures, hasil_adaptasi.poin):  # urutan submit == urutan hasil (itbx, intensitas, dampak)
        sisa = batas_waktu.sisa()
        try:
            hasil_poin.append(f.result(timeout=None if sisa is None else max(0.5, sisa + _TOLERANSI_TUNGGU_S)))
        except FuturesTimeoutError:
            logger.error("Poin %r tak selesai dalam anggaran waktu — fallback low_confidence.", poin.poin_id)
            hasil_poin.append(_poin_batas_waktu(poin))
    poin_list = _terapkan_catatan_batas([p for p, _ in hasil_poin], hasil_adaptasi.poin)
    diagnostik = [d for _, d in hasil_poin]

    # Batas yang lebih ketat (intensitas vs dampak) harus terbaca di NARASI dan KESIMPULAN juga, bukan
    # hanya di disclaimer: pembaca ringkasan akan mengikuti angka dampak yang lebih longgar.
    catatan_batas = _catatan_batas_berbeda(hasil_adaptasi.poin)
    sintesis: dict = {}
    # Narasi dan kesimpulan saling bebas: dijalankan PARALEL (menghemat satu panggilan LLM berurutan).
    f_narasi = _executor.submit(contextvars.copy_context().run, _rakit_narasi_rekomendasi,
                                poin_list, hasil_adaptasi.rekomendasi_sistem, sintesis)
    f_kesimpulan = _executor.submit(contextvars.copy_context().run, _rakit_kesimpulan,
                                    poin_list, hasil_adaptasi.rekomendasi_sistem, sintesis)
    narasi, kesimpulan = f_narasi.result(), f_kesimpulan.result()
    if catatan_batas:
        narasi = narasi.model_copy(update={"paragraf_dampak": f"{narasi.paragraf_dampak} {catatan_batas}".strip()})
        kesimpulan = kesimpulan.model_copy(update={"langkah_berdampak": [*kesimpulan.langkah_berdampak, catatan_batas]})

    output = OutputL3(
        ringkasan_gate=_rakit_ringkasan_gate(assessment, itbx_fallback),
        ringkasan_dampak=_rakit_ringkasan_dampak(assessment),
        poin=poin_list,
        rekomendasi_sistem=hasil_adaptasi.rekomendasi_sistem,
        narasi_rekomendasi=narasi,
        kesimpulan=kesimpulan,
        catatan_global=_rakit_catatan_global(assessment, itbx_fallback, poin_list,
                                             di_luar_wilayah, subzona_tak_pasti),
        # Di luar cakupan = seluruh dasar hukum patut diragukan -> tandai low_confidence walau
        # ketiga poin sendiri lolos guardrail dgn mulus.
        low_confidence_keseluruhan=any(p.low_confidence for p in poin_list) or di_luar_wilayah,
    )
    if any(d.sebab() == "melewati_batas_waktu" for d in diagnostik) or any(
            "BatasWaktuTerlampaui" in str(v) for v in sintesis.values()):
        output.catatan_global.append(CATATAN_BATAS_WAKTU)
        output.low_confidence_keseluruhan = True

    try:
        log_precheck(assessment, output, diagnostik=_serialisasi_diagnostik(diagnostik), sintesis=sintesis,
                     durasi_detik=round(time.perf_counter() - mulai, 2))
    except Exception:
        logger.exception("Gagal menulis log precheck — melanjutkan tanpa menggagalkan respons.")

    observability.catat_precheck_trace(
        "jalankan_precheck", assessment.model_dump(mode="json"), output.model_dump(mode="json")
    )

    return output
