"""Template kalimat deterministik — TIDAK ADA panggilan LLM di sini.

Dua kasus dari CLAUDE.md § Konvensi kode:
- Poin "aman" (mis. ITBX Diizinkan tanpa fallback, intensitas tanpa pelanggaran, dampak tanpa
  mitigasi) -> template, tanpa memanggil LLM (hemat kuota — pola dipertahankan dari model lama).
- Guardrail gagal -> ... -> fallback template + tanda low_confidence.

Status di kedua template SELALU `poin.status` apa adanya (ground truth back-end/adapter) — model L2
tidak punya skor per-poin lagi, jadi tidak ada `status_dari_skor` seperti versi lama.
"""

from app.reasoning.calculator import (
    bangun_langkah_konkret_dampak,
    bangun_langkah_konkret_intensitas,
    pilih_target_mitigasi_dampak,
    pilih_target_utama_intensitas,
)
from app.schemas import LangkahKonkretOutput, PoinKonteks, PoinOutput, RekomendasiOutput


def _target_untuk_poin(poin: PoinKonteks) -> float | str | None:
    # Bug ditemukan live (2026-08-18): sebelumnya HANYA menangani "numerik" (intensitas) — poin
    # dampak ("numerik-mitigasi") yang jatuh ke template ini SELALU dapat target=None, walau
    # fakta['target_mitigasi'] sebenarnya ada. Disamakan dgn app/reasoning/generator.py::
    # generate_poin() & guardrail.py::perbaiki_poin (2 tempat lain yang menghitung target) —
    # SEHARUSNYA 3 tempat ini selalu selaras, JANGAN ditambah tanpa update ketiganya.
    if poin.tipe_rekomendasi == "numerik":
        return pilih_target_utama_intensitas(poin.fakta.get("target") or {})
    if poin.tipe_rekomendasi == "numerik-mitigasi":
        return pilih_target_mitigasi_dampak(poin.fakta.get("target_mitigasi") or {})
    return None


def _langkah_konkret_untuk_poin(poin: PoinKonteks) -> list[LangkahKonkretOutput]:
    # Bug ditemukan live (2026-08-18): langkah_konkret ditambahkan ke generate_poin()/perbaiki_poin
    # tapi LUPA di-wire ke template fallback ini — poin yang jatuh low_confidence SELALU dapat
    # langkah_konkret=[] kosong, walau `target` di atas terisi benar (ambang tetap dari kalkulator,
    # bukan LLM, jadi HARUSNYA tetap tersedia persis seperti target).
    if poin.tipe_rekomendasi == "numerik":
        mentah = bangun_langkah_konkret_intensitas(
            poin.fakta.get("parameter") or {}, poin.fakta.get("target") or {}
        )
    elif poin.tipe_rekomendasi == "numerik-mitigasi":
        mentah = bangun_langkah_konkret_dampak(poin.fakta.get("target_mitigasi") or {})
    else:
        mentah = []
    return [LangkahKonkretOutput(**item) for item in mentah]


def template_aman(poin: PoinKonteks) -> PoinOutput:
    """Template untuk poin yang jelas aman/lolos — tidak memanggil LLM sama sekali."""
    reasoning_pendek = f"{poin.kategori}: tidak ada catatan berisiko untuk poin ini."
    reasoning_panjang = (
        f"Berdasarkan data yang diberikan, poin '{poin.kategori}' berstatus '{poin.status}' dan "
        "tidak memerlukan tindakan lebih lanjut."
    )

    return PoinOutput(
        poin_id=poin.poin_id,
        kategori=poin.kategori,
        status=poin.status,
        reasoning_pendek=reasoning_pendek,
        reasoning_panjang=reasoning_panjang,
        sitasi=[],
        rekomendasi=RekomendasiOutput(
            tipe=poin.tipe_rekomendasi,
            target=_target_untuk_poin(poin),
            saran="Tidak diperlukan tindakan khusus untuk poin ini.",
            disclaimer=None,
            langkah_konkret=_langkah_konkret_untuk_poin(poin),
        ),
        low_confidence=False,
    )


def template_low_confidence(poin: PoinKonteks) -> PoinOutput:
    """Fallback saat generasi LLM gagal validasi guardrail berulang kali.

    Status & target tetap dari fakta/calculator (kode, bukan LLM) — yang gagal cuma narasi bahasa.
    """
    reasoning_pendek = "Penjelasan otomatis tidak tersedia untuk poin ini."
    reasoning_panjang = (
        f"Sistem tidak berhasil menghasilkan penjelasan yang memenuhi standar validasi untuk poin "
        f"'{poin.kategori}' ({poin.poin_id}) setelah beberapa kali percobaan. Status ({poin.status}) "
        "tetap berdasarkan data asli dari back-end; mohon dilakukan peninjauan manual oleh petugas."
    )

    return PoinOutput(
        poin_id=poin.poin_id,
        kategori=poin.kategori,
        status=poin.status,
        reasoning_pendek=reasoning_pendek,
        reasoning_panjang=reasoning_panjang,
        sitasi=[],
        rekomendasi=RekomendasiOutput(
            tipe=poin.tipe_rekomendasi,
            target=_target_untuk_poin(poin),
            saran="Perlu peninjauan manual oleh petugas terkait poin ini.",
            disclaimer="Penjelasan otomatis tidak tersedia untuk poin ini; perlu verifikasi manual.",
            langkah_konkret=_langkah_konkret_untuk_poin(poin),
        ),
        low_confidence=True,
    )
