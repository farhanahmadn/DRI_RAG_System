"""Template kalimat deterministik — TIDAK ADA panggilan LLM di sini.

Dua kasus dari CLAUDE.md § Konvensi kode:
- Poin "aman" (mis. ITBX Diizinkan tanpa fallback, intensitas tanpa pelanggaran, dampak tanpa
  mitigasi) -> template, tanpa memanggil LLM (hemat kuota — pola dipertahankan dari model lama).
- Guardrail gagal -> ... -> fallback template + tanda low_confidence.

Status di kedua template SELALU `poin.status` apa adanya (ground truth back-end/adapter) — model L2
tidak punya skor per-poin lagi, jadi tidak ada `status_dari_skor` seperti versi lama.
"""

from app.reasoning.calculator import pilih_target_utama_intensitas
from app.schemas import PoinKonteks, PoinOutput, RekomendasiOutput


def _target_untuk_poin(poin: PoinKonteks) -> float | str | None:
    if poin.tipe_rekomendasi != "numerik":
        return None
    return pilih_target_utama_intensitas(poin.fakta.get("target") or {})


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
        ),
        low_confidence=True,
    )
