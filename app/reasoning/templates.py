"""Template kalimat deterministik — TIDAK ADA panggilan LLM di sini.

Dua kasus dari CLAUDE.md § Konvensi kode:
- Indikator skor 0 ("Aman") -> template, tanpa memanggil LLM.
- Guardrail gagal -> ... -> fallback template + tanda low_confidence.
"""

from app.reasoning.calculator import hitung_target_rekomendasi, klasifikasi_tipe_rekomendasi
from app.schemas import IndikatorJejak, PoinOutput, RekomendasiOutput


def _pilih_target_utama(target: dict[str, float] | None) -> float | None:
    if target is None:
        return None
    if "target_maks" in target:
        return target["target_maks"]
    return target.get("target")


def _status_dari_skor(skor: float) -> str:
    return "Aman" if skor <= 0 else "Tidak Aman"


def template_aman(indikator: IndikatorJejak) -> PoinOutput:
    """Template untuk indikator yang skornya 0 ('Aman') — tidak memanggil LLM sama sekali."""
    if indikator.skor != 0:
        raise ValueError(
            f"template_aman hanya untuk indikator skor 0, indikator {indikator.poin_id!r} "
            f"punya skor={indikator.skor}."
        )

    target = hitung_target_rekomendasi(indikator)

    reasoning_pendek = f"Indikator {indikator.kategori} dinyatakan Aman."
    reasoning_panjang = (
        f"Berdasarkan jejak aturan, nilai input ({indikator.nilai_input}) untuk indikator "
        f"{indikator.kategori} ({indikator.poin_id}) memenuhi ambang batas yang berlaku "
        f"({indikator.operator} {indikator.ambang}), sehingga tidak berkontribusi terhadap risiko."
    )

    return PoinOutput(
        poin_id=indikator.poin_id,
        kategori=indikator.kategori,
        status="Aman",
        kontribusi=indikator.kontribusi,
        reasoning_pendek=reasoning_pendek,
        reasoning_panjang=reasoning_panjang,
        sitasi=[],
        rekomendasi=RekomendasiOutput(
            tipe=klasifikasi_tipe_rekomendasi(indikator.kategori),
            target=_pilih_target_utama(target),
            saran="Tidak diperlukan tindakan khusus untuk indikator ini.",
            disclaimer=None,
        ),
        low_confidence=False,
    )


def template_low_confidence(indikator: IndikatorJejak) -> PoinOutput:
    """Fallback saat generasi LLM gagal validasi guardrail berulang kali.

    Status & target tetap dihitung deterministik dari jejak/calculator (kode, bukan LLM) — yang
    gagal cuma narasi bahasa, bukan angka.
    """
    status = _status_dari_skor(indikator.skor)
    target = hitung_target_rekomendasi(indikator)

    reasoning_pendek = "Penjelasan otomatis tidak tersedia untuk indikator ini."
    reasoning_panjang = (
        f"Sistem tidak berhasil menghasilkan penjelasan yang memenuhi standar validasi untuk "
        f"indikator {indikator.kategori} ({indikator.poin_id}) setelah beberapa kali percobaan. "
        f"Skor ({indikator.skor}) dan kontribusi ({indikator.kontribusi}) tetap berdasarkan jejak "
        "aturan asli; mohon dilakukan peninjauan manual oleh petugas."
    )

    return PoinOutput(
        poin_id=indikator.poin_id,
        kategori=indikator.kategori,
        status=status,
        kontribusi=indikator.kontribusi,
        reasoning_pendek=reasoning_pendek,
        reasoning_panjang=reasoning_panjang,
        sitasi=[],
        rekomendasi=RekomendasiOutput(
            tipe=klasifikasi_tipe_rekomendasi(indikator.kategori),
            target=_pilih_target_utama(target),
            saran="Perlu peninjauan manual oleh petugas terkait indikator ini.",
            disclaimer=(
                "Penjelasan otomatis tidak tersedia untuk indikator ini; perlu verifikasi manual."
            ),
        ),
        low_confidence=True,
    )
