"""Guardrail — cek murah deterministik atas PoinOutput SEBELUM diloloskan, + loop retry + fallback.

CLAUDE.md: "Guardrail gagal -> regenerasi terarah (maks 1-2x) -> fallback template + tanda
low_confidence. JANGAN loop tak terbatas." Ini lapisan pertahanan TERPISAH dari generator.py —
memeriksa ulang PoinOutput apa pun sumbernya, bukan mengandalkan generator.py selalu benar.

Verifikasi entailment sitasi/verdict (NLI/LLM) DITUNDA sampai eval membuktikan perlu — lihat stub
`verifikasi_entailment_sitasi` di bawah, tidak dipanggil di alur utama.

TODO(pipeline-rewire model L2): setelah generate_poin_dengan_guardrail dipindah ke L2Assessment,
panggil app.adapter.cek_konsistensi_intensitas(assessment) di sini. Non-kosong -> tandai
low_confidence=True + log detail masalah. JANGAN menimpa/mengoreksi field dari back-end
(final_gate_status/status/parameter tetap dipakai apa adanya, sesuai prinsip Faithful CLAUDE.md) —
guardrail hanya menurunkan tingkat kepercayaan output, bukan "membetulkan" data back-end.
"""

from app.reasoning.calculator import hitung_target_rekomendasi, pilih_target_utama
from app.reasoning.generator import ambil_chunks_pendukung, generate_poin
from app.reasoning.templates import status_dari_skor, template_aman, template_low_confidence
from app.retrieval.base import Chunk, Retriever
from app.schemas import IndikatorJejak, PoinOutput, SitasiOutput


def perbaiki_poin(
    poin: PoinOutput,
    indikator: IndikatorJejak,
    chunks: list[Chunk],
) -> tuple[PoinOutput, list[str]]:
    """Cek & perbaiki PoinOutput terhadap ground truth (jejak/calculator/chunk).

    Mengembalikan (poin_hasil_perbaikan, daftar_masalah). Daftar_masalah kosong berarti poin siap
    diloloskan; tidak kosong berarti perlu regenerasi teks (reasoning/saran/sitasi kosong).
    """
    masalah: list[str] = []

    target_benar = pilih_target_utama(hitung_target_rekomendasi(indikator))
    rekomendasi_bersih = poin.rekomendasi.model_copy(update={"target": target_benar})

    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    sitasi_bersih: list[SitasiOutput] = []
    if chunks:
        for s in poin.sitasi:
            chunk = chunk_by_id.get(s.citation_id)
            if chunk is None:
                continue
            sitasi_bersih.append(
                SitasiOutput(
                    citation_id=chunk.id,
                    dokumen=chunk.dokumen,
                    pasal=chunk.pasal or "",
                    halaman=chunk.halaman or 0,
                    kutipan=s.kutipan,
                    terverifikasi=True,
                )
            )
    # chunks kosong -> sitasi_bersih tetap [] (paksa)

    status_benar = status_dari_skor(indikator.skor)

    poin_bersih = poin.model_copy(
        update={
            "status": status_benar,
            "kontribusi": indikator.kontribusi,
            "sitasi": sitasi_bersih,
            "rekomendasi": rekomendasi_bersih,
        }
    )

    if not poin_bersih.reasoning_pendek.strip() or not poin_bersih.reasoning_panjang.strip():
        masalah.append("reasoning_pendek/reasoning_panjang kosong.")
    if not poin_bersih.rekomendasi.saran.strip():
        masalah.append("rekomendasi.saran kosong.")
    if chunks and not sitasi_bersih:
        masalah.append(
            "Pasal tersedia tapi tidak ada sitasi valid setelah verifikasi — kemungkinan narasi "
            "tidak grounded pada pasal yang diberikan."
        )

    return poin_bersih, masalah


def verifikasi_entailment_sitasi(poin: PoinOutput, chunks: list[Chunk]) -> bool:
    """TODO(Fase 3+): verifikasi entailment sitasi/verdict via model NLI kecil atau panggilan LLM.

    DITUNDA sampai eval membuktikan perlu (CLAUDE.md § Guardrail). Tidak dipanggil di alur utama —
    placeholder untuk pengembangan berikutnya.
    """
    return True


def generate_poin_dengan_guardrail(
    indikator: IndikatorJejak,
    retriever: Retriever,
    *,
    max_retry: int = 2,
) -> PoinOutput:
    """Entrypoint utama: generate_poin + guardrail + retry terarah + fallback low_confidence."""
    if indikator.skor == 0:
        return template_aman(indikator)

    chunks = ambil_chunks_pendukung(indikator, retriever)

    masalah: list[str] = []
    for percobaan in range(max_retry + 1):
        catatan = "; ".join(masalah) if percobaan > 0 else None
        suhu = 0.4 if percobaan > 0 else 0.0

        try:
            poin = generate_poin(
                indikator,
                retriever,
                catatan_perbaikan=catatan,
                temperature=suhu,
            )
        except Exception as exc:  # generasi gagal dihitung sebagai percobaan gagal, bukan crash
            masalah = [str(exc)]
            continue

        poin_bersih, masalah = perbaiki_poin(poin, indikator, chunks)
        if not masalah:
            return poin_bersih

    return template_low_confidence(indikator)
