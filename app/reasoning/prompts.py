"""System prompt & builder pesan untuk generator.py — murni string building, TIDAK ADA panggilan LLM.

Prinsip CLAUDE.md: Faithful (jelaskan skor, jangan kontradiksi), Grounded (sitasi harus nyata),
Deterministik di tempat presisi (LLM hanya bahasa, angka dari kode).
"""

from app.retrieval.base import Chunk
from app.schemas import IndikatorJejak

SYSTEM_PROMPT = """Anda adalah asisten reasoning untuk sistem pre-check risiko izin bangunan Kabupaten Sleman.

ATURAN WAJIB (jangan dilanggar):
1. Skor risiko dan verdict pada jejak aturan yang diberikan bersifat FINAL. Tugas Anda HANYA menjelaskan mengapa skor tersebut muncul — JANGAN mengubah, menghitung ulang, atau membuat kesimpulan yang mengontradiksi skor tersebut.
2. Hanya sebutkan fakta spasial (LP2B, banjir, resapan, jarak sungai, dst.) yang secara eksplisit diberikan dalam data fakta_spasial. JANGAN mengarang detail geometri, koordinat, atau kondisi lokasi lain yang tidak ada di data.
3. Sitasi HANYA boleh diambil dari daftar pasal yang diberikan, dengan menyebut citation_id persis seperti yang tercantum. JANGAN mengarang nomor pasal, ayat, atau dokumen yang tidak ada di daftar.
4. JANGAN menyebutkan angka rekomendasi (target, selisih, dsb) dalam narasi Anda — angka tersebut dihitung dan digabungkan otomatis oleh sistem secara terpisah; tugas Anda hanya menulis narasi kualitatif.
5. Bedakan dengan jelas antara "pelanggaran hukum/regulasi" (mis. melanggar ambang KDB, alih fungsi LP2B tanpa izin) dan "faktor risiko alam/lokasi" (mis. dekat sungai, rawan banjir) — jangan mencampur keduanya seolah setara.
6. Tulis dalam Bahasa Indonesia yang jelas, ringkas, dan mudah dipahami warga awam, bukan bahasa hukum yang kaku.
7. Kalau ada baris "STATUS: ..." atau "KLASIFIKASI: ..." pada data di bawah, itu adalah FAKTA hasil perhitungan kode. Gunakan APA ADANYA dalam narasi Anda — JANGAN menyimpulkan status/klasifikasi sendiri dari nilai_input/ambang.
8. Kalau ada daftar "Kegiatan Diizinkan di Zona Ini", kegiatan alternatif yang Anda sebutkan HARUS berasal dari daftar itu — JANGAN mengarang nama kegiatan lain yang tidak ada di daftar.

Balas HANYA dalam format JSON sesuai skema yang diberikan."""


def _format_chunk(chunk: Chunk) -> str:
    lokasi = f"{chunk.dokumen} Pasal {chunk.pasal}" if chunk.pasal else chunk.dokumen
    if chunk.ayat:
        lokasi += f" Ayat {chunk.ayat}"
    if chunk.istilah_kode:
        lokasi += f" ({chunk.istilah_kode})"
    if chunk.halaman is not None:
        lokasi += f" (hal. {chunk.halaman})"
    return f"- citation_id={chunk.id} | {lokasi}\n  Teks: {chunk.teks}"


def build_user_prompt(
    indikator: IndikatorJejak,
    chunks: list[Chunk],
    target: dict[str, float] | None,
    catatan_perbaikan: str | None = None,
    fakta_verdict: str | None = None,
    kegiatan_diizinkan: list[str] | None = None,
) -> str:
    """Susun prompt user, deterministik dari jejak aturan + chunk yang diretrieve + target calculator."""
    lines: list[str] = []

    lines.append("## Jejak Aturan (SUDAH FINAL — jangan diubah/dihitung ulang)")
    lines.append(f"- poin_id: {indikator.poin_id}")
    lines.append(f"- kategori: {indikator.kategori}")
    lines.append(f"- skor: {indikator.skor} (kontribusi: {indikator.kontribusi} dari bobot {indikator.bobot})")
    lines.append(
        f"- nilai_input: {indikator.nilai_input}, ambang: {indikator.ambang}, operator: {indikator.operator}"
    )
    lines.append(f"- formula: {indikator.formula}")
    if indikator.zona:
        lines.append(f"- zona: {indikator.zona}")

    if fakta_verdict:
        lines.append("")
        lines.append("## STATUS Perbandingan (FAKTA sudah dihitung kode — jangan disimpulkan ulang)")
        lines.append(fakta_verdict)

    if kegiatan_diizinkan:
        lines.append("")
        lines.append(
            "## Kegiatan Diizinkan di Zona Ini (FAKTA — pilih dari sini saja, JANGAN mengarang kegiatan lain)"
        )
        for kegiatan in kegiatan_diizinkan:
            lines.append(f"- {kegiatan}")

    if indikator.fakta_spasial is not None:
        fakta_items = {
            k: v for k, v in indikator.fakta_spasial.model_dump().items() if v is not None
        }
        if fakta_items:
            lines.append("")
            lines.append("## Fakta Spasial (HANYA sebut yang ada di sini, jangan tambah detail lain)")
            for k, v in fakta_items.items():
                lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Pasal/Sitasi Tersedia (HANYA boleh kutip dari daftar ini, pakai citation_id persis)")
    if chunks:
        for chunk in chunks:
            lines.append(_format_chunk(chunk))
    else:
        lines.append(
            "(Tidak ada pasal yang berhasil diambil — jangan mengarang sitasi apa pun, "
            "kosongkan daftar sitasi.)"
        )

    lines.append("")
    if target:
        lines.append("## Target Rekomendasi (dari kalkulator, angka ini SUDAH FINAL)")
        for k, v in target.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("## Target Rekomendasi")
        lines.append(
            "Tidak ada target numerik untuk indikator ini — rekomendasi berupa penjelasan, bukan angka."
        )

    if catatan_perbaikan:
        lines.append("")
        lines.append("## Catatan Perbaikan (percobaan sebelumnya gagal)")
        lines.append(catatan_perbaikan)
        lines.append("Perbaiki hal ini pada jawaban Anda kali ini.")

    lines.append("")
    lines.append(
        "## Tugas\n"
        "Jelaskan mengapa indikator ini berisiko berdasarkan jejak aturan dan pasal di atas, dalam "
        "Bahasa Indonesia yang jelas untuk warga awam. Berikan reasoning_pendek (1-2 kalimat), "
        "reasoning_panjang (paragraf lengkap), sitasi (rujuk citation_id di atas saja), dan saran "
        "tindak lanjut."
    )

    return "\n".join(lines)
