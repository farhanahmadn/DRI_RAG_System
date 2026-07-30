"""Turunkan rekomendasi_sistem secara deterministik dari gate_hukum + impact_assessment.

Murni fungsi, tanpa I/O/LLM — sesuai prinsip "Deterministik di tempat presisi" (CLAUDE.md).
"""

KATEGORI_DAMPAK_BERSYARAT = {"Tinggi", "Sangat Tinggi"}


def turunkan_rekomendasi(gate_status: str, impact_category: str | None) -> str:
    """Matriks rekomendasi_sistem.

    `impact_category` HARUS sudah dinormalisasi Title Case oleh caller
    (lihat app/reasoning/calculator.py::normalisasi_kategori_dampak) sebelum dipanggil di sini.

    - Tidak Lolos -> Tidak Setuju (short-circuit, kategori dampak tidak relevan).
    - Lolos Bersyarat -> Setuju Bersyarat (pelanggaran intensitas sudah cukup, kategori dampak tidak
      bisa menaikkan/menurunkan ini lagi).
    - Lolos + dampak Tinggi/Sangat Tinggi -> Setuju Bersyarat.
    - Lolos + dampak lainnya (termasuk None/belum dinilai) -> Setuju. Dampak yang belum dinilai
      TIDAK menahan approval (asumsi: absennya penilaian bukan alasan untuk membersyaratkan).
    """
    if gate_status == "Tidak Lolos":
        return "Tidak Setuju"
    if gate_status == "Lolos Bersyarat":
        return "Setuju Bersyarat"
    if gate_status == "Lolos":
        if impact_category in KATEGORI_DAMPAK_BERSYARAT:
            return "Setuju Bersyarat"
        return "Setuju"
    raise ValueError(f"gate_status tidak dikenal: {gate_status!r}")
