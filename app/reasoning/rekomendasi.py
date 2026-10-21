"""Turunkan rekomendasi_sistem secara deterministik dari gate_hukum + impact_assessment.

Murni fungsi, tanpa I/O/LLM — sesuai prinsip "Deterministik di tempat presisi" (CLAUDE.md).

Mengikuti kontrak BE "Spesifikasi Kontrak Variabel Output AI Reasoning (L3)" v1.0.0 (2026-10-20), bagian 4.
Nilai sah `rekomendasi_sistem`: Setuju, Setuju Bersyarat, Lolos Terbatas, Tolak.
"""

# Dampak yang MEMERLUKAN mitigasi (spesifikasi BE bagian 4: "Sedang/Tinggi/Sangat Tinggi yang memerlukan
# mitigasi"). Sebelum 2026-10-20 hanya Tinggi/Sangat Tinggi; Sedang kini ikut karena spesifikasi BE
# menjadikannya pemicu Setuju Bersyarat.
KATEGORI_DAMPAK_BERSYARAT = {"Sedang", "Tinggi", "Sangat Tinggi"}

REKOMENDASI_TOLAK = "Tolak"
REKOMENDASI_TERBATAS = "Lolos Terbatas"
REKOMENDASI_BERSYARAT = "Setuju Bersyarat"
REKOMENDASI_SETUJU = "Setuju"


def gate_efektif(gate_status: str, status_itbx: str | None) -> str:
    """Status gate setelah ITBX diperlakukan sebagai penentu tunggal (Sovereign Gate, tanggapan BE 2026-10-21):
    X -> Tidak Lolos, T/TB -> Lolos Terbatas, I -> Lolos walau BE melabeli Lolos Bersyarat (intensitas melampaui
    batas tidak menurunkan kelolosan legal; ia tetap dicatat sebagai MELAMPAUI_BATAS beserta saran reduksi)."""
    if gate_status == "Tidak Lolos" or status_itbx == "X":
        return "Tidak Lolos"
    if gate_status == "Lolos Terbatas" or status_itbx in ("T", "TB"):
        return "Lolos Terbatas"
    if status_itbx == "I" and gate_status == "Lolos Bersyarat":
        return "Lolos"
    return gate_status


def turunkan_rekomendasi(gate_status: str, impact_category: str | None, status_itbx: str | None = None) -> str:
    """Matriks rekomendasi_sistem, urut prioritas (kondisi yang lebih tinggi mengalahkan yang di bawahnya).

    `impact_category` HARUS sudah dinormalisasi Title Case oleh caller
    (lihat app/reasoning/calculator.py::normalisasi_kategori_dampak) sebelum dipanggil di sini.

    1. Tolak: gate Tidak Lolos, atau ITBX = X.
    2. Lolos Terbatas: gate Lolos Terbatas, atau ITBX = T (juga TB: unsur terbatas).
    3. Setuju Bersyarat: gate Lolos Bersyarat, ITBX = B (intensitas melampaui batas TIDAK ikut menentukan bila
       ITBX = I; ITBX penentu tunggal gate legal), atau dampak
       Sedang/Tinggi/Sangat Tinggi yang memerlukan mitigasi.
    4. Setuju: semua aspek legal I, intensitas patuh, dampak Rendah (atau belum dinilai: absennya penilaian
       bukan alasan untuk membersyaratkan).

    `status_itbx` opsional: bila diberikan, ia ikut menentukan (gate BE kadang melabeli ITBX T sebagai Lolos
    Bersyarat, padahal menurut spesifikasi harus Lolos Terbatas)."""
    if gate_status not in ("Lolos", "Lolos Bersyarat", "Lolos Terbatas", "Tidak Lolos"):
        raise ValueError(f"gate_status tidak dikenal: {gate_status!r}")
    gate_status = gate_efektif(gate_status, status_itbx)
    if gate_status == "Tidak Lolos":
        return REKOMENDASI_TOLAK
    if gate_status == "Lolos Terbatas":
        return REKOMENDASI_TERBATAS
    if gate_status == "Lolos Bersyarat" or status_itbx == "B":
        return REKOMENDASI_BERSYARAT
    if impact_category in KATEGORI_DAMPAK_BERSYARAT:
        return REKOMENDASI_BERSYARAT
    return REKOMENDASI_SETUJU
