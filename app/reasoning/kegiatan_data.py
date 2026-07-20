"""Data kegiatan diizinkan (klasifikasi I) per zona — dari matriks ITBX RDTR.

ITEM KONTRAK: idealnya field ini (atau referensinya) dikirim back-end / diambil dari tabel
terstruktur RDTR asli (CLAUDE.md § Data: "matriks/tabel ... ke tabel terstruktur, bukan
embedding"). Untuk sekarang di-mock lokal supaya generator/guardrail bisa diuji end-to-end sebelum
data asli tersedia. Ganti isi `_KEGIATAN_DIIZINKAN_PER_ZONA` (atau sumbernya) saat data backend
sudah siap — kode pemanggil (`kegiatan_diizinkan_untuk_prompt`) tidak perlu berubah.
"""

from app.reasoning.calculator import klasifikasi_tipe_rekomendasi
from app.schemas import IndikatorJejak

_KEGIATAN_DIIZINKAN_PER_ZONA: dict[str, list[str]] = {
    "C-1": [
        "Rumah toko (ruko) skala kecil",
        "Perdagangan eceran",
        "Jasa perkantoran skala kecil",
    ],
}


def cari_kegiatan_diizinkan(zona: str | None) -> list[str]:
    """Daftar kegiatan berklasifikasi I (diizinkan) di suatu zona. [] kalau zona None/tidak dikenal."""
    if zona is None:
        return []
    return _KEGIATAN_DIIZINKAN_PER_ZONA.get(zona, [])


def kegiatan_diizinkan_untuk_prompt(indikator: IndikatorJejak) -> list[str] | None:
    """Daftar kegiatan diizinkan untuk disuntik ke prompt — hanya relevan saat klasifikasi X (dilarang)."""
    if klasifikasi_tipe_rekomendasi(indikator.kategori) != "kegiatan":
        return None

    klasifikasi = str(indikator.nilai_input).strip().upper()
    if klasifikasi != "X":
        return None

    return cari_kegiatan_diizinkan(indikator.zona)
