"""System prompt & builder pesan untuk generator.py — murni string building, TIDAK ADA panggilan LLM.

Pola fact-injection (CLAUDE.md, docs/Blueprint-Adaptasi-Model-L2-Gate-Impact.md §5) TIDAK berubah
dari model lama: fakta (status/verdict/angka/kategori) DISUNTIKKAN sebagai ground truth; LLM hanya
membungkusnya jadi Bahasa Indonesia yang jelas. LLM TIDAK PERNAH menyimpulkan verdict sendiri.
"""

from app.retrieval.base import Chunk
from app.schemas import DasarHukum, MetaL2, PoinKonteks, PoinOutput

SYSTEM_PROMPT = """Anda adalah asisten reasoning untuk sistem pre-check risiko izin bangunan Kabupaten Sleman.

ATURAN WAJIB (jangan dilanggar):
1. Fakta (status, verdict, angka, kategori) pada data di bawah bersifat FINAL — hasil kode/back-end. Tugas Anda HANYA menjelaskan mengapa fakta itu muncul — JANGAN mengubah, menghitung ulang, membalik arah, atau menyimpulkan verdict/status/kategori sendiri.
2. SKOR DAMPAK INVERS — skor mentah TIDAK disuntikkan ke prompt (biar tidak membingungkan): gunakan KATEGORI_DAMPAK yang diberikan APA ADANYA ("Rendah"/"Sedang"/"Tinggi"/"Sangat Tinggi"), JANGAN menyimpulkan arah dampak dari angka skor mana pun — kalau skor invers disebut-sebut di tempat lain, ingat skor TINGGI berarti dampak RENDAH, bukan sebaliknya.
3. ITBX FALLBACK DATA-KOSONG — kalau ada baris "FALLBACK_DATA_KOSONG: True" di bawah, status yang lolos (mis. "I") BUKAN kepatuhan yang terverifikasi. WAJIB sertakan caveat persis: "diloloskan otomatis karena data matriks RDTR kosong, bukan kepatuhan terverifikasi". DILARANG menulis "kegiatan sesuai/diizinkan" tanpa caveat itu.
4. MAKNA X GANDA — kalau "STATUS_ITBX: X", makna sebenarnya (dilarang eksplisit / tidak ditemukan di matriks / di luar area RDTR) mengikuti teks "REASON" di bawah. JANGAN default ke "dilarang".
5. Kalau ada baris "CAVEAT" di bawah, WAJIB disebut/diteruskan dalam reasoning Anda — jangan disembunyikan atau diabaikan.
6. Sitasi HANYA boleh diambil dari daftar pasal yang diberikan (anchor dasar hukum back-end maupun pasal tambahan RAG), dengan menyebut citation_id persis seperti tercantum. JANGAN mengarang nomor pasal, ayat, atau dokumen yang tidak ada di daftar.
7. Kalau ada daftar kegiatan diizinkan/terbatas/bersyarat, kegiatan alternatif yang Anda sebutkan HARUS berasal dari daftar itu — JANGAN mengarang nama kegiatan lain.
8. JANGAN MENGHITUNG atau MENGARANG angka apa pun. Kamu BOLEH menyebut angka FAKTA yang diberikan di atas (usulan/ambang/target/skor) secara verbatim di reasoning_pendek, reasoning_panjang, maupun saran untuk memperjelas narasi (contoh BENAR: "KDB usulan 90% melampaui batas maksimum 60%") — SELAMA angka itu persis tercantum di fakta yang diberikan, bukan dihitung/diperkirakan/dikarang sendiri. Angka final rekomendasi (target) tetap dirakit sistem di field terpisah.
9. Tulisan ini adalah bahan decision-support untuk REVIEWER (petugas Pemda/pengambil keputusan) yang akan meng-ACC atau memberi feedback atas permohonan — BUKAN nasihat langsung ke pemohon. Sebut pemohon sebagai orang ketiga ("pemohon"/"permohonan ini"), JANGAN memakai "Anda". Tulis dalam Bahasa Indonesia yang jelas, profesional, analitis, dan dapat diaudit, agar reviewer dapat menilai dan memutuskan.

Balas HANYA dalam format JSON sesuai skema yang diberikan."""

_LABEL_ITBX = {
    "I": "Diizinkan",
    "T": "Terbatas",
    "B": "Bersyarat",
    "TB": "Terbatas & Bersyarat",
    # X sengaja TIDAK diberi label pasti (mis. "Dilarang") — makna sebenarnya WAJIB ikut REASON
    # back-end (lihat SYSTEM_PROMPT aturan #4, "makna X ganda").
    "X": "lihat REASON di bawah untuk makna sebenarnya",
}

_MAKS_KETERANGAN_KETENTUAN = 15


def _format_chunk(chunk: Chunk) -> str:
    lokasi = f"{chunk.dokumen} Pasal {chunk.pasal}" if chunk.pasal else chunk.dokumen
    if chunk.ayat:
        lokasi += f" Ayat {chunk.ayat}"
    if chunk.istilah_kode:
        lokasi += f" ({chunk.istilah_kode})"
    if chunk.halaman is not None:
        lokasi += f" (hal. {chunk.halaman})"
    return f"- citation_id={chunk.id} | {lokasi}\n  Teks: {chunk.teks}"


def _format_anchor(index: int, dasar_hukum: DasarHukum) -> str:
    lokasi = f"{dasar_hukum.dokumen} {dasar_hukum.pasal}" if dasar_hukum.pasal else dasar_hukum.dokumen
    return f"- citation_id=anchor-{index} | {lokasi}\n  Teks: {dasar_hukum.kutipan}"


def _bangun_fakta_itbx(poin: PoinKonteks) -> list[str]:
    fakta = poin.fakta
    label = _LABEL_ITBX.get(poin.status, poin.status)
    lines = [f"STATUS_ITBX: {poin.status} ({label})"]

    kbli = fakta.get("kbli_diusulkan")
    kegiatan_diusulkan = fakta.get("kegiatan_diusulkan")
    if kegiatan_diusulkan:
        lines.append(f"KEGIATAN_DIUSULKAN: {kegiatan_diusulkan}" + (f" (KBLI {kbli})" if kbli else ""))
    if fakta.get("reason"):
        lines.append(f"REASON: {fakta['reason']}")
    lines.append(f"FALLBACK_DATA_KOSONG: {fakta.get('fallback_data_kosong', False)}")

    for label_daftar, key in (
        ("Kegiatan Diizinkan", "kegiatan_diizinkan"),
        ("Kegiatan Terbatas", "kegiatan_terbatas"),
        ("Kegiatan Bersyarat", "kegiatan_bersyarat"),
        ("Kegiatan Terbatas & Bersyarat", "kegiatan_terbatas_bersyarat"),
    ):
        daftar = fakta.get(key) or []
        if daftar:
            lines.append(f"\n{label_daftar} di Zona Ini (FAKTA — pilih dari sini saja):")
            lines.extend(f"- {item}" for item in daftar)

    # keterangan_ketentuan cuma relevan (syarat) utk status T/B/TB — I/X tak butuh, & fixture nyata
    # bisa >100 item sehingga WAJIB dipotong demi kuota token.
    if poin.status in ("T", "B", "TB"):
        ketentuan = fakta.get("keterangan_ketentuan") or []
        if ketentuan:
            lines.append("\nKeterangan Ketentuan (syarat yang berlaku):")
            dipotong = ketentuan[:_MAKS_KETERANGAN_KETENTUAN]
            lines.extend(f"- {item}" for item in dipotong)
            if len(ketentuan) > _MAKS_KETERANGAN_KETENTUAN:
                lines.append(f"(dipotong dari {len(ketentuan)} item — hanya {_MAKS_KETERANGAN_KETENTUAN} pertama ditampilkan)")

    return lines


def _bangun_fakta_intensitas(poin: PoinKonteks) -> list[str]:
    fakta = poin.fakta
    lines = [f"STATUS_INTENSITAS: {poin.status}"]

    target_map = fakta.get("target") or {}
    for nama, param in (fakta.get("parameter") or {}).items():
        baris = (
            f"- {nama}: usulan={param['usulan']} {param['satuan']}, "
            f"ambang_maks={param['ambang_maks']}, ambang_min={param['ambang_min']}, "
            f"memenuhi={param['memenuhi']}"
        )
        if nama in target_map:
            baris += f"  [TARGET PATUH: {target_map[nama]}]"
        lines.append(baris)

    if fakta.get("luas_tapak_m2") is not None:
        lines.append(f"luas_tapak_m2: {fakta['luas_tapak_m2']}")
    if fakta.get("jumlah_lantai") is not None:
        lines.append(f"jumlah_lantai: {fakta['jumlah_lantai']}")
    if fakta.get("luas_rth_usulan_m2") is not None:
        lines.append(f"luas_rth_usulan_m2: {fakta['luas_rth_usulan_m2']}")

    return lines


def _bangun_fakta_dampak(poin: PoinKonteks) -> list[str]:
    # Fix #2 (temuan dampak intermiten low_confidence, diagnosis live APP-2026-6191): skor mentah
    # (IMPACT_SCORE/RUNOFF_CHANGE_INDEX) SENGAJA TIDAK disuntikkan lagi ke prompt — angka invers
    # (mis. 40, kategori Tinggi) membingungkan LLM ("40 terlihat rendah -> dampak rendah"), padahal
    # arahnya terbalik. Guardrail Cek #1 (_cek_invers_skor) menangkap ini dgn benar, tapi LLM sering
    # mengulang kesalahan yang sama di retry -> flaky fallback low_confidence. Skor/index tetap
    # muncul deterministik di ringkasan_dampak (assemble.py) — LLM tak butuh angka mentahnya utk
    # narasi per-poin, cukup KATEGORI (sudah final, tak perlu ditafsirkan arahnya).
    fakta = poin.fakta
    lines = [
        f"KATEGORI_DAMPAK: {poin.status}",
        (
            f"Kategori dampak ini FINAL — jelaskan dampak tergolong {poin.status} dan (bila Tinggi/"
            "Sangat Tinggi) perlu mitigasi. JANGAN menafsirkan atau menyebut skor angka; gunakan "
            "KATEGORI apa adanya."
        ),
    ]

    mitigasi = fakta.get("mitigasi") or {}
    if mitigasi.get("perlu_mitigasi"):
        lines.append("\nArah Mitigasi (FAKTA kualitatif, bukan angka pasti):")
        lines.extend(f"- {arah}" for arah in mitigasi.get("arah", []))

    return lines


_BANGUN_FAKTA = {
    "itbx": _bangun_fakta_itbx,
    "intensitas": _bangun_fakta_intensitas,
    "dampak": _bangun_fakta_dampak,
}


def build_user_prompt(
    poin: PoinKonteks,
    chunks: list[Chunk],
    meta: MetaL2 | None = None,
    catatan_perbaikan: str | None = None,
) -> str:
    """Susun prompt user, deterministik dari poin (fakta adapter) + chunk yang diretrieve."""
    lines: list[str] = []

    lines.append("## Poin (SUDAH FINAL — jangan diubah/dihitung ulang)")
    lines.append(f"- poin_id: {poin.poin_id}")
    lines.append(f"- kategori: {poin.kategori}")
    lines.append(f"- tipe_rekomendasi: {poin.tipe_rekomendasi}")
    lines.append("")

    lines.append("## Fakta (dihitung kode — gunakan APA ADANYA, jangan disimpulkan ulang)")
    bangun_fakta = _BANGUN_FAKTA.get(poin.poin_id)
    lines.extend(bangun_fakta(poin) if bangun_fakta else [f"STATUS: {poin.status}"])

    # data_confidence_keseluruhan SENGAJA TIDAK disuntikkan ke prompt (Fix #4) — label kepercayaan
    # adalah FAKTA, dirakit deterministik di guardrail.py::_paksa_field_wajib, bukan bahasa yang
    # diserahkan ke LLM utk echo/parafrase (sumber kebocoran token mentah "DATA_CONFIDENCE: X").
    if meta and meta.caveats:
        lines.append("")
        lines.append("## Catatan (WAJIB disebutkan dalam reasoning)")
        for caveat in meta.caveats:
            lines.append(f"CAVEAT: {caveat}")

    lines.append("")
    lines.append("## Pasal/Sitasi Tersedia (HANYA boleh kutip dari daftar ini, pakai citation_id persis)")
    ada_sitasi = False
    if poin.dasar_hukum:
        lines.append("Anchor dasar hukum dari back-end (prioritaskan ini):")
        for i, dasar_hukum in enumerate(poin.dasar_hukum):
            lines.append(_format_anchor(i, dasar_hukum))
        ada_sitasi = True
    if chunks:
        lines.append("Pasal tambahan dari RAG:")
        for chunk in chunks:
            lines.append(_format_chunk(chunk))
        ada_sitasi = True
    if not ada_sitasi:
        lines.append(
            "(Tidak ada pasal yang tersedia — jangan mengarang sitasi apa pun, kosongkan daftar sitasi.)"
        )

    if catatan_perbaikan:
        lines.append("")
        lines.append("## Catatan Perbaikan (percobaan sebelumnya gagal)")
        lines.append(catatan_perbaikan)
        lines.append("Perbaiki hal ini pada jawaban Anda kali ini.")

    lines.append("")
    lines.append(
        "## Tugas\n"
        "Jelaskan poin ini berdasarkan fakta dan pasal di atas, untuk reviewer/petugas Pemda "
        "sebagai bahan pengambilan keputusan. Berikan reasoning_pendek (1-2 kalimat), "
        "reasoning_panjang (paragraf lengkap), sitasi (rujuk citation_id di atas saja), dan saran "
        "— yaitu pertimbangan/rekomendasi yang dapat dijadikan syarat atau dasar keputusan reviewer "
        "(mis. \"pemohon perlu menyesuaikan X agar memenuhi Y\"), ditulis sebagai orang ketiga "
        "tentang pemohon, BUKAN \"Anda perlu...\"."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sintesis kesimpulan — SATU panggilan per precheck (bukan per-poin), lihat assemble.py.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_KESIMPULAN = """Anda merangkum hasil pre-check izin bangunan Kabupaten Sleman menjadi kesimpulan akhir.

ATURAN WAJIB (jangan dilanggar):
1. Ringkasan per-poin di bawah SUDAH FINAL (status, reasoning, saran) — tugas Anda HANYA merangkum jadi langkah_berdampak (daftar langkah/pertimbangan untuk REVIEWER dalam mengambil keputusan — ACC / ACC bersyarat / tolak) dan catatan_lokasi (satu kalimat kalau relevan, atau null kalau tidak ada). JANGAN menghitung ulang angka, menyimpulkan status baru, atau mengubah verdict apa pun.
2. langkah_berdampak HARUS dirangkai/diringkas dari saran per-poin yang diberikan — JANGAN menambah langkah yang tidak berdasar pada poin manapun.
3. JANGAN menyebutkan angka apa pun (skor, target, dsb) di langkah_berdampak atau catatan_lokasi.
4. Tulisan ini adalah bahan decision-support untuk REVIEWER (petugas Pemda), bukan nasihat langsung ke pemohon — sebut pemohon sebagai orang ketiga, JANGAN memakai "Anda". Tulis dalam Bahasa Indonesia yang jelas, profesional, dan ringkas.

Balas HANYA dalam format JSON sesuai skema yang diberikan."""


def build_kesimpulan_prompt(poin_list: list[PoinOutput], rekomendasi_sistem: str) -> str:
    """Susun prompt sintesis kesimpulan — HANYA dari ringkasan per-poin yang sudah lolos guardrail,
    TIDAK ada fakta mentah/angka (poin ini sudah bebas angka per SYSTEM_PROMPT aturan #8).
    """
    lines: list[str] = []
    lines.append("## Ringkasan Per-Poin (SUDAH FINAL — jangan dihitung ulang)")
    lines.append(f"rekomendasi_sistem keseluruhan: {rekomendasi_sistem}")
    lines.append("")
    for poin in poin_list:
        lines.append(f"- [{poin.poin_id}] status={poin.status}, low_confidence={poin.low_confidence}")
        lines.append(f"  reasoning: {poin.reasoning_pendek}")
        lines.append(f"  saran: {poin.rekomendasi.saran}")

    lines.append("")
    lines.append(
        "## Tugas\n"
        "Rangkum ringkasan per-poin di atas menjadi langkah_berdampak (daftar kalimat "
        "langkah/pertimbangan untuk reviewer dalam mengambil keputusan, pemohon disebut sebagai "
        "orang ketiga) dan catatan_lokasi (satu kalimat atau null)."
    )

    return "\n".join(lines)
