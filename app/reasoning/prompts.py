"""System prompt & builder pesan untuk generator.py — murni string building, TIDAK ADA panggilan LLM.

Pola fact-injection (CLAUDE.md, docs/Blueprint-Adaptasi-Model-L2-Gate-Impact.md §5) TIDAK berubah
dari model lama: fakta (status/verdict/angka/kategori) DISUNTIKKAN sebagai ground truth; LLM hanya
membungkusnya jadi Bahasa Indonesia yang jelas. LLM TIDAK PERNAH menyimpulkan verdict sendiri.
"""

import os
import re

from app import sanitize
from app.reasoning.calculator import format_target_parameter
from app.retrieval.base import Chunk
from app.schemas import DasarHukum, MetaL2, PoinKonteks, PoinOutput

SYSTEM_PROMPT_PENDEK = """Anda asisten reasoning pre-check izin bangunan Kabupaten Sleman. Tulisan Anda adalah bahan decision-support untuk REVIEWER (petugas Pemda), bukan nasihat ke pemohon.

ATURAN WAJIB:
1. Fakta (status, verdict, angka, kategori) di bawah bersifat FINAL (hasil kode/back-end). Anda HANYA menjelaskan mengapa fakta itu muncul. JANGAN mengubah, menghitung ulang, membalik arah, atau menyimpulkan verdict/status/kategori sendiri.
2. SKOR DAMPAK INVERS: skor mentah tidak disuntikkan. Pakai KATEGORI_DAMPAK apa adanya ("Rendah"/"Sedang"/"Tinggi"/"Sangat Tinggi"). Jika skor invers disebut di tempat lain, ingat skor TINGGI berarti dampak RENDAH.
3. ITBX FALLBACK DATA-KOSONG: jika ada "FALLBACK_DATA_KOSONG: True", data matriks tidak lengkap. Jika STATUS_ITBX "I", itu BUKAN kepatuhan terverifikasi: WAJIB tulis caveat persis "diloloskan otomatis karena data matriks RDTR kosong, bukan kepatuhan terverifikasi" dan DILARANG menulis "kegiatan sesuai/diizinkan" tanpa caveat itu. Jika STATUS_ITBX bukan "I" (mis. "X"), JANGAN PERNAH memakai kata "diloloskan", WAJIB tulis caveat persis "penentuan status ini didasarkan pada data matriks RDTR yang mungkin belum lengkap — perlu verifikasi manual apakah kegiatan benar-benar dilarang atau datanya belum tersedia".
4. MAKNA X GANDA: untuk "STATUS_ITBX: X", maknanya (dilarang eksplisit / tidak ditemukan di matriks / di luar area RDTR) mengikuti teks REASON. JANGAN default ke "dilarang".
5. Baris "CAVEAT" WAJIB diteruskan di reasoning.
6. Sitasi HANYA dari daftar pasal yang diberikan, dengan citation_id persis. JANGAN mengarang pasal, ayat, atau dokumen di luar daftar.
7. Kegiatan alternatif yang disebut HARUS dari daftar kegiatan yang diberikan. JANGAN mengarang nama kegiatan.
8. JANGAN MENGHITUNG atau MENGARANG angka. Angka FAKTA yang diberikan boleh disebut verbatim (contoh: "KDB usulan 90% melampaui batas maksimum 60%"), tetapi tidak boleh dihitung, diperkirakan, atau dikarang. Angka target final dirakit sistem di field terpisah.
9. Sebut pemohon sebagai orang ketiga ("pemohon"/"permohonan ini"), JANGAN memakai "Anda". Bahasa Indonesia jelas, profesional, analitis, dapat diaudit.
10a. BAHASA UNTUK PETUGAS, BUKAN BAHASA SISTEM. DILARANG menyalin nama field/label teknis ("KATEGORI_DAMPAK", "STATUS_ITBX", "FALLBACK_DATA_KOSONG", "ambang_maks") atau menjelaskan mekanisme internal sistem. Tulis substansinya dengan kata biasa ("dampak tergolong Rendah", "kegiatan termasuk kategori Bersyarat").
10. POIN LOLOS TETAP WAJIB DIJELASKAN KONKRET: jika poin lolos/memenuhi (STATUS_ITBX "I", STATUS_INTENSITAS "MEMENUHI_SYARAT", dampak "Rendah"), JANGAN menulis "tidak ada catatan berisiko". Jelaskan mengapa lolos dengan fakta relevan (kategori kegiatan di zona, angka usulan vs ambang, kategori dampak) dan sitasi yang tersedia.
11. SARAN HARUS KONKRET & DAPAT DITINDAKLANJUTI. Sebut APA yang disesuaikan dan angka targetnya persis jika ada. Jika ada lebih dari satu langkah/opsi di fakta, tulis "saran" sebagai daftar bernomor ("1. ...\\n2. ...") satu opsi per baris. Jika ada TARGET_MITIGASI_KUANTITATIF atau target numerik lain, sebut angkanya di salah satu baris saran. Untuk intensitas, jika "[TARGET PATUH: ...]" memuat angka m², angka m² itu WAJIB dikutip persis di saran, bukan hanya persentasenya.
12. FORMAT PENEKANAN: di "reasoning_panjang" tandai BOLD (diapit tanda bintang ganda **seperti ini**): (a) status/verdict poin, (b) alasan utama (satu frasa pendek), (c) kalimat tindak-lanjut terakhir. Jangan bold seluruh paragraf.
13. Tulis angka sebagai kalimat kualitatif, bukan notasi mentah: BENAR "KDH sebesar 19,6% berada di bawah ketentuan minimal 30%", SALAH "KDH (19.591212) Min (30)" atau "KDB (96.095) Max (70)".
14. SETIAP ANGKA WAJIB PAKAI SIMBOL/SATUANNYA: (%, m², m³) sesuai fakta.
15. Label huruf besar/snake_case di prompt ini adalah label INTERNAL, tulis ulang menjadi frasa biasa ("KATEGORI_DAMPAK" -> "Kategori Dampak", "ambang_maks" -> "ambang maksimal"). PENGECUALIAN: singkatan resmi regulasi (ITBX, KDB, KLB, KDH, RTH, KKOP) tetap dipakai.
16. SINGKATAN SELALU KAPITAL PENUH: (ITBX, KDB, KLB, KDH, RTH, KKOP) SELALU huruf kapital penuh.
17. citation_id (mis. "rdtr-sleman-tengah-p53-a3", "anchor-0") JANGAN PERNAH menuliskannya di reasoning_pendek/reasoning_panjang/saran. Rujuk dengan nama dokumen/pasal ("Pasal 53 RDTR Sleman Tengah"). JANGAN PERNAH memakai tanda titik koma (;): pisahkan kalimat dengan titik atau koma.
18. JANGAN membuat daftar "Referensi:" di narasi, sumber sudah ada di field "sitasi".
19. SYARAT HANYA UNTUK KEGIATAN PEMOHON: sebut syarat hanya jika tertulis di Keterangan Ketentuan untuk kegiatan pemohon atau jelas untuk kelompok kegiatan yang SAMA. DILARANG mengambil syarat kelompok lain (jam operasional, dokumen lingkungan, pengelolaan limbah, luas lantai penjualan, dll.) hanya karena ada di teks pasal yang sama. Jika tidak ada syarat untuk kegiatan pemohon di fakta, katakan syarat rinci perlu dicek petugas, jangan menebak.
20. JANGAN MENGARANG PROSEDUR: DILARANG menyarankan permohonan penyesuaian, pengecualian, dispensasi, izin khusus, atau perubahan zona, kecuali tertulis di fakta/sitasi.
21. ITBX HANYA SOAL KLASIFIKASI KEGIATAN: DILARANG membahas KDB, KLB, KDH, atau ambang intensitas di poin ITBX.
22. SYARAT BERSYARAT BELUM TERBUKTI TERPENUHI: untuk status B/T/TB, syarat adalah kewajiban yang HARUS dipenuhi dan data TIDAK memuat bukti syarat sudah terpenuhi. DILARANG menulis 'sudah/telah memenuhi', 'memenuhi syarat luas', atau 'diloloskan otomatis'. Keputusan ada pada petugas.
23. JANGAN MENAMBAH KEWAJIBAN ADMINISTRATIF: DILARANG menambah kewajiban dokumen, perhitungan, atau berkas, dan DILARANG menyuruh 'menambah blok'. Saran hanya mengulang syarat yang tertulis di fakta.
24. HANYA YANG ADA DI FAKTA: DILARANG membuat klaim sebab-akibat atau teknis yang tidak tertulis di fakta (mis. koefisien limpasan berbeda antar jenis bangunan, perilaku tanah), DILARANG menyebut Pasal 53 ayat (3) huruf d sebagai dasar kewajiban bangunan baru (huruf d untuk lahan terbangun yang sudah ada, kewajiban bangunan baru adalah zero delta Q pada huruf a), dan di poin dampak DILARANG menyatakan pemohon sudah memenuhi syarat kawasan resapan. Jangan menulis istilah dalam kurung seperti "(proporsi tapak/bangunan)".

Balas HANYA dalam format JSON sesuai skema yang diberikan."""

# Sakelar uji A/B (PROMPT_SISTEM=panjang|pendek, bawaan pendek). Dibaca saat modul dimuat.
if os.getenv("PROMPT_SISTEM", "pendek").strip().lower() == "panjang":
    from app.reasoning.prompt_panjang import SYSTEM_PROMPT_PANJANG as SYSTEM_PROMPT
else:
    SYSTEM_PROMPT = SYSTEM_PROMPT_PENDEK

_LABEL_ITBX = {
    "I": "Diizinkan",
    "T": "Terbatas",
    "B": "Bersyarat",
    "TB": "Terbatas & Bersyarat",
    # X sengaja TIDAK diberi label pasti (mis. "Dilarang") — makna sebenarnya WAJIB ikut REASON
    # back-end (lihat SYSTEM_PROMPT aturan #4, "makna X ganda").
    "X": "lihat REASON di bawah untuk makna sebenarnya",
}

# APP-2026-8090: BE mulai kirim keterangan_ketentuan LENGKAP (bukan dipotong sepihak spt sebelumnya
# — lihat commit sebelumnya soal "(disembunyikan)"), tapi utk zona Bersyarat besar bisa >150 item
# (satu dokumen zona memuat ketentuan SEMUA kelompok KBLI + overlay KKOP/LP2B/bencana/sempadan
# sekaligus, bukan cuma yg relevan ke kegiatan pemohon). Naikkan dari 15 (headroom konteks
# gpt-oss-20b 131K token, jauh dari mepet) TAPI kenaikan jumlah SENDIRIAN tidak menyelesaikan
# masalah inti: tanpa pengurutan relevansi, potongan tetap ambil item PALING AWAL (selalu kelompok
# "a. pertanian/kehutanan/perikanan" dst, apapun kegiatan pemohon) — lihat _urutkan_relevansi_ketentuan.
_MAKS_KETERANGAN_KETENTUAN = 20

# APP-2026-9461 (413 Request too large, live): BE mengirim `dasar_hukum` bisa puluhan item
# (kutipan utuh tiap pasal) -> prompt satu poin >8-11K token -> Groq 413/429 padahal kuota TPM
# cuma 8K. Anchor dipotong ke N PALING AWAL (urutan BE = prioritas hukum yg dipakai pengambilan
# keputusan) + catatan eksplisit; LLM tetap dilarang mengarang sitasi di luar daftar yg tampil
# (SYSTEM_PROMPT aturan #6). citation_id (anchor-{i}) tetap konsisten dgn generator.py karena
# indeks asli dipertahankan.
_MAKS_ANCHOR = 12

_RE_KATA = re.compile(r"[a-zA-Z]{4,}")

# Kata umum Bahasa Indonesia yg TIDAK informatif utk overlap relevansi (stopword kasar, bukan
# kamus lengkap) — tanpa ini kata spt "yang"/"dengan"/"untuk" mendominasi skor & menenggelamkan
# kata kunci kegiatan yg sebenarnya (mis. "pertanian", "reparasi", "mobil").
_STOPWORD_RELEVANSI = frozenset({
    "yang", "dengan", "untuk", "pada", "dari", "atau", "dan", "secara", "dapat", "tidak", "adalah",
    "wajib", "memenuhi", "ketentuan", "kegiatan", "kelompok", "sebagai", "berikut", "diperbolehkan",
    "sesuai", "peraturan", "perundang", "undangan", "sebagaimana", "dimaksud", "meliputi", "serta",
})


def _kata_kunci(teks: str) -> set[str]:
    return {w.lower() for w in _RE_KATA.findall(teks)} - _STOPWORD_RELEVANSI


def _urutkan_relevansi_ketentuan(ketentuan: list[str], kegiatan_diusulkan: str | None) -> list[str]:
    """Urutkan (stable sort, BUKAN filter/buang) item keterangan_ketentuan berdasar overlap kata
    kunci dgn kegiatan_diusulkan pemohon — supaya kalau daftar terlalu panjang & harus dipotong,
    yang terpotong adalah item PALING TAK RELEVAN, bukan sekadar item paling akhir di daftar asli.

    Ditemukan live (APP-2026-8090, zona Bersyarat >150 item): potongan 15-teratas versi lama SELALU
    kelompok "a. pertanian/kehutanan/perikanan" (urutan pertama di dokumen), apapun kegiatan yg
    diusulkan pemohon — kalau kegiatan sebenarnya ada di kelompok lain (mis. "f. konstruksi" atau
    "n. jasa lainnya"), syarat yg BENAR-BENAR relevan tidak pernah sampai ke LLM sama sekali.

    Heuristik overlap kata kunci teks, BUKAN tabel kode KBLI resmi (RDTR Sleman tak menyediakan
    pemetaan digit KBLI -> nama kelompok yg bisa diverifikasi di sini) — sengaja demikian: risiko
    tabel kode yg salah/kadaluarsa lebih berbahaya (silent wrong-filter) drpd heuristik teks yg
    predictable & mudah diaudit manusia dari `kegiatan_diusulkan` fakta itu sendiri.
    """
    if not kegiatan_diusulkan:
        return ketentuan
    kata_kegiatan = _kata_kunci(kegiatan_diusulkan)
    if not kata_kegiatan:
        return ketentuan

    def _skor(item: str) -> int:
        return len(kata_kegiatan & _kata_kunci(item))

    # sorted() Python stable -> item skor sama tetap urutan asli relatif satu sama lain.
    return sorted(ketentuan, key=_skor, reverse=True)


def _format_chunk(chunk: Chunk, konteks_induk: str | None = None) -> str:
    lokasi = f"{chunk.dokumen} Pasal {chunk.pasal}" if chunk.pasal else chunk.dokumen
    if chunk.ayat:
        lokasi += f" Ayat {chunk.ayat}"
    if chunk.istilah_kode:
        lokasi += f" ({chunk.istilah_kode})"
    if chunk.halaman is not None:
        lokasi += f" (hal. {chunk.halaman})"
    baris = f"- citation_id={chunk.id} | {lokasi}\n  Teks: {chunk.teks}"
    if konteks_induk:
        # Small-to-big: teks pasal induk ditempel sbg KONTEKS, BUKAN kandidat sitasi baru. 62%
        # chunk ayat memuat "sebagaimana dimaksud pada ayat (N)" — tanpa induknya, LLM membaca
        # rujukan ke teks yang tak pernah ia lihat. Sitasi TETAP memakai citation_id ayat di
        # atas: id pasal induk tidak masuk daftar kandidat, jadi menyitasinya akan ditolak
        # guardrail dan membuang retry — karena itu larangannya ditulis eksplisit ke LLM.
        baris += (
            "\n  Konteks pasal induk (untuk memahami rujukan antar-ayat — JANGAN "
            "disitasi, tetap kutip citation_id ayat di atas): "
        ) + konteks_induk
    return baris


def _format_anchor(index: int, dasar_hukum: DasarHukum) -> str:
    lokasi = f"{dasar_hukum.dokumen} {dasar_hukum.pasal}" if dasar_hukum.pasal else dasar_hukum.dokumen
    return f"- citation_id=anchor-{index} | {lokasi}\n  Teks: {dasar_hukum.kutipan}"


def _bangun_fakta_itbx(poin: PoinKonteks) -> list[str]:
    # Allowlist eksplisit (app/sanitize.py) — key `poin.fakta` yang tak terdaftar utk poin_id ini
    # DIBUANG sebelum sampai ke prompt LLM; key free-text (mis. reason/keterangan_ketentuan) di-scrub
    # pola PII. Pagar ini, bukan asumsi "fakta pasti bersih", yang mencegah field baru bocor nanti.
    fakta = sanitize.sanitize_fakta(poin.poin_id, poin.fakta)
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

    if poin.status in ("T", "B", "TB"):
        lines.append(
            "\nPEDOMAN_PENJELASAN (WAJIB untuk status ini): tulis reasoning_panjang dalam 3 bagian pendek. "
            "(1) Sebut status kegiatan dan zonanya. (2) Sebut syarat atau batasan yang berlaku HANYA dari "
            "Keterangan Ketentuan di bawah, dikutip apa adanya. (3) Tutup dengan kalimat bahwa pemohon wajib "
            "memenuhi syarat itu, pemenuhannya belum terbukti dari data, dan penilaian akhir ada pada petugas. "
            "DILARANG: menulis syarat 'sudah/telah/memenuhi syarat', menyebut kegiatan 'lolos otomatis', "
            "menambah kewajiban dokumen/berkas/perhitungan/bukti, menyebut syarat kelompok kegiatan lain "
            "(limbah, jam operasional, dokumen lingkungan) yang tidak tertulis di Keterangan Ketentuan, atau "
            "membahas KDB/KLB/KDH."
        )

    # keterangan_ketentuan cuma relevan (syarat) utk status T/B/TB — I/X tak butuh, & fixture nyata
    # bisa >150 item (satu dokumen zona memuat SEMUA kelompok KBLI + overlay sekaligus) sehingga
    # WAJIB dipotong demi kuota token — tapi diurutkan relevansi dulu (lihat
    # _urutkan_relevansi_ketentuan), BUKAN dipotong dari urutan asli dokumen apa adanya.
    if poin.status in ("T", "B", "TB"):
        ketentuan = fakta.get("keterangan_ketentuan") or []
        if ketentuan:
            terurut = _urutkan_relevansi_ketentuan(ketentuan, kegiatan_diusulkan)
            lines.append("\nKeterangan Ketentuan (syarat yang berlaku, diurutkan berdasar relevansi ke kegiatan pemohon):")
            dipotong = terurut[:_MAKS_KETERANGAN_KETENTUAN]
            lines.extend(f"- {item}" for item in dipotong)
            lines.append(
                "Catatan: field saran untuk poin ini ditetapkan sistem dari syarat di atas. Isi saran dengan "
                "satu kalimat pendek saja, lalu fokuskan reasoning pada alasan status dan syarat yang berlaku."
            )
            if len(terurut) > _MAKS_KETERANGAN_KETENTUAN:
                lines.append(
                    f"(dipotong dari {len(terurut)} item total — {_MAKS_KETERANGAN_KETENTUAN} item "
                    "PALING RELEVAN ke kegiatan pemohon ditampilkan, bukan sekadar urutan pertama "
                    "di dokumen)"
                )

    return lines


def _angka_prompt(nilai: object) -> str:
    """Format angka utk prompt: bulatkan float berekor panjang ke maks 2 desimal.

    Ditemukan live (replay 2026-09-08, APP-2026-3468): `usulan` KDB dikirim BE sebagai
    47.05882352941176 dan disuntikkan APA ADANYA ke prompt, lalu LLM menyalinnya utuh ke narasi
    ("KDB pemohon sebesar 47.05882352941176%") — presisi palsu yang tak berarti apa pun bagi petugas.
    Dibulatkan DI PROMPT saja; fakta/kalkulator tetap presisi penuh, dan guardrail sudah menerima
    versi bulat lewat `_dekat_dgn_pembulatan` (APP-2026-8376) sehingga cek provenance tidak terganggu.
    Nilai yang pembulatannya akan menghilangkan angka (mis. 0.004 -> 0.0) dibiarkan apa adanya.
    """
    if isinstance(nilai, bool) or not isinstance(nilai, float):
        return str(nilai)
    dibulatkan = round(nilai, 2)
    if nilai == int(nilai):
        return str(int(nilai))  # 800.0 -> "800" (bukan "800.0")
    if dibulatkan == nilai or dibulatkan == 0:
        return str(nilai)
    return f"{dibulatkan:.2f}".rstrip("0").rstrip(".")


def _baris_zona_intensitas(poin: PoinKonteks, ada_parameter: bool) -> list[str]:
    """Nama zona/sub-zona pemohon + kewajiban menyebutnya, khusus blok fakta intensitas.

    Kenapa ini ada. Ambang KDB/KLB/KDH bukan angka umum: ia dibaca dari tabel Lampiran VI PER
    ZONA. Narasi yang tak menyebut zona pemohon karena itu tidak menjelaskan dari mana ambangnya
    datang, dan pembaca tak punya cara memeriksa apakah ambang yang dipakai memang milik zonanya.

    Terukur, bukan dugaan gaya bahasa: jangkar `zona_pemohon` di `eval/metrik_generasi.py::
    kekhususan` hanya kena 30% (3/10) di poin intensitas, vs 100% (11/11) di itbx. Sebabnya
    ditelusuri ke prompt, bukan ke model — pada SELURUH 10 narasi intensitas di log, nama zona
    maupun kode sub-zona pemohon TIDAK PERNAH muncul di prompt poin itu (`poin.zona` ada di
    PoinKonteks tapi tak pernah dibaca di sini). Di itbx ia ikut hanya karena menumpang free-text
    `reason` back-end ("Lolos karena kegiatan Diizinkan (I) di zona Zona Pertanian", APP-2026-2428);
    `fakta` intensitas tak punya field setara. Tiga narasi intensitas yang kebetulan kena pun
    mencocoki KODE sub-zona saja ("P-1", APP-2026-2428/-8090/-INNER-01) dan tak pernah nama zona
    induknya — satu-satunya sumber kode itu di prompt lama adalah teks chunk RAG, bukan fakta
    permohonannya, jadi kebetulan pun bergantung pada chunk apa yang kebetulan terambil.

    Labelnya sengaja frasa Bahasa Indonesia biasa, BUKAN label kapital/snake_case seperti baris
    fakta lain di blok ini: aturan #10a/#15 SYSTEM_PROMPT melarang label teknis disalin ke narasi,
    dan di sini nilai yang WAJIB disalin justru ada di barisnya sendiri — label yang menyerupai
    nama variabel akan menarik model melanggar aturan itu sambil mematuhi yang ini.

    Tak ada angka baru yang masuk narasi lewat jalan ini, jadi aturan #8 (dan gerbang
    `guardrail._angka_terlacak_ke_sumber`) tak tersentuh: seluruh kode sub-zona di korpus RDTR
    Sleman Tengah berakhiran satu digit (R-2, P-1, SPU-3, RTH-7, …) sementara cek provenance hanya
    memeriksa angka >=2 digit (`_RE_ANGKA_MENCURIGAKAN`).
    """
    if not (poin.zona or poin.zona_subzone):
        return []
    if poin.status == "Tidak Dinilai" and not poin.zona:
        # Tanpa nama zona induk, satu-satunya yang bisa disebut adalah kode sub-zona — dan
        # menyuruh model menyebutnya saat gate berhenti sebelum intensitas dinilai berarti
        # menuntut presisi atas ketentuan yang tak pernah dikutip sama sekali.
        return []

    if poin.zona_subzone:
        baris = f"Zona pemohon: {poin.zona}, sub-zona {poin.zona_subzone}" if poin.zona else \
            f"Zona pemohon: sub-zona {poin.zona_subzone} (nama zona induk tidak dikirim back-end)"
    else:
        baris = f"Zona pemohon: {poin.zona} (sub-zona presisi TIDAK dikonfirmasi back-end)"
    lines = [baris]

    asal = (
        "dari mana ambang KDB/KLB/KDH di bawah berasal — ambang itu ketentuan PER ZONA, bukan "
        "angka umum yang berlaku di seluruh wilayah"
        if ada_parameter else
        "poin ini — ketentuan intensitas berbeda antar zona, jadi pembaca perlu tahu zona mana "
        "yang ketentuannya belum diperiksa"
    )
    wajib = f"WAJIB: sebut nama zona pemohon di atas apa adanya saat menjelaskan {asal}."
    if not poin.zona_subzone:
        # Senada CAVEAT_SUBZONA_TAK_TERKONFIRMASI (cabang mayoritas: 96.4% permohonan, 425 dari
        # 441, tak mengirim rdtr_subzone — lihat app/reasoning/generator.py): yang disebut WAJIB
        # zona INDUK. Kewajiban baru ini tidak boleh berubah jadi tekanan mengarang kode sub-zona
        # presisi — justru kelas bug yang caveat itu ada untuk mencegah. Larangannya hanya
        # ditempel di cabang ini: kalau sub-zonanya memang dikonfirmasi, tak ada yang perlu
        # ditebak dan kalimat tambahan cuma membakar token pada prompt yang TPM-nya sudah mepet.
        wajib += (
            " Sub-zona presisinya TIDAK diketahui, jadi sebut nama zona INDUK saja — DILARANG "
            "menyebut, menebak, atau mengarang kode sub-zona (pola huruf-angka) apa pun."
        )
    lines.append(wajib)
    return lines


def _bangun_fakta_intensitas(poin: PoinKonteks) -> list[str]:
    # Allowlist eksplisit (app/sanitize.py) — key `poin.fakta` yang tak terdaftar utk poin_id ini
    # DIBUANG sebelum sampai ke prompt LLM; key free-text (mis. reason/keterangan_ketentuan) di-scrub
    # pola PII. Pagar ini, bukan asumsi "fakta pasti bersih", yang mencegah field baru bocor nanti.
    fakta = sanitize.sanitize_fakta(poin.poin_id, poin.fakta)
    lines = [f"STATUS_INTENSITAS: {poin.status}"]
    lines.extend(_baris_zona_intensitas(poin, bool(fakta.get("parameter"))))

    dasar = fakta.get("dasar_ambang") or {}
    if dasar:
        bagian = []
        if dasar.get("nama_zona"):
            bagian.append(f"zona {dasar['nama_zona']}" + (f" ({dasar['kode_zona']})" if dasar.get("kode_zona") else ""))
        if dasar.get("hierarki_jalan"):
            bagian.append(f"hierarki jalan {dasar['hierarki_jalan']}")
        if dasar.get("kawasan_resapan") is True:
            bagian.append("lokasi di kawasan resapan air")
        elif dasar.get("kawasan_resapan") is False:
            bagian.append("lokasi bukan kawasan resapan air")
        if bagian:
            lines.append(
                "Dasar pemilihan ambang (dari back-end): " + ", ".join(bagian) + ". Saat menjelaskan "
                "dari mana ambang berasal, sebut dasar ini apa adanya."
            )

    target_map = fakta.get("target") or {}
    for nama, param in (fakta.get("parameter") or {}).items():
        # Satuan disertakan di SETIAP angka (usulan DAN ambang) — sebelumnya cuma "usulan" yang
        # dapat satuan, "ambang_maks"/"ambang_min" tampil telanjang (mis. "ambang_maks=70" tanpa
        # "%"), memicu LLM ikut menyalin tanpa satuan ke narasi (item permintaan user 2026-09-21:
        # "19 -> 19%"). "rasio" (KLB) sengaja TIDAK dapat sufiks — bukan persentase.
        satuan_sufiks = "%" if param["satuan"] == "persen" else ""
        baris = (
            f"- {nama}: usulan={_angka_prompt(param['usulan'])}{satuan_sufiks}, "
            f"ambang_maks={_angka_prompt(param['ambang_maks'])}{satuan_sufiks if param['ambang_maks'] is not None else ''}, "
            f"ambang_min={_angka_prompt(param['ambang_min'])}{satuan_sufiks if param['ambang_min'] is not None else ''}, "
            f"memenuhi={param['memenuhi']}"
        )
        if nama in target_map:
            baris += f"  [TARGET PATUH: {format_target_parameter(target_map[nama])}]"
        lines.append(baris)

    # Usulan vs batas ditulis sebagai dua kalimat terpisah (APP-2026-UJI-4: LLM menulis "KDB 83,57%
    # setara dengan kebutuhan lantai dasar maksimal 861,6 m2", menyamakan persentase USULAN dengan luas
    # BATAS).
    info_kdb, info_kdh = target_map.get("kdb") or {}, target_map.get("kdh") or {}
    param_kdb, param_kdh = (fakta.get("parameter") or {}).get("kdb") or {}, (fakta.get("parameter") or {}).get("kdh") or {}
    if "footprint_maks_m2" in info_kdb and fakta.get("luas_tapak_m2") is not None:
        lines.append(
            f"Luas lantai dasar USULAN pemohon: {_angka_prompt(fakta['luas_tapak_m2'])} m2 "
            f"(KDB {_angka_prompt(param_kdb.get('usulan'))}%). Luas lantai dasar MAKSIMAL yang diperbolehkan: "
            f"{_angka_prompt(info_kdb['footprint_maks_m2'])} m2 (KDB {_angka_prompt(info_kdb.get('target_kdb'))}%)."
        )
    if "rth_dibutuhkan_m2" in info_kdh and fakta.get("luas_rth_usulan_m2") is not None:
        lines.append(
            f"RTH USULAN pemohon: {_angka_prompt(fakta['luas_rth_usulan_m2'])} m2 "
            f"(KDH {_angka_prompt(param_kdh.get('usulan'))}%). RTH MINIMAL yang dibutuhkan: "
            f"{_angka_prompt(info_kdh['rth_dibutuhkan_m2'])} m2 (KDH {_angka_prompt(info_kdh.get('target_kdh'))}%)."
        )
    if info_kdb or info_kdh:
        lines.append(
            "Usulan dan batas adalah dua hal berbeda: JANGAN menulis persentase usulan 'setara' dengan luas "
            "batas, dan JANGAN menukar keduanya."
        )

    if fakta.get("luas_tapak_m2") is not None:
        lines.append(f"luas_tapak_m2: {_angka_prompt(fakta['luas_tapak_m2'])}")
    if fakta.get("jumlah_lantai") is not None:
        lines.append(f"jumlah_lantai: {fakta['jumlah_lantai']}")
    if fakta.get("luas_rth_usulan_m2") is not None:
        lines.append(f"luas_rth_usulan_m2: {_angka_prompt(fakta['luas_rth_usulan_m2'])}")

    return lines


def _bangun_fakta_dampak(poin: PoinKonteks) -> list[str]:
    # Fix #2 (temuan dampak intermiten low_confidence, diagnosis live APP-2026-6191): skor mentah
    # (IMPACT_SCORE/RUNOFF_CHANGE_INDEX) SENGAJA TIDAK disuntikkan lagi ke prompt — angka invers
    # (mis. 40, kategori Tinggi) membingungkan LLM ("40 terlihat rendah -> dampak rendah"), padahal
    # arahnya terbalik. Guardrail Cek #1 (_cek_invers_skor) menangkap ini dgn benar, tapi LLM sering
    # mengulang kesalahan yang sama di retry -> flaky fallback low_confidence. Skor/index tetap
    # muncul deterministik di ringkasan_dampak (assemble.py) — LLM tak butuh angka mentahnya utk
    # narasi per-poin, cukup KATEGORI (sudah final, tak perlu ditafsirkan arahnya).
    # Allowlist eksplisit (app/sanitize.py) — key `poin.fakta` yang tak terdaftar utk poin_id ini
    # DIBUANG sebelum sampai ke prompt LLM; key free-text (mis. reason/keterangan_ketentuan) di-scrub
    # pola PII. Pagar ini, bukan asumsi "fakta pasti bersih", yang mencegah field baru bocor nanti.
    fakta = sanitize.sanitize_fakta(poin.poin_id, poin.fakta)
    lines = [
        f"KATEGORI_DAMPAK: {poin.status}",
        (
            f"Kategori dampak ini FINAL — jelaskan dampak tergolong {poin.status} dan (bila Sedang/"
            "Tinggi/Sangat Tinggi) perlu mitigasi. JANGAN menafsirkan atau menyebut skor angka; gunakan "
            "KATEGORI apa adanya."
        ),
    ]

    # APP-2026-2428: dinilai=False sekarang punya penyebab BARU selain "gate berhenti sebelum
    # dampak dievaluasi" — poligon pemohon bersinggungan dgn >1 bidang persil, back-end SENGAJA tak
    # menghitung skor hidrologi (bukan kegagalan sistem). `limitations` bawa alasan itu apa adanya
    # dari back-end — WAJIB disuntikkan di sini supaya LLM tahu alasan SEBENARNYA (bukan menebak/
    # mengarang "gate berhenti"), dan supaya reasoning_pendek/panjang konsisten dgn `saran` yang
    # sudah ditemplate deterministik di generator.py::_saran_tidak_dinilai (sumber sama).
    if poin.status == "Tidak Dinilai" and fakta.get("limitations"):
        lines.append(
            f"\nALASAN_TIDAK_DINILAI: {fakta['limitations']}\n"
            "Ini SEBAB SEBENARNYA dampak tak dinilai — JANGAN mengarang alasan lain (mis. 'gate "
            "berhenti di tahap sebelumnya') kalau bukan ini alasannya. JANGAN merekomendasikan "
            "solusi teknis drainase/mitigasi lingkungan apa pun di sini — itu tak relevan selama "
            "penyebab di atas belum diselesaikan (mis. kalau soal batas persil, solusinya "
            "administratif/spasial, bukan teknis lingkungan)."
        )

    # APP-2026-2428: usulan tapak+RTH melebihi luas persil tercatat — dampak TETAP dinilai (skor
    # tetap dihitung back-end), tapi akurasinya berkurang. WAJIB tampil sbg peringatan eksplisit
    # (deterministik JUGA ditambahkan ke disclaimer di guardrail.py — dua lapis, bukan
    # mengandalkan LLM ingat menyebutnya di sini saja).
    if fakta.get("luas_usulan_melebihi_persil"):
        lines.append(
            "\nPERINGATAN_LUAS_PERSIL: usulan luas tapak bangunan + RTH pemohon MELEBIHI luas "
            "bidang persil yang tercatat di sistem. Sebutkan ini sbg catatan di reasoning & saran — "
            "sarankan pemohon meninjau ulang proporsi KDB/KDH agar sesuai batas fisik persil."
        )

    tipe = fakta.get("tipe_bangunan") or {}
    if tipe.get("kegiatan") or tipe.get("kelas_atap"):
        bagian = []
        if tipe.get("kegiatan"):
            bagian.append(f"kegiatan yang diusulkan: {tipe['kegiatan']}"
                          + (f" (KBLI {tipe['kbli']})" if tipe.get("kbli") else ""))
        if tipe.get("kelas_atap"):
            bagian.append(f"kelas bangunan pada perhitungan limpasan: {tipe['kelas_atap']}")
        if tipe.get("jumlah_lantai"):
            bagian.append(f"{tipe['jumlah_lantai']} lantai")
        lines.append(
            "\nTIPE_BANGUNAN (dari back-end): " + ", ".join(bagian) + ". WAJIB sebut jenis kegiatan atau "
            "bangunan ini di reasoning_pendek dan reasoning_panjang. Di reasoning_panjang cukup SATU kalimat "
            "tentang pengaruhnya, persis berbunyi: 'Jenis bangunan memengaruhi koefisien limpasan atap pada "
            "perhitungan dampak.' DILARANG membandingkan koefisien dengan bangunan lain atau menyebutnya "
            "'lebih tinggi' atau 'berbeda'. Jangan menyebut angka koefisien dan jangan mengarang jenis "
            "bangunan lain."
        )

    kawasan = fakta.get("kawasan_hidrologi") or {}
    if kawasan.get("nama"):
        lines.append(
            f"\nKawasan yang dinilai pada perhitungan hidrologi (dari back-end): {kawasan['nama']}"
            + (f" (kode {kawasan['kode']})" if kawasan.get("kode") else "")
            + "."
        )
    lines.append(
        "\nSEBAB_KATEGORI_DAMPAK: kategori dampak ditentukan oleh indeks limpasan air, yaitu akibat luas "
        "bangunan dan RTH yang diusulkan, BUKAN oleh lokasi. Status kawasan resapan air hanya menentukan "
        "ketentuan yang berlaku. DILARANG menulis bahwa dampak 'disebabkan oleh' atau 'karena' lokasi "
        "berada di kawasan resapan air."
    )
    if poin.kawasan_resapan is True:
        lines.append(
            "\nKAWASAN_RESAPAN_AIR: YA. Lokasi berada di kawasan resapan air, jadi ketentuan khusus "
            "kawasan resapan air berlaku. Untuk PENGEMBANGAN BARU (permohonan ini) kewajibannya adalah "
            "memastikan air tidak melimpas (zero delta Q), tidak merusak fungsi resapan, dan tidak "
            "mencemari air (Pasal 53 ayat 3 huruf a sampai c). Sumur atau kolam resapan adalah SARANA "
            "memenuhi zero delta Q itu. Teks Pasal 53 ayat 3 huruf d sendiri mewajibkan sumur atau kolam "
            "pada LAHAN TERBANGUN YANG SUDAH ADA, jadi JANGAN menyebut huruf d sebagai dasar kewajiban "
            "bangunan baru. Untuk dampak Tinggi, tulis sumur atau kolam resapan sebagai kewajiban untuk "
            "memenuhi zero delta Q (ketentuan khusus kawasan resapan air), dan menurunkan KDB serta "
            "menaikkan KDH/RTH sebagai kewajiban. Jelaskan ketentuan ini di reasoning dan kaitkan dengan "
            "saran, rujuk pasalnya dengan bahasa manusia."
        )
    elif poin.kawasan_resapan is False:
        lines.append(
            "\nKAWASAN_RESAPAN_AIR: TIDAK. Lokasi BUKAN kawasan resapan air. DILARANG menerapkan "
            "ketentuan khusus kawasan resapan air (zero delta Q, Pasal 53) sebagai ketentuan yang "
            "berlaku dan DILARANG menyatakan lokasi berada di kawasan atau zona resapan air. Anda BOLEH "
            "menulis SATU kalimat bahwa ketentuan khusus kawasan resapan air (Pasal 53) tidak berlaku "
            "karena lokasi di luar kawasan resapan. Pada dampak Rendah, sumur atau kolam "
            "resapan BOLEH dianjurkan sebagai praktik baik. Pada dampak Sedang, Tinggi, atau Sangat Tinggi, JANGAN "
            "menyebut sumur atau kolam resapan sama sekali: mitigasinya adalah kewajiban menurunkan KDB "
            "dan menaikkan KDH/RTH. Bila sumur disebut, sebutkan sebagai rekomendasi, BUKAN kewajiban dari "
            "ketentuan kawasan resapan air. JANGAN "
            "memakai kata 'mengharuskan', 'wajib', atau 'menuntut' untuk sumur atau kolam resapan, dan "
            "JANGAN menyebut ketentuan RDTR mewajibkannya. Jangan menulis bahwa lokasi 'tidak dapat "
            "memanfaatkan' Pasal 53 (pasal itu bukan manfaat, ia hanya tidak berlaku di sini)."
        )
    else:
        lines.append(
            "\nKAWASAN_RESAPAN_AIR: TIDAK DIKETAHUI. JANGAN menyatakan lokasi berada atau tidak "
            "berada di kawasan resapan air, dan jangan menyebut ketentuan khusus kawasan resapan "
            "sebagai berlaku."
        )

    lines.append(
        "\nSTRUKTUR_REASONING (WAJIB): reasoning_pendek 1 kalimat. reasoning_panjang MAKSIMAL 4 kalimat tanpa judul "
        "bold berlabel (jangan menulis '**Kategori Dampak**:' atau sejenisnya): (1) kategori dampak dan sebabnya "
        "(indeks limpasan dipengaruhi luas bangunan, RTH, dan jenis bangunan), (2) ketentuan kawasan resapan bila "
        "berlaku, (3) tindak lanjut utama. Jangan mengulang seluruh daftar saran di reasoning."
    )

    mitigasi = fakta.get("mitigasi") or {}
    if mitigasi.get("perlu_mitigasi"):
        lines.append("\nArah Mitigasi (FAKTA kualitatif, bukan angka pasti):")
        lines.extend(f"- {arah}" for arah in mitigasi.get("arah", []))
    elif poin.status != "Tidak Dinilai" and mitigasi.get("arah"):
        # Dampak Rendah/Sedang (permintaan user 2026-10-14): tak ada mitigasi KOREKTIF, tapi saran
        # TETAP wajib berisi langkah pencegahan/pemeliharaan — bukan "tidak diperlukan tindakan".
        pemeliharaan = mitigasi.get("pemeliharaan") or {}
        butir = []
        if pemeliharaan.get("luas_rth_m2") is not None:
            butir.append(f"pertahankan luas RTH minimal {_angka_prompt(pemeliharaan['luas_rth_m2'])} m² sesuai usulan")
        if pemeliharaan.get("luas_bangunan_m2") is not None:
            butir.append(f"jaga luas bangunan tidak melebihi {_angka_prompt(pemeliharaan['luas_bangunan_m2'])} m² sesuai usulan")
        if poin.kawasan_resapan is True:
            butir.append("pastikan air hujan tidak melimpas (zero delta Q), sediakan sumur atau kolam resapan untuk mendukungnya, dan rawat saluran drainase")
        elif poin.kawasan_resapan is False:
            butir.append("sediakan sumur atau kolam resapan air hujan dan rawat saluran drainase sebagai rekomendasi praktik baik, bukan kewajiban kawasan resapan")
        else:
            butir.append("sediakan sumur atau kolam resapan air hujan dan rawat saluran drainase sebagai praktik baik")
        lines.append(
            f"\nSARAN_PEMELIHARAAN (WAJIB): dampak tergolong {poin.status}, jadi TIDAK perlu mitigasi "
            "korektif. Namun saran WAJIB tetap berisi langkah pencegahan dan pemeliharaan yang konkret. "
            "DILARANG menulis 'tidak diperlukan tindakan khusus', 'tidak perlu tindakan', atau kalimat "
            "senada di saran. Muat langkah berikut di saran: " + ", ".join(butir) + ". "
            "Sebutkan angka luas di atas APA ADANYA, JANGAN mengarang atau menghitung angka lain."
        )

    # Target kuantitatif (ambang runoff_change_index, dari app/reasoning/calculator.py) — BEDA dgn
    # skor invers yang dilarang Aturan #2 di atas (skor TINGGI = dampak RENDAH): angka di sini arahnya
    # LURUS (index turun = dampak membaik), sudah diinterpretasikan penuh di sini, LLM tinggal kutip.
    target_mitigasi = fakta.get("target_mitigasi") or {}
    ambang = target_mitigasi.get("runoff_change_index_maks")
    if ambang is not None:
        lines.append(
            f"\nTARGET_MITIGASI_KUANTITATIF: indikator limpasan air (runoff) perlu ditekan hingga "
            f"DI BAWAH {_angka_prompt(ambang)} (saat ini {_angka_prompt(target_mitigasi['index_saat_ini'])}) "
            f"supaya kategori dampak turun dari {poin.status} ke {target_mitigasi['kategori_target']}. "
            "Angka ini BUKAN skor invers (beda dari Aturan #2) — arah SELALU 'turunkan sampai di bawah "
            "angka ini', jangan ditafsirkan arah lain. WAJIB sebutkan angka target ini secara eksplisit "
            "& konkret di reasoning_pendek dan saran (kutip apa adanya, JANGAN dihitung ulang)."
        )

    # APP-2026-8025/-5067: BE kini (kadang) kirim rincian penyesuaian lahan KONKRET (bukan cuma
    # ambang index abstrak di atas) — luas bangunan/RTH dalam m², persen KDB/KDH target, dimensi
    # sumur/kolam resapan. `saran` final poin ini AKAN ditimpa verbatim dgn narasi BE sendiri
    # (generator.py::_saran_mitigasi_dampak, tak bergantung LLM) — fakta ini disuntikkan HANYA
    # supaya reasoning_pendek/panjang LLM (yg TETAP murni dari LLM, tak ditimpa) konsisten & tak
    # menyebut angka lain yang kontradiktif dgn saran final.
    penyesuaian = target_mitigasi.get("penyesuaian_lahan") or {}
    if penyesuaian:
        detail = []
        if penyesuaian.get("luas_bangunan_maks_m2") is not None:
            detail.append(f"luas bangunan maksimal {_angka_prompt(penyesuaian['luas_bangunan_maks_m2'])} m²")
        if penyesuaian.get("luas_rth_min_m2") is not None:
            detail.append(f"RTH minimal {_angka_prompt(penyesuaian['luas_rth_min_m2'])} m²")
        dimensi = target_mitigasi.get("dimensi_minimum_resapan") or {}
        if dimensi.get("nilai") is not None:
            detail.append(
                "ATAU sumur/kolam resapan dimensi minimum "
                f"{_angka_prompt(dimensi['nilai'])} {dimensi.get('satuan') or ''}"
            )
        if detail:
            # ", " BUKAN "; " — item permintaan user 2026-09-21: DILARANG tanda titik koma di
            # template maupun luaran LLM (SYSTEM_PROMPT aturan #10 baru).
            lines.append(
                "\nRINCIAN_MITIGASI_KONKRET (SUDAH dihitung penuh oleh back-end, FINAL): "
                + ", ".join(detail) + ". Rekomendasi resmi poin ini SUDAH ditetapkan persis angka "
                "ini — jelaskan di reasoning_pendek/panjang KENAPA mitigasi ini diperlukan (kategori "
                f"{poin.status}), sebutkan angka-angka di atas APA ADANYA, JANGAN mengarang angka "
                "lain atau menghitung ulang. Persentase KDB dan KDH pada rincian ini adalah TARGET untuk "
                "menurunkan kategori dampak, BUKAN ambang intensitas ruang. JANGAN membandingkannya "
                "dengan ambang KDB/KDH dari ketentuan intensitas (itu dibahas di poin lain) dan JANGAN "
                "menulis 'batas KDB maksimum' atau 'ambang KDH' di poin dampak."
            )

    return lines


_BANGUN_FAKTA = {
    "itbx": _bangun_fakta_itbx,
    "intensitas": _bangun_fakta_intensitas,
    "dampak": _bangun_fakta_dampak,
}


CAVEAT_SUBZONA_TAK_TERKONFIRMASI = (
    "Sub-zona presisi lokasi permohonan tidak dikonfirmasi oleh back-end, sehingga ketentuan "
    "intensitas yang dikutip diambil dari tabel tingkat keluarga zona. Ambang KDB/KLB/KDH "
    "antar sub-zona dalam satu keluarga dapat berbeda, jadi angka ambang pada sitasi wajib "
    "diverifikasi terhadap sub-zona yang sebenarnya sebelum dipakai sebagai dasar keputusan."
)


def build_user_prompt(
    poin: PoinKonteks,
    chunks: list[Chunk],
    meta: MetaL2 | None = None,
    catatan_perbaikan: str | None = None,
    *,
    konteks_induk: dict[str, str] | None = None,
    catatan_tambahan: list[str] | None = None,
) -> str:
    """Susun prompt user, deterministik dari poin (fakta adapter) + chunk yang diretrieve.

    `konteks_induk` ({chunk_id: teks pasal induk}, dari generator.ambil_konteks_induk) melengkapi
    small-to-big: chunk ayat sering merujuk ayat lain ("sebagaimana dimaksud pada ayat (1)") yang
    tak ikut terkirim. Ditempel sbg konteks baca, BUKAN kandidat sitasi baru.

    `catatan_tambahan` masuk ke blok "Catatan (WAJIB disebutkan)" yang sama dengan
    `meta.caveats` — dipakai untuk batas yang diketahui KODE, bukan dikirim back-end (mis.
    sub-zona tak terkonfirmasi, lihat CAVEAT_SUBZONA_TAK_TERKONFIRMASI). Caveat di sini TIDAK
    menggantikan catatan_global deterministik di assemble.py: yang ini membuat narasinya tidak
    mengklaim lebih dari yang diketahui, yang itu menjamin pembaca tetap melihatnya walau LLM
    gagal atau narasinya jatuh ke template."""
    lines: list[str] = []

    lines.append("## Poin (SUDAH FINAL — jangan diubah/dihitung ulang)")
    lines.append(f"- poin_id: {poin.poin_id}")
    lines.append(f"- kategori: {poin.kategori}")
    lines.append(f"- tipe_rekomendasi: {poin.tipe_rekomendasi}")
    if poin.zona_info is not None and poin.zona_info.nama_zona:
        z = poin.zona_info
        lines.append(f"- ZONA_YANG_DINILAI: {z.nama_zona}" + (f" (kode {z.kode_zona})" if z.kode_zona else ""))
        lines.append(
            "- Permohonan ini menyilang beberapa zona RDTR dan tiap zona dinilai terpisah. Jelaskan HANYA untuk "
            "zona ini, sebut nama zonanya, dan jangan membahas atau menyimpulkan penilaian zona lain."
        )
    lines.append("")

    lines.append("## Fakta (dihitung kode — gunakan APA ADANYA, jangan disimpulkan ulang)")
    bangun_fakta = _BANGUN_FAKTA.get(poin.poin_id)
    lines.extend(bangun_fakta(poin) if bangun_fakta else [f"STATUS: {poin.status}"])

    # data_confidence_keseluruhan SENGAJA TIDAK disuntikkan ke prompt (Fix #4) — label kepercayaan
    # adalah FAKTA, dirakit deterministik di guardrail.py::_paksa_field_wajib, bukan bahasa yang
    # diserahkan ke LLM utk echo/parafrase (sumber kebocoran token mentah "DATA_CONFIDENCE: X").
    semua_caveat = list(meta.caveats) if meta and meta.caveats else []
    semua_caveat += list(catatan_tambahan or [])
    if semua_caveat:
        lines.append("")
        lines.append("## Catatan (WAJIB disebutkan dalam reasoning)")
        for caveat in semua_caveat:
            lines.append(f"CAVEAT: {caveat}")

    lines.append("")
    lines.append("## Pasal/Sitasi Tersedia (HANYA boleh kutip dari daftar ini, pakai citation_id persis)")
    ada_sitasi = False
    if poin.dasar_hukum:
        lines.append("Anchor dasar hukum dari back-end (prioritaskan ini):")
        # APP-2026-9461: potong ke _MAKS_ANCHOR (lihat komentar di atas) — prevent 413/429.
        anchor_inti = poin.dasar_hukum[:_MAKS_ANCHOR]
        for i, dasar_hukum in enumerate(anchor_inti):
            lines.append(_format_anchor(i, dasar_hukum))
        if len(poin.dasar_hukum) > _MAKS_ANCHOR:
            lines.append(
                f"(cetakan kiri: {len(poin.dasar_hukum) - _MAKS_ANCHOR} item dasar hukum lain "
                "tidak ditampilkan untuk menjaga ukuran konteks — jangan mengarang pasal di luar daftar)"
            )
        ada_sitasi = True
    if chunks:
        lines.append("Pasal tambahan dari RAG:")
        for chunk in chunks:
            lines.append(_format_chunk(chunk, (konteks_induk or {}).get(chunk.id)))
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
1. Ringkasan per-poin di bawah SUDAH FINAL (status, saran) — tugas Anda HANYA merangkum jadi langkah_berdampak (daftar langkah/pertimbangan untuk REVIEWER dalam mengambil keputusan — ACC / ACC bersyarat / tolak) dan catatan_lokasi (satu kalimat kalau relevan, atau null kalau tidak ada). JANGAN menghitung ulang angka, menyimpulkan status baru, atau mengubah verdict apa pun.
2. Poin ber-zona (itbx_X, intensitas_X) menilai SATU zona; kalau suatu langkah khusus untuk satu zona, sebut nama zonanya di langkah itu. langkah_berdampak HARUS dirangkai/diringkas HANYA dari "saran" per-poin persis seperti tertulis — JANGAN menambah langkah yang tidak berdasar pada saran poin manapun. KHUSUS: kalau saran suatu poin berbunyi "tidak diperlukan tindakan khusus" ATAU "tidak dievaluasi" (atau senada — termasuk poin yang belum pernah diperiksa krn proses berhenti di tahapan sebelumnya), JANGAN tulis langkah/kewajiban apa pun untuk poin itu di langkah_berdampak — poin itu cukup dilewati, bukan diberi kalimat pengganti yang menyiratkan ada yang harus dilakukan atau sudah terpenuhi.
3. Angka BOLEH disebut HANYA bila dikutip PERSIS seperti tertulis di "saran" per-poin (mis. luas maksimal, persentase target, dimensi) — itu lebih berguna bagi reviewer daripada "persentase tertentu". JANGAN menulis frasa samar seperti "persentase tertentu" atau "dimensi tertentu" kalau angkanya ada di saran. JANGAN membuat, menghitung, atau membulatkan angka sendiri, dan JANGAN menyebut skor.
4. Tulisan ini adalah bahan decision-support untuk REVIEWER (petugas Pemda), bukan nasihat langsung ke pemohon — sebut pemohon sebagai orang ketiga, JANGAN memakai "Anda". Tulis dalam Bahasa Indonesia yang jelas, profesional, dan ringkas.

5. "ringkasan" adalah SATU kalimat tindak lanjut utama untuk pemohon (kewajiban terpenting dari saran, mis. batas luas, RTH, atau syarat kegiatan), MAKSIMAL 25 kata. Kalimat status dan rekomendasi_sistem SUDAH ditulis sistem, jadi JANGAN mengulang rekomendasi_sistem, status ITBX, atau status intensitas, dan JANGAN memulai dengan "Rekomendasi sistem". Aturan angka dan larangan tanda titik koma (;) sama seperti di atas. Bila kewajiban khusus satu zona, sebut nama zonanya.

Balas HANYA dalam format JSON sesuai skema yang diberikan."""


def build_kesimpulan_prompt(poin_list: list[PoinOutput], rekomendasi_sistem: str) -> str:
    """Susun prompt sintesis kesimpulan — HANYA dari ringkasan per-poin yang sudah lolos guardrail,
    TIDAK ada fakta mentah/angka (poin ini sudah bebas angka per SYSTEM_PROMPT aturan #8).

    `reasoning_pendek` SENGAJA TIDAK disertakan di sini (beda dari versi lama) — bug ditemukan live
    (APP-2026-8376): poin dampak "aman" (saran="Tidak diperlukan tindakan khusus...") tetap
    menghasilkan langkah_berdampak yang menyiratkan ada kewajiban ("...harus memastikan tidak
    merusak..."), krn reasoning_pendek poin itu (narasi bebas LLM, menjelaskan KENAPA lolos per
    SYSTEM_PROMPT aturan #10) ditaruh berdampingan dgn saran dan menarik LLM sintesis mengikuti
    nada reasoning, bukan saran — padahal Aturan #2 di bawah SUDAH bilang "HANYA dari saran". Hapus
    reasoning dari konteks di sini MEMAKSA kepatuhan itu secara struktural, bukan cuma instruksi.
    """
    lines: list[str] = []
    lines.append("## Ringkasan Per-Poin (SUDAH FINAL — jangan dihitung ulang)")
    lines.append(f"rekomendasi_sistem keseluruhan: {rekomendasi_sistem}")
    lines.append("")
    for poin in poin_list:
        zona = f", zona={poin.zona.nama_zona}" if poin.zona and poin.zona.nama_zona else ""
        lines.append(f"- [{poin.poin_id}] status={poin.status}{zona}, low_confidence={poin.low_confidence}")
        lines.append(f"  saran: {poin.rekomendasi.saran}")

    lines.append("")
    lines.append(
        "## Tugas\n"
        "Rangkum ringkasan per-poin di atas menjadi ringkasan (1-2 kalimat), langkah_berdampak (daftar kalimat "
        "langkah/pertimbangan untuk reviewer dalam mengambil keputusan, pemohon disebut sebagai "
        "orang ketiga) dan catatan_lokasi (satu kalimat atau null)."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Narasi rekomendasi 2 paragraf — SATU panggilan TAMBAHAN per precheck, lihat assemble.py.
# Item permintaan user 2026-09-21: prosa naratif ringkas berbeda dari langkah_berdampak
# (checklist per-butir) — paragraf 1 gerbang hukum (ITBX+Intensitas), paragraf 2 dampak hidrologi.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_NARASI_REKOMENDASI = """Anda menulis narasi rekomendasi akhir pre-check izin bangunan Kabupaten Sleman untuk reviewer Pemda, dalam 2 paragraf.

ATURAN WAJIB (jangan dilanggar):
1. Ringkasan per-poin di bawah SUDAH FINAL (status, saran) — tugas Anda HANYA menulis ulang jadi 2 paragraf prosa mengalir, BUKAN daftar butir. JANGAN menghitung ulang angka, menyimpulkan status baru, atau mengubah verdict apa pun.
2. "paragraf_gate_intensitas" merangkum HANYA poin itbx dan intensitas (klasifikasi kegiatan dan intensitas bangunan KDB/KLB/KDH) — sebutkan status keduanya dan inti saran/tindak lanjutnya kalau ada. Bila ada beberapa zona (poin itbx_X dan intensitas_X per zona), jelaskan tiap zona secara terpisah dengan menyebut nama zonanya, jangan dicampur.
3. "paragraf_dampak" merangkum HANYA poin dampak (Dampak Terhadap Lingkungan/Hidrologi) — sebutkan status/kategorinya dan inti saran/tindak lanjutnya kalau ada.
4. Kalau saran suatu poin berbunyi "tidak diperlukan tindakan khusus" ATAU "tidak dievaluasi" (atau senada), JANGAN tulis kewajiban/tindakan apa pun untuk poin itu — sebut singkat bahwa poin itu sudah memenuhi ketentuan atau belum dinilai, TANPA menyiratkan ada yang harus dilakukan.
5. Angka BOLEH disebut HANYA bila dikutip PERSIS seperti tertulis di "saran" per-poin (mis. luas maksimal, persentase target) — JANGAN menulis frasa samar seperti "persentase tertentu" kalau angkanya ada di saran. JANGAN membuat, menghitung, atau membulatkan angka sendiri, dan JANGAN menyebut skor.
6. JANGAN menyalin citation_id, nama field/label teknis (mis. "KATEGORI_DAMPAK", "rekomendasi_sistem"), atau tanda titik koma (;) — tulis dalam Bahasa Indonesia biasa, kalimat dipisah titik/koma.
7. Singkatan resmi (ITBX, KDB, KLB, KDH, RTH) SELALU ditulis kapital semua.
8. Tulisan ini bahan decision-support untuk REVIEWER (petugas Pemda), bukan nasihat ke pemohon — sebut pemohon sebagai orang ketiga, JANGAN memakai "Anda". Tiap paragraf ringkas (2-4 kalimat), jelas, dan profesional.

Balas HANYA dalam format JSON sesuai skema yang diberikan."""


def build_narasi_rekomendasi_prompt(poin_list: list[PoinOutput], rekomendasi_sistem: str) -> str:
    """Susun prompt narasi 2 paragraf — HANYA dari ringkasan per-poin yang sudah lolos guardrail
    (pola sama dgn build_kesimpulan_prompt: reasoning_pendek SENGAJA tak disertakan, cuma
    status+saran, supaya narasi konsisten dgn saran final bukan mengikuti nada reasoning bebas)."""
    by_id = {p.poin_id: p for p in poin_list}
    lines: list[str] = []
    lines.append("## Ringkasan Per-Poin (SUDAH FINAL — jangan dihitung ulang)")
    lines.append(f"rekomendasi_sistem keseluruhan: {rekomendasi_sistem}")
    lines.append("")
    lines.append("### Untuk paragraf_gate_intensitas (ITBX + Intensitas SAJA):")
    for poin in poin_list:
        if poin.poin_id.split("_", 1)[0] in ("itbx", "intensitas"):
            zona = f" zona={poin.zona.nama_zona}" if poin.zona and poin.zona.nama_zona else ""
            lines.append(f"- [{poin.poin_id}] status={poin.status}{zona}")
            lines.append(f"  saran: {poin.rekomendasi.saran}")
    lines.append("")
    lines.append("### Untuk paragraf_dampak (Dampak Terhadap Lingkungan/Hidrologi SAJA):")
    poin_dampak = by_id.get("dampak")
    if poin_dampak:
        lines.append(f"- [{poin_dampak.poin_id}] status={poin_dampak.status}")
        lines.append(f"  saran: {poin_dampak.rekomendasi.saran}")

    lines.append("")
    lines.append(
        "## Tugas\n"
        "Tulis paragraf_gate_intensitas (rangkuman ITBX+Intensitas) dan paragraf_dampak (rangkuman "
        "Dampak Terhadap Lingkungan/Hidrologi), masing-masing satu paragraf prosa ringkas."
    )

    return "\n".join(lines)
