"""Guardrail — cek murah deterministik atas PoinOutput SEBELUM diloloskan, + loop retry + fallback.

CLAUDE.md / docs/Blueprint-Adaptasi-Model-L2-Gate-Impact.md §5: "Guardrail gagal -> regenerasi
terarah (maks 1-2x) -> fallback template + tanda low_confidence. JANGAN loop tak terbatas." Semua
cek di sini murni Python (regex/perbandingan) — TIDAK ADA panggilan LLM.

Dua kategori cek, disengaja dipisah:
- **Paksaan** (`_paksa_field_wajib`): diterapkan ke SEMUA jalur keluar (aman/template/hasil-bersih)
  TANPA memicu retry — fallback ITBX data-kosong, caveat meta, wiring `cek_konsistensi_intensitas`.
  Regenerasi LLM tidak bisa memperbaiki data back-end yang tak konsisten atau caveat yang hilang;
  lebih murah & pasti benar kalau disuntik langsung via kode.
- **Masalah teks** (`perbaiki_poin`): dipanggil di SEMUA poin setelah panggilan LLM (APP-2026-3468:
  poin aman kini juga lewat LLM+guardrail penuh, bukan cuma poin non-aman spt komentar lama di
  sini) — cek reasoning/saran kosong, sitasi hilang, invers-skor, konsistensi verdict, angka di
  narasi. Target rekomendasi HARUS ikut gerbang `apakah_aman()` (APP-2026-8376: `perbaiki_poin`
  sempat menghitung ulang target TANPA cek aman, menimpa `target=None` yang benar dari
  `generate_poin()` dgn angka kalkulator — kontradiktif dgn saran "tidak perlu tindakan").

Verifikasi entailment sitasi/verdict (NLI/LLM) DITUNDA sampai eval membuktikan perlu — lihat stub
`verifikasi_entailment_sitasi` di bawah, tidak dipanggil di alur utama.
"""

import logging
import re
from dataclasses import dataclass, field

from app.adapter import cek_konsistensi_intensitas
from app.reasoning.calculator import (
    bangun_langkah_konkret_dampak,
    bangun_langkah_konkret_intensitas,
    pilih_target_mitigasi_dampak,
    pilih_target_utama_intensitas,
)
from app.reasoning.generator import ambil_chunks_pendukung, apakah_aman, generate_poin
from app.reasoning.templates import template_low_confidence
from app.retrieval.base import Chunk, Retriever
from app.schemas import L2Assessment, LangkahKonkretOutput, PoinKonteks, PoinOutput

logger = logging.getLogger(__name__)

CAVEAT_FALLBACK_ITBX_LOLOS = (
    "diloloskan otomatis karena data matriks RDTR kosong, bukan kepatuhan terverifikasi"
)
CAVEAT_FALLBACK_ITBX_NON_LOLOS = (
    "penentuan status ini didasarkan pada data matriks RDTR yang mungkin belum lengkap — perlu "
    "verifikasi manual apakah kegiatan benar-benar dilarang atau datanya belum tersedia"
)


def caveat_fallback_itbx(status: str) -> str:
    """Bug APP-2026-3335: caveat fallback ITBX HARUS sadar status, bukan satu kalimat generik.
    Status "I" pakai frasa "diloloskan otomatis" (akurat — memang lolos, tapi tak terverifikasi).
    Status lain (mis. "X" / Tidak Lolos) DILARANG memakai kata "diloloskan" sama sekali — kalimat
    itu menyiratkan permohonan lolos padahal verdict sebenarnya bisa Tidak Lolos, kontradiktif
    dengan reasoning yang menyertainya. Framing netral dipakai sebagai gantinya, tidak mengklaim
    arah keputusan apa pun.
    """
    if status == "I":
        return CAVEAT_FALLBACK_ITBX_LOLOS
    return CAVEAT_FALLBACK_ITBX_NON_LOLOS

_LABEL_DATA_CONFIDENCE = {"high": "tinggi", "medium": "sedang", "low": "rendah"}


def _kalimat_tingkat_kepercayaan(data_confidence: str | None) -> str | None:
    """Fix #4: label kepercayaan = FAKTA, dirakit DI KODE dari data_confidence — jangan diserahkan
    ke LLM (token mentah "DATA_CONFIDENCE: X" tak lagi disuntikkan ke prompt, lihat prompts.py).
    None/tak dikenal -> jangan tampilkan label kepercayaan sama sekali.
    """
    if not data_confidence:
        return None
    label = _LABEL_DATA_CONFIDENCE.get(data_confidence.strip().lower())
    if label is None:
        return None
    return f"Tingkat kepercayaan data: {label}."


_FRASA_DAMPAK_TINGGI = ("risiko tinggi", "dampak tinggi", "sangat berisiko", "risiko sangat tinggi")
_FRASA_DAMPAK_RENDAH = ("risiko rendah", "dampak rendah", "aman sepenuhnya", "tanpa risiko")

_RE_BAND = re.compile(r"(?P<op>[<>])?\s*(?P<n1>\d+(?:\.\d+)?)\s*(?:-\s*(?P<n2>\d+(?:\.\d+)?))?")
_RE_ANGKA_MENCURIGAKAN = re.compile(r"\b\d+[.,]\d+\b|\b\d{2,}\b")


# ---------------------------------------------------------------------------
# Cek #1 — Invers-skor dampak (teks + sanity-check data)
# ---------------------------------------------------------------------------


def _cari_band_untuk_index(index: float, bands: dict[str, str]) -> str | None:
    """Cari kategori mana yang cocok dgn `index` menurut `threshold_bands` APA ADANYA dari
    back-end (mis. "index < 1.5", "1.5-2.5", "> 4.0") — TIDAK menghitung ulang rumus C.
    """
    for kategori, rentang in bands.items():
        match = _RE_BAND.search(rentang)
        if not match:
            continue
        op, n1, n2 = match.group("op"), float(match.group("n1")), match.group("n2")
        if n2 is not None:
            if n1 <= index <= float(n2):
                return kategori
        elif op == "<":
            if index < n1:
                return kategori
        elif op == ">":
            if index > n1:
                return kategori
    return None


def _cek_invers_skor(poin_output: PoinOutput, poin: PoinKonteks) -> list[str]:
    if poin.poin_id != "dampak":
        return []
    masalah: list[str] = []
    teks = f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang}".lower()
    kategori = poin.status

    if kategori in ("Rendah", "Sedang"):
        for frasa in _FRASA_DAMPAK_TINGGI:
            if _frasa_muncul_tanpa_negasi(teks, frasa):
                masalah.append(
                    f"Reasoning menyiratkan dampak tinggi ('{frasa}') padahal kategori aktual "
                    f"'{kategori}' — cek arah skor invers."
                )
                break
    elif kategori in ("Tinggi", "Sangat Tinggi"):
        for frasa in _FRASA_DAMPAK_RENDAH:
            if _frasa_muncul_tanpa_negasi(teks, frasa):
                masalah.append(
                    f"Reasoning menyiratkan dampak rendah ('{frasa}') padahal kategori aktual "
                    f"'{kategori}'."
                )
                break

    index = poin.fakta.get("runoff_change_index")
    bands = poin.fakta.get("threshold_bands")
    if index is not None and bands:
        band_kategori = _cari_band_untuk_index(index, bands)
        if band_kategori is not None and band_kategori != kategori:
            masalah.append(
                f"runoff_change_index={index} jatuh di band '{band_kategori}' menurut threshold_bands "
                f"back-end, tapi impact_category='{kategori}' — data back-end tak konsisten."
            )

    return masalah


# ---------------------------------------------------------------------------
# Cek #5 — Konsistensi verdict (teks)
# ---------------------------------------------------------------------------

_KATA_NEGASI = ("tidak", "bukan", "belum", "tanpa")
_JARAK_NEGASI_KARAKTER = 20


def _frasa_muncul_tanpa_negasi(teks: str, frasa: str) -> bool:
    """True kalau `frasa` muncul di `teks` TANPA didahului kata negasi (tidak/bukan/belum/tanpa)
    dalam jarak dekat. APP-2026-3468: poin aman/lolos sekarang dijelaskan LLM (bukan template
    generik lagi), jadi wajar reasoning menyebut mis. "usulan TIDAK melampaui ambang maksimum" —
    match substring naif lama akan salah menganggap ini sebagai kontradiksi verdict.
    """
    for match in re.finditer(re.escape(frasa), teks):
        awal = max(0, match.start() - _JARAK_NEGASI_KARAKTER)
        konteks_sebelum = teks[awal : match.start()].split()
        if not any(kata in _KATA_NEGASI for kata in konteks_sebelum):
            return True
    return False


def _cek_konsistensi_verdict(poin_output: PoinOutput, poin: PoinKonteks) -> list[str]:
    masalah: list[str] = []
    teks = f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang}".lower()

    if poin.poin_id == "itbx" and poin.status == "X":
        # Cek POSITIF (verdict larangan harus ditegaskan ADA), bukan negatif (kata "diizinkan"
        # dilarang muncul sama sekali) — SYSTEM_PROMPT aturan #7 mewajibkan LLM menyebut kegiatan
        # ALTERNATIF yang diizinkan di zona ini, jadi kata "diizinkan" WAJAR muncul (merujuk
        # kegiatan lain, bukan kegiatan yang diusulkan). Cek negatif lama false-positive di sini.
        frasa_larangan = ("dilarang", "tidak diizinkan", "tidak diperbolehkan", "tidak boleh")
        if not any(frasa in teks for frasa in frasa_larangan):
            masalah.append(
                "Reasoning tidak menegaskan larangan ('dilarang'/'tidak diizinkan'/dst tidak "
                "ditemukan) padahal status ITBX = X — verdict harus dinyatakan jelas."
            )

    if poin.poin_id == "intensitas":
        if poin.status == "MEMENUHI_SYARAT":
            for frasa in ("melanggar", "melampaui batas", "melampaui ambang"):
                if _frasa_muncul_tanpa_negasi(teks, frasa):
                    masalah.append(
                        f"Reasoning menyiratkan pelanggaran ('{frasa}') padahal status = MEMENUHI_SYARAT."
                    )
                    break
        elif poin.status == "MELAMPAUI_BATAS":
            for frasa in ("memenuhi seluruh standar", "tidak ada pelanggaran", "sudah sesuai semua"):
                if _frasa_muncul_tanpa_negasi(teks, frasa):
                    masalah.append(
                        f"Reasoning menyiratkan kepatuhan penuh ('{frasa}') padahal status = MELAMPAUI_BATAS."
                    )
                    break

    return masalah


# ---------------------------------------------------------------------------
# Cek #6 — Konsistensi numerik (teks)
# ---------------------------------------------------------------------------


def _angka_fakta_poin(poin: PoinKonteks) -> list[str]:
    """Angka GROUND TRUTH dari fakta back-end/calculator poin ini (bukan dihitung/dikarang LLM) —
    sah dikutip verbatim di narasi. Diagnosis flaky-fallback intensitas (APP-2026-6191, live thd DB
    nyata): intensitas TAK PERNAH punya keterangan_ketentuan/dasar_hukum (lihat app/adapter.py), jadi
    _angka_terlacak_ke_sumber versi lama menolak SETIAP angka >=2 digit di narasi — termasuk usulan/
    ambang yang justru WAJIB disebut LLM utk menjelaskan pelanggaran intensitas. Di sini "sumber"
    diperluas mencakup angka fakta poin ini sendiri (parameter/target/luasan utk intensitas;
    impact_score/runoff_change_index/c_before/c_after utk dampak) — angka lain (dihitung/dikarang
    LLM sendiri) TETAP ditolak seperti semula.
    """
    fakta = poin.fakta
    angka: list[str] = []

    def _tambah(nilai: object) -> None:
        if nilai is None or isinstance(nilai, bool):
            return
        if isinstance(nilai, (int, float)):
            angka.append(str(nilai))
            if isinstance(nilai, float) and nilai.is_integer():
                angka.append(str(int(nilai)))
            elif isinstance(nilai, int):
                angka.append(f"{nilai}.0")

    if poin.poin_id == "itbx":
        # kbli_diusulkan itu STRING ("0111"), bukan angka Python — lewat `_tambah` (int/float only)
        # tak akan pernah tertangkap. Ditemukan live (APP-2026-6191): LLM menyebut "KBLI 0111"
        # (FAKTA sah, ada persis di fakta['kbli_diusulkan']) di reasoning/saran, tapi sebelumnya
        # SELALU ditolak Cek #6 sbg "angka tak terlacak" krn itbx tak pernah dimasukkan ke sini.
        kbli = fakta.get("kbli_diusulkan")
        if kbli:
            angka.append(str(kbli))
    elif poin.poin_id == "intensitas":
        for param in (fakta.get("parameter") or {}).values():
            _tambah(param.get("usulan"))
            _tambah(param.get("ambang_maks"))
            _tambah(param.get("ambang_min"))
        for target in (fakta.get("target") or {}).values():
            nilai_target = target.values() if isinstance(target, dict) else [target]
            for nilai in nilai_target:
                _tambah(nilai)
        _tambah(fakta.get("luas_tapak_m2"))
        _tambah(fakta.get("jumlah_lantai"))
        _tambah(fakta.get("luas_rth_usulan_m2"))
    elif poin.poin_id == "dampak":
        _tambah(fakta.get("impact_score"))
        _tambah(fakta.get("runoff_change_index"))
        _tambah(fakta.get("c_before"))
        _tambah(fakta.get("c_after"))
        target_mitigasi = fakta.get("target_mitigasi") or {}
        _tambah(target_mitigasi.get("runoff_change_index_maks"))
        _tambah(target_mitigasi.get("index_saat_ini"))
        # APP-2026-8025/-5067: angka rincian penyesuaian lahan/dimensi sumur resapan dari
        # rekomendasi_mitigasi BE (lihat calculator.py::hitung_target_mitigasi_dampak) — GROUND
        # TRUTH BE, sah dikutip verbatim persis spt angka lain di atas.
        penyesuaian = target_mitigasi.get("penyesuaian_lahan") or {}
        _tambah(penyesuaian.get("luas_bangunan_maks_m2"))
        _tambah(penyesuaian.get("luas_rth_min_m2"))
        _tambah(penyesuaian.get("kdb_maks_persen"))
        _tambah(penyesuaian.get("kdh_min_persen"))
        _tambah(target_mitigasi.get("luas_bangunan_saat_ini_m2"))
        _tambah(target_mitigasi.get("luas_rth_saat_ini_m2"))
        dimensi = target_mitigasi.get("dimensi_minimum_resapan") or {}
        _tambah(dimensi.get("nilai"))

    return angka


_RE_ANGKA_DI_SUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _dekat_dgn_pembulatan(angka_str: str, sumber: str) -> bool:
    """True kalau `angka_str` adalah versi DIBULATKAN (ke presisi berapa pun) dari salah satu angka
    di `sumber`. Ditemukan live (APP-2026-8376): fakta `kdh.usulan=29.411764705882355` (presisi
    penuh float back-end) — LLM WAJAR menulis "29.41" di narasi (tak ada yg menulis 15 digit desimal
    dalam kalimat), tapi match string persis (word-boundary) di atas menolaknya sbg "angka karangan".
    Bukan pelonggaran umum: dicek angka SUMBER dibulatkan ke presisi PERSIS SAMA dgn yg ditulis LLM
    harus SAMA PERSIS (bukan toleransi rentang) — angka yang benar-benar beda tetap ditolak."""
    try:
        nilai_llm = float(angka_str)
    except ValueError:
        return False
    desimal = len(angka_str.split(".", 1)[1]) if "." in angka_str else 0
    for kandidat in _RE_ANGKA_DI_SUMBER.findall(sumber):
        try:
            nilai_sumber = float(kandidat)
        except ValueError:
            continue
        if round(nilai_sumber, desimal) == nilai_llm:
            return True
    return False


def _teks_chunk_disitasi(poin_output: PoinOutput | None, chunks: list[Chunk] | None) -> str:
    """Teks chunk RAG yang BENAR-BENAR DISITASI output ini (`sitasi[].citation_id` -> `chunk.id`),
    bukan semua chunk yang kebetulan disodorkan ke LLM.

    Pembedaan itu inti dari pelonggaran ini (bukti replay 8 fixture, 2026-09-08): angka yang selama
    ini ditolak ternyata nilai asli Lampiran VI utk sub-zona pemohon ("KDB maksimum 10%, KLB 0.1,
    KDH 85-88% untuk zona P-1"), dikutip verbatim dari chunk yang kita sodorkan sendiri lengkap dgn
    atribusi pasalnya — jelas BUKAN halusinasi, tapi ditolak karena whitelist tak pernah mencakup isi
    chunk. Kalau dilonggarkan ke SEMUA chunk yang disodorkan, model bebas menyeret ambang intensitas
    ke narasi ITBX/dampak tanpa menyebut sumbernya — persis pola yang terlihat di replay itu. Dgn
    syarat "harus disitasi", tiap angka yang muncul selalu punya sumber yang tercantum di output dan
    bisa diaudit reviewer; mengutip tanpa menyitasi TETAP ditolak seperti semula.
    """
    if poin_output is None or not chunks:
        return ""
    id_disitasi = {s.citation_id for s in poin_output.sitasi if s.citation_id}
    return " ".join(c.teks for c in chunks if c.id in id_disitasi)


def _angka_terlacak_ke_sumber(
    angka: str,
    poin: PoinKonteks,
    poin_output: PoinOutput | None = None,
    chunks: list[Chunk] | None = None,
) -> bool:
    """Investigasi ITBX APP-2026-6191: Cek #6 versi lama melarang SEMUA angka tanpa pandang sumber
    — menangkap angka ambang yang dikutip verbatim dari `keterangan_ketentuan`/`dasar_hukum` back-end
    (mis. "RTH minimal 20 dari luas persil"), padahal itu FAKTA sah, bukan halusinasi/hitungan LLM.

    Provenance check: angka BOLEH muncul di narasi HANYA kalau tercantum verbatim (word-boundary) di
    fakta sumber poin ini — `keterangan_ketentuan`/`dasar_hukum.kutipan` (ITBX) ATAU angka fakta
    poin itu sendiri via `_angka_fakta_poin` (intensitas/dampak) ATAU nomor pasal/ayat yang memang
    DIRUJUK poin ini (`poin.dasar_hukum[].pasal`) atau disitasi LLM sendiri (`poin_output.sitasi[].pasal`
    — retry sia-sia lama: LLM menyebut "Pasal 62" di narasi karena itu pasal yang benar-benar
    disitasi, bukan dikarang, tapi angkanya tak terlacak ke fakta sehingga selalu ditolak & memicu
    regenerasi tak perlu) ATAU nama dokumen yang disitasi (`poin.dasar_hukum[].dokumen`/
    `poin_output.sitasi[].dokumen` — investigasi APP-2026-3468 live: dokumen RAG asli bernama
    "Peraturan Bupati Sleman Nomor 80 Tahun 2023...", LLM menyebut "Nomor 80 Tahun 2023" saat
    merujuk sumbernya, angka itu bagian nama dokumen yang benar-benar disitasi, bukan karangan)
    ATAU versi DIBULATKAN dari salah satu angka fakta di atas (`_dekat_dgn_pembulatan`, APP-2026-8376
    — LLM wajar membulatkan angka float presisi tinggi saat menulis prosa)
    ATAU tercantum di teks chunk RAG yang BENAR-BENAR DISITASI output ini (`_teks_chunk_disitasi`,
    2026-09-08 — lihat helper itu utk bukti & alasan kenapa dibatasi ke yang disitasi saja)
    ATAU tercantum verbatim di narasi `rekomendasi_mitigasi` BE (APP-2026-8025/-5067 — lihat
    catatan di bawah kenapa teks MENTAH, bukan cuma angkanya, yang ditambahkan).
    Angka yang tak cocok sumber manapun TETAP ditolak — ini MEMPERKETAT presisi cek, bukan melonggarkan.
    """
    sumber = " ".join(poin.fakta.get("keterangan_ketentuan") or [])
    sumber += " " + " ".join(d.kutipan for d in poin.dasar_hukum)
    sumber += " " + " ".join(_angka_fakta_poin(poin))
    sumber += " " + " ".join(d.pasal or "" for d in poin.dasar_hukum)
    sumber += " " + " ".join(d.dokumen or "" for d in poin.dasar_hukum)
    # APP-2026-8025/-5067: `rekomendasi_mitigasi.saran`/`catatan` BE ditulis format Indonesia
    # (titik ribuan, koma desimal, mis. "11.551,06 m²") — `_RE_ANGKA_MENCURIGAKAN` men-tokenize ini
    # BEDA dari representasi float Python (`_angka_fakta_poin` di atas sudah menambah
    # "11551.06"/"11551", TAK match token "11.551"/"06" hasil tokenisasi format Indonesia).
    # Drpd menormalisasi format angka (rapuh, banyak kasus tepi ribuan/desimal), tambahkan teks BE
    # VERBATIM (byte-identik, TANPA reformat) ke `sumber` — saran poin dampak di sini SELALU echo
    # persis teks ini (generator.py::_saran_mitigasi_dampak), jadi token apa pun yang diekstrak dari
    # situ otomatis ketemu via substring match di bawah, terlepas skema tokenisasi.
    target_mitigasi = poin.fakta.get("target_mitigasi") or {}
    if target_mitigasi.get("saran_be"):
        sumber += " " + target_mitigasi["saran_be"]
    penyesuaian_lahan = target_mitigasi.get("penyesuaian_lahan") or {}
    if penyesuaian_lahan.get("catatan"):
        sumber += " " + penyesuaian_lahan["catatan"]
    if poin_output is not None:
        sumber += " " + " ".join(s.pasal or "" for s in poin_output.sitasi)
        sumber += " " + " ".join(s.dokumen or "" for s in poin_output.sitasi)
    sumber += " " + _teks_chunk_disitasi(poin_output, chunks)
    if re.search(rf"\b{re.escape(angka)}\b", sumber) is not None:
        return True
    return _dekat_dgn_pembulatan(angka, sumber)


def _cek_konsistensi_numerik(
    poin_output: PoinOutput, poin: PoinKonteks, chunks: list[Chunk] | None = None
) -> list[str]:
    """SYSTEM_PROMPT (prompts.py) aturan #8 melarang LLM menyebut angka yang TIDAK bisa dilacak ke
    fakta sumber (angka final tetap dirakit kode dari calculator/back-end, tidak pernah dari sini).
    Tidak scan `sitasi[].kutipan` — kutipan pasal boleh memuat angka (nomor pasal/ayat) yang sah.
    """
    teks = f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang} {poin_output.rekomendasi.saran}"
    for match in _RE_ANGKA_MENCURIGAKAN.finditer(teks):
        angka = match.group(0)
        if not _angka_terlacak_ke_sumber(angka, poin, poin_output, chunks):
            return [
                f"Reasoning/saran menyebutkan angka {angka!r} yang tidak tercantum di fakta sumber "
                "poin ini — dilarang (SYSTEM_PROMPT aturan #8). Angka harus berasal dari "
                "calculator/back-end via field terpisah, atau dikutip verbatim dari "
                "keterangan_ketentuan/dasar_hukum, bukan dihitung/dikarang LLM."
            ]
    return []


# ---------------------------------------------------------------------------
# Scrub deterministik — disclaimer fallback PALSU (LLM mengarang caveat "data matriks RDTR belum
# lengkap" sendiri). BUKAN cek pemicu-retry (lihat catatan di bawah kenapa).
# ---------------------------------------------------------------------------

# Ditemukan live (APP-2026-6191, poin itbx status T): reason "Lolos karena kegiatan Terbatas (T) di
# zona Zona Perumahan" TIDAK memicu deteksi_fallback_itbx() (dikonfirmasi False) — tapi LLM tetap
# menulis disclaimer yang nyaris identik dgn CAVEAT_FALLBACK_ITBX_NON_LOLOS atas inisiatifnya
# sendiri. Bukan bug di heuristik deteksi (sudah diverifikasi benar), tapi LLM "meniru" pola caveat
# yang dilihatnya di SYSTEM_PROMPT (aturan #3) padahal FALLBACK_DATA_KOSONG=False di fakta poin ini
# — menyesatkan reviewer (menyiratkan data tak lengkap padahal lengkap).
#
# PERCOBAAN PERTAMA (dibuang): jadikan ini cek pemicu-retry (`perbaiki_poin`) supaya LLM
# meregenerasi. TERBUKTI SALAH live: LLM mengulang pola yang SAMA di ketiga percobaan retry, retry
# exhaust, poin JATUH ke low_confidence penuh (reasoning/sitasi valid ikut terbuang) — lebih buruk
# drpd disclaimer yang cuma salah 1 kalimat. Diganti scrub DETERMINISTIK (`_paksa_field_wajib`,
# TIDAK memicu retry) — hapus fragmen kalimat yang cocok pola, sisa disclaimer/reasoning/sitasi tetap
# utuh. Faithful principle: ini bukan "mengoreksi fakta back-end" (dilarang), murni membuang kalimat
# TAMBAHAN yang LLM karang sendiri di luar fakta yang diberikan.
_FRASA_CAVEAT_FALLBACK = (
    "matriks rdtr yang mungkin belum lengkap",
    "data matriks rdtr kosong",
    "data matriks rdtr yang tidak lengkap",
)

_RE_PEMISAH_KALIMAT = re.compile(r"(?<=[.!?])\s+")

# Nama label teknis yang memang disuntikkan ke prompt (prompts.py) dan terbukti disalin mentah oleh
# LLM ke narasi utk petugas — replay 2026-09-08: "KATEGORI_DAMPAK: Rendah menunjukkan bahwa...",
# "KATEGORI_DAMPAK 'Rendah' menunjukkan...". Pola sama persis dgn kebocoran "DATA_CONFIDENCE: X"
# yang dulu diperbaiki dgn tidak menyuntikkan tokennya sama sekali — tapi label-label ini TIDAK bisa
# dihapus dari prompt (model memang butuh nilainya), jadi ditangani di sisi keluaran.
# SCRUB DETERMINISTIK, TIDAK memicu retry — alasannya sama dgn _bersihkan_disclaimer_fallback_palsu
# di atas: satu kata salah tak sebanding dgn membuang seluruh reasoning+sitasi yang sudah benar.
_LABEL_TEKNIS = {
    "KATEGORI_DAMPAK": "kategori dampak",
    "STATUS_ITBX": "status kegiatan",
    "STATUS_INTENSITAS": "status intensitas",
    "KEGIATAN_DIUSULKAN": "kegiatan yang diusulkan",
    "FALLBACK_DATA_KOSONG": "kelengkapan data matriks RDTR",
    # APP-2026-8025/-5067 (dampak mitigasi BE) — lihat prompts.py::_bangun_fakta_dampak.
    "ALASAN_TIDAK_DINILAI": "alasan belum dinilai",
    "PERINGATAN_LUAS_PERSIL": "peringatan luas persil",
    "TARGET_MITIGASI_KUANTITATIF": "target mitigasi",
    "RINCIAN_MITIGASI_KONKRET": "rincian mitigasi",
}
# WAJIB ada garis bawah: cegah akronim sah ikut tergilas (RDTR, KDB, KLB, KDH, ITBX, LP2B, PBG, RTH).
_RE_LABEL_TEKNIS = re.compile(
    "(?<![A-Za-z0-9_])[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+(?![A-Za-z0-9_])"
)


def _bersihkan_label_teknis(teks: str | None) -> str | None:
    """Ganti nama label teknis yang tersalin dari prompt dgn padanan Bahasa Indonesia biasa.
    Label tak dikenal tetap dinormalkan (huruf kecil, garis bawah -> spasi) supaya label BARU yang
    ditambahkan ke prompt nanti tidak diam-diam bocor lagi."""
    if not teks:
        return teks
    return _RE_LABEL_TEKNIS.sub(
        lambda m: _LABEL_TEKNIS.get(m.group(0), m.group(0).lower().replace("_", " ")), teks
    )


def _bersihkan_disclaimer_fallback_palsu(disclaimer: str | None) -> str | None:
    """Buang kalimat yang menyerupai caveat fallback-data-kosong dari `disclaimer` — dipanggil HANYA
    saat `fallback_data_kosong` benar-benar False (kalau True, caveat itu WAJIB ada, lihat Cek #2)."""
    if not disclaimer:
        return disclaimer
    kalimat = _RE_PEMISAH_KALIMAT.split(disclaimer)
    bersih = [k for k in kalimat if not any(frasa in k.lower() for frasa in _FRASA_CAVEAT_FALLBACK)]
    hasil = " ".join(k.strip() for k in bersih if k.strip())
    return hasil or None


# ---------------------------------------------------------------------------
# Paksaan wajib — cek #2, #3, #4. Diterapkan ke SEMUA jalur keluar, TIDAK memicu retry.
# ---------------------------------------------------------------------------


def _gabung_kalimat(bagian: list[str]) -> str:
    """Gabung beberapa kalimat/fragmen disclaimer jadi satu string, PASTIKAN ada pemisah kalimat
    (titik) antar fragmen — bug ditemukan live (APP-2026-6191): `" ".join(...)` polos bisa
    menyambung 2 kalimat tanpa titik kalau fragmen sebelumnya (mis. teks bebas dari LLM) tidak
    diakhiri tanda baca, menghasilkan disclaimer yang terbaca nyambung/rusak."""
    hasil: list[str] = []
    for b in bagian:
        b = (b or "").strip()
        if not b:
            continue
        if hasil and not hasil[-1].endswith((".", "!", "?")):
            hasil[-1] += "."
        hasil.append(b)
    return " ".join(hasil)


def _paksa_field_wajib(
    poin_output: PoinOutput, poin: PoinKonteks, assessment: L2Assessment
) -> PoinOutput:
    """Pastikan field yang WAJIB benar terlepas dari LLM/template — TIDAK PERNAH mengoreksi/
    menimpa fakta back-end, hanya memastikan penanda low_confidence & catatan wajib benar-benar
    ada (Faithful, CLAUDE.md/Blueprint §5).
    """
    update: dict = {}
    disclaimer_tambahan: list[str] = []
    disclaimer_dasar = poin_output.rekomendasi.disclaimer
    teks_sudah_ada = f"{poin_output.reasoning_panjang} {poin_output.rekomendasi.disclaimer or ''}"

    # Cek #2 — ITBX fallback data-kosong: paksa low_confidence + caveat wajib, wording sadar status
    # (APP-2026-3335: status X tak boleh dapat caveat "diloloskan otomatis" — lihat caveat_fallback_itbx).
    if poin.poin_id == "itbx" and poin.fakta.get("fallback_data_kosong"):
        update["low_confidence"] = True
        caveat = caveat_fallback_itbx(poin.status)
        if caveat.lower() not in teks_sudah_ada.lower():
            disclaimer_tambahan.append(caveat.capitalize() + ".")
    elif poin.poin_id == "itbx":
        # fallback_data_kosong MEMANG False -> scrub kalau LLM sempat menulis caveat serupa sendiri
        # (ditemukan live APP-2026-6191; deterministik, TIDAK memicu retry — lihat catatan di atas
        # fungsi _bersihkan_disclaimer_fallback_palsu kenapa bukan cek pemicu-retry).
        disclaimer_dasar = _bersihkan_disclaimer_fallback_palsu(disclaimer_dasar)

    # Cek #3 — meta.caveats / data_confidence WAJIB muncul. Kalimat kepercayaan SELALU dirakit
    # deterministik (bukan echo raw value LLM/back-end) — konsisten sama persis di ketiga poin.
    meta = assessment.meta
    if meta:
        kalimat_confidence = _kalimat_tingkat_kepercayaan(meta.data_confidence_keseluruhan)
        if kalimat_confidence and kalimat_confidence.lower() not in teks_sudah_ada.lower():
            disclaimer_tambahan.append(kalimat_confidence)
        for caveat in meta.caveats or []:
            if caveat not in teks_sudah_ada and caveat not in " ".join(disclaimer_tambahan):
                disclaimer_tambahan.append(f"Catatan: {caveat}")

    # Cek #3b — impact_assessment.luas_usulan_melebihi_persil (APP-2026-2428): peringatan WAJIB
    # muncul di disclaimer, dirakit DETERMINISTIK di sini — TIDAK bergantung LLM mengingat
    # menyebutnya sendiri di reasoning/saran (prompts.py sudah instruksikan LLM juga, tapi ini
    # jaring pengaman kedua, pola sama spt Cek #3 di atas utk data_confidence/caveats).
    if poin.poin_id == "dampak" and poin.fakta.get("luas_usulan_melebihi_persil"):
        peringatan_persil = (
            "Luas usulan tapak bangunan + RTH melebihi luas bidang persil yang tercatat — hasil "
            "perhitungan dampak berikut berpotensi kurang akurat, perlu peninjauan manual."
        )
        if peringatan_persil.lower() not in teks_sudah_ada.lower() and peringatan_persil not in disclaimer_tambahan:
            disclaimer_tambahan.append(peringatan_persil)

    # Cek #4 — wire cek_konsistensi_intensitas dari adapter.py. TIDAK PERNAH menimpa status/parameter.
    if poin.poin_id == "intensitas":
        masalah_konsistensi = cek_konsistensi_intensitas(assessment)
        if masalah_konsistensi:
            update["low_confidence"] = True
            logger.warning(
                "cek_konsistensi_intensitas menemukan masalah data back-end utk poin %r: %s",
                poin.poin_id,
                masalah_konsistensi,
            )

    if disclaimer_tambahan or disclaimer_dasar != poin_output.rekomendasi.disclaimer:
        gabungan = _gabung_kalimat(([disclaimer_dasar] if disclaimer_dasar else []) + disclaimer_tambahan)
        update["rekomendasi"] = poin_output.rekomendasi.model_copy(update={"disclaimer": gabungan or None})

    if update:
        poin_output = poin_output.model_copy(update=update)

    # Cek #5 — label teknis prompt yang tersalin mentah ke narasi petugas (replay 2026-09-08:
    # "KATEGORI_DAMPAK: Rendah menunjukkan bahwa..."). Dijalankan TERAKHIR supaya juga membersihkan
    # disclaimer hasil rakitan di atas, dan diterapkan ke SEMUA jalur keluar (aman/template/bersih).
    # Deterministik, TIDAK memicu retry — lihat _bersihkan_label_teknis.
    rekomendasi = poin_output.rekomendasi
    poin_output = poin_output.model_copy(update={
        "reasoning_pendek": _bersihkan_label_teknis(poin_output.reasoning_pendek),
        "reasoning_panjang": _bersihkan_label_teknis(poin_output.reasoning_panjang),
        "rekomendasi": rekomendasi.model_copy(update={
            "saran": _bersihkan_label_teknis(rekomendasi.saran),
            "disclaimer": _bersihkan_label_teknis(rekomendasi.disclaimer),
        }),
    })
    return poin_output


# ---------------------------------------------------------------------------
# Masalah teks — hanya jalur non-aman, memicu retry.
# ---------------------------------------------------------------------------

# Item permintaan user 2026-09-21 (aturan #17, prompts.py): citation_id ("rdtr-sleman-tengah-
# p53-a3", "anchor-0") adalah ID baris basis data, BUKAN bahasa manusia — reviewer tak paham/tak
# perlu tahu ID mentah ini. Dicek terhadap SEMUA id yang TERSEDIA (anchor dari poin.dasar_hukum +
# id chunk hasil retrieval), bukan cuma yang benar-benar disitasi output ini — supaya kebocoran ID
# yang SEHARUSNYA tak pernah dipilih pun (mis. LLM menyalin id chunk yg sekadar dibaca) tertangkap.
def _cek_citation_id_bocor(poin_output: PoinOutput, poin: PoinKonteks, chunks: list[Chunk]) -> list[str]:
    teks = (
        f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang} "
        f"{poin_output.rekomendasi.saran} {poin_output.rekomendasi.disclaimer or ''}"
    )
    kandidat_id = [f"anchor-{i}" for i in range(len(poin.dasar_hukum))] + [c.id for c in chunks]
    bocor = sorted({cid for cid in kandidat_id if cid and cid in teks})
    if bocor:
        return [
            f"reasoning/saran menyebutkan citation_id mentah {bocor!r} — DILARANG (SYSTEM_PROMPT "
            "aturan #17). Rujuk sumber pakai bahasa manusia (nama dokumen/nomor pasal), bukan ID "
            "internal — citation_id hanya boleh muncul di field sitasi."
        ]
    return []


# Item permintaan user 2026-09-21 (aturan #17, prompts.py): titik koma bukan gaya bahasa umum bagi
# pembaca non-teknis. Regenerasi (bukan scrub otomatis) — scrub berisiko menyatukan 2 klausa jadi
# kalimat rusak tanpa pemisah yang jelas (pola sama alasannya dgn kenapa _bersihkan_disclaimer_
# fallback_palsu di atas TIDAK dipakai utk kasus yang butuh regenerasi semantik, bukan sekadar hapus
# frasa).
_RE_TANDA_BACA_DILARANG = re.compile(r";")


def _cek_tanda_baca_dilarang(poin_output: PoinOutput) -> list[str]:
    teks = (
        f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang} "
        f"{poin_output.rekomendasi.saran} {poin_output.rekomendasi.disclaimer or ''}"
    )
    if _RE_TANDA_BACA_DILARANG.search(teks):
        return [
            "reasoning/saran memakai tanda titik koma (;) — DILARANG (SYSTEM_PROMPT aturan #17). "
            "Gunakan tanda baca umum (titik atau koma) sesuai bahasa sehari-hari."
        ]
    return []


def perbaiki_poin(
    poin_output: PoinOutput,
    poin: PoinKonteks,
    chunks: list[Chunk],
    assessment: L2Assessment,
) -> tuple[PoinOutput, list[str]]:
    """Cek & perbaiki PoinOutput terhadap ground truth (poin/calculator/chunk).

    Mengembalikan (poin_hasil_perbaikan, daftar_masalah). Daftar_masalah kosong berarti poin siap
    diloloskan; tidak kosong berarti perlu regenerasi teks.
    """
    masalah: list[str] = []

    # Forces defensif (idempoten) — jaring pengaman kalau generator.py suatu saat salah; bukan
    # sumber utama lagi karena generate_poin() baru sudah merakit field ini dgn benar.
    anchor_by_id = {f"anchor-{i}": d for i, d in enumerate(poin.dasar_hukum)}
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    sitasi_bersih = [
        s for s in poin_output.sitasi if s.citation_id in anchor_by_id or s.citation_id in chunk_by_id
    ]

    # Target HARUS ikut gerbang apakah_aman() sama persis dgn generate_poin() — JANGAN hitung ulang
    # dari kalkulator tanpa syarat (APP-2026-8376: kalkulator target & gerbang "aman" bisa punya
    # kriteria berbeda, mis. dampak "Sedang" -> aman=True tapi kalkulator tetap kasih angka ->
    # target bocor kontradiktif dgn saran "tidak perlu tindakan").
    target = None
    # langkah_konkret ikut gerbang yang SAMA persis dgn target di atas (alasan sama: aman/Tidak
    # Dinilai -> tak ada aksi konkret, jangan sampai kalkulator tetap kasih daftar item yg
    # kontradiktif dgn saran "tidak perlu tindakan").
    langkah_konkret: list[dict] = []
    if not apakah_aman(poin):
        if poin.tipe_rekomendasi == "numerik":
            target = pilih_target_utama_intensitas(poin.fakta.get("target") or {})
            langkah_konkret = bangun_langkah_konkret_intensitas(
                poin.fakta.get("parameter") or {}, poin.fakta.get("target") or {}
            )
        elif poin.tipe_rekomendasi == "numerik-mitigasi":
            target = pilih_target_mitigasi_dampak(poin.fakta.get("target_mitigasi") or {})
            langkah_konkret = bangun_langkah_konkret_dampak(poin.fakta.get("target_mitigasi") or {})

    poin_bersih = poin_output.model_copy(
        update={
            "status": poin.status,
            "sitasi": sitasi_bersih,
            "rekomendasi": poin_output.rekomendasi.model_copy(
                update={
                    "tipe": poin.tipe_rekomendasi,
                    "target": target,
                    # model_copy() TIDAK memvalidasi field yg di-update (beda dari konstruktor
                    # biasa) — konversi eksplisit ke LangkahKonkretOutput di sini, JANGAN kirim
                    # list[dict] mentah (akan lolos type checker statis tapi runtime jadi dict,
                    # bukan objek — ditemukan & diperbaiki saat implementasi, verifikasi empiris
                    # via python -c langsung sebelum wiring ini).
                    "langkah_konkret": [LangkahKonkretOutput(**item) for item in langkah_konkret],
                }
            ),
        }
    )

    if not poin_bersih.reasoning_pendek.strip() or not poin_bersih.reasoning_panjang.strip():
        masalah.append("reasoning_pendek/reasoning_panjang kosong.")
    if not poin_bersih.rekomendasi.saran.strip():
        masalah.append("rekomendasi.saran kosong.")
    if (chunks or poin.dasar_hukum) and not sitasi_bersih:
        masalah.append(
            "Pasal/anchor tersedia tapi tidak ada sitasi valid setelah verifikasi — kemungkinan "
            "narasi tidak grounded."
        )

    masalah.extend(_cek_invers_skor(poin_bersih, poin))
    masalah.extend(_cek_konsistensi_verdict(poin_bersih, poin))
    masalah.extend(_cek_konsistensi_numerik(poin_bersih, poin, chunks))
    masalah.extend(_cek_citation_id_bocor(poin_bersih, poin, chunks))
    masalah.extend(_cek_tanda_baca_dilarang(poin_bersih))

    poin_bersih = _paksa_field_wajib(poin_bersih, poin, assessment)

    return poin_bersih, masalah


def verifikasi_entailment_sitasi(poin: PoinOutput, chunks: list[Chunk]) -> bool:
    """TODO(Fase 3+): verifikasi entailment sitasi/verdict via model NLI kecil atau panggilan LLM.

    DITUNDA sampai eval membuktikan perlu (CLAUDE.md § Guardrail). Tidak dipanggil di alur utama —
    placeholder untuk pengembangan berikutnya.
    """
    return True


@dataclass
class DiagnosaPoin:
    """Kenapa satu poin berakhir seperti itu — sebab, bukan cuma akibat.

    Motivasi (2026-09-07): dari 76 permohonan dgn LLM sungguhan di `logs/precheck.jsonl`, 39 (51%)
    punya minimal satu poin `low_confidence`. Tapi begitu retry habis, `generate_poin_dengan_guardrail`
    hanya mengembalikan `template_low_confidence` — daftar `masalah` terakhir dibuang, dan
    `log_precheck` cuma menyimpan request+response akhir. Akibatnya tiga sebab yang butuh
    penanganan BERBEDA jadi tak terbedakan sama sekali:
      1. guardrail menolak isi/gaya bahasa jawaban model (mis. `_cek_konsistensi_verdict` menuntut
         frasa persis "dilarang"/"tidak diizinkan" — peninggalan tuning `llama-3.3-70b`),
      2. panggilan LLM gagal total (rate limit Groq / timeout / BadRequestError),
      3. retrieval tak memberi chunk pendukung sama sekali.
    Struktur ini membawa sebabnya keluar supaya bisa dihitung, BUKAN ditebak.

    TIDAK ikut ke `OutputL3` — ini data operasional, bukan bagian kontrak dgn back-end/reviewer.
    """

    poin_id: str
    berhasil: bool
    percobaan: int = 0                              # berapa kali generate_poin benar-benar dipanggil
    jumlah_chunk: int = 0                           # chunk pendukung yang berhasil diretrieve
    masalah_terakhir: list[str] = field(default_factory=list)   # temuan guardrail di percobaan terakhir
    exception_terakhir: str | None = None           # "RateLimitError: ..." kalau LLM-nya yang gagal
    # Narasi yang DITOLAK di percobaan terakhir (dipotong). `masalah_terakhir` cuma menyebut LABEL
    # temuannya — mis. "menyebutkan angka '10' yang tidak tercantum di fakta sumber" — dan label itu
    # TIDAK cukup utk memutuskan perbaikan: "10" bisa berarti "Pasal 10" (rujukan sah yang dibaca
    # LLM dari chunk RAG yang kita sodorkan sendiri) ATAU "kurangi 10%" (angka karangan). Dua hal itu
    # penanganannya BERLAWANAN — longgarkan cek vs perketat prompt — jadi kalimatnya wajib ikut.
    # SENGAJA tidak dimasukkan ke `masalah_terakhir`: daftar itu dirangkai jadi `catatan_perbaikan`
    # yang dikirim balik ke LLM saat retry, sehingga menambahinya = mengubah perilaku yang sedang
    # diukur. Ini murni pengamatan, satu arah keluar.
    teks_ditolak_terakhir: str | None = None

    def sebab(self) -> str:
        """Satu label kasar utk dihitung agregat: kenapa poin ini jatuh ke low_confidence."""
        if self.berhasil:
            return "berhasil"
        if self.exception_terakhir:
            return "panggilan_llm_gagal"
        if self.jumlah_chunk == 0:
            return "retrieval_kosong"
        if self.masalah_terakhir:
            return "guardrail_menolak"
        return "tak_diketahui"


# 400 terbukti terlalu kecil (replay 8 fixture, 2026-09-08): KELIMA teks yang ditolak terpotong
# persis di batas, dan di 4 dari 5 kasus angka yang dipermasalahkan berada SETELAH titik potong —
# jadi rekamannya ada tapi tak menjawab apa pun. reasoning_panjang sendiri biasanya ~500-900 char.
_MAKS_TEKS_DITOLAK = 1600


def _ringkas_teks_ditolak(poin_output: PoinOutput) -> str:
    """Gabung field narasi yang memang dipindai guardrail (`_cek_konsistensi_numerik` &
    `_cek_konsistensi_verdict` membaca ketiganya), lalu potong supaya baris log tetap wajar."""
    bagian = (
        poin_output.reasoning_pendek,
        poin_output.reasoning_panjang,
        poin_output.rekomendasi.saran,
    )
    return " | ".join(b for b in bagian if b)[:_MAKS_TEKS_DITOLAK]


def generate_poin_terdiagnosis(
    poin: PoinKonteks,
    retriever: Retriever,
    assessment: L2Assessment,
    *,
    max_retry: int = 2,
) -> tuple[PoinOutput, DiagnosaPoin]:
    """Sama persis dgn `generate_poin_dengan_guardrail`, TAPI ikut mengembalikan sebabnya.

    Alur & perilaku TIDAK berubah sedikit pun — ini murni penambahan pengamatan (lihat DiagnosaPoin).
    """
    diagnosa = DiagnosaPoin(poin_id=poin.poin_id, berhasil=False)

    try:
        chunks = ambil_chunks_pendukung(poin, retriever)
    except Exception as exc:
        # Retrieval gagal (mis. PIIDetectedError, DB putus) — sebelumnya exception ini merambat ke
        # assemble._generate_poin_defensif dan sebabnya cuma muncul di traceback. Catat dulu, lalu
        # teruskan apa adanya supaya perilaku isolasi per-poin di assemble tetap sama.
        diagnosa.exception_terakhir = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "ambil_chunks_pendukung gagal utk poin %r: %s", poin.poin_id, diagnosa.exception_terakhir
        )
        raise
    diagnosa.jumlah_chunk = len(chunks)

    masalah: list[str] = []
    for percobaan in range(max_retry + 1):
        catatan = "; ".join(masalah) if percobaan > 0 else None
        suhu = 0.4 if percobaan > 0 else 0.0
        diagnosa.percobaan = percobaan + 1

        try:
            hasil = generate_poin(
                poin,
                retriever,
                assessment.meta,
                catatan_perbaikan=catatan,
                temperature=suhu,
            )
        except Exception as exc:  # generasi gagal dihitung sebagai percobaan gagal, bukan crash
            # Ditemukan live (migrasi model 2026-08-15): exception di sini SEBELUMNYA tak pernah
            # dilog — begitu ketiga percobaan habis & jatuh ke template_low_confidence, root cause
            # asli (rate limit? BadRequestError? timeout?) hilang tak berbekas, tak bisa dibedakan
            # dari "model memang lemah". Log di sini TIDAK mengubah alur (masih retry lalu fallback
            # spt semula), cuma bikin kegagalan terlihat.
            logger.warning(
                "generate_poin gagal (percobaan %d/%d) utk poin %r: %s: %s",
                percobaan + 1, max_retry + 1, poin.poin_id, type(exc).__name__, exc,
            )
            diagnosa.exception_terakhir = f"{type(exc).__name__}: {exc}"
            masalah = [str(exc)]
            diagnosa.masalah_terakhir = list(masalah)
            continue

        # Percobaan ini sampai ke LLM & dapat jawaban -> exception percobaan SEBELUMNYA (kalau ada)
        # tak lagi menjadi sebab; jangan sampai salah melabeli "panggilan_llm_gagal".
        diagnosa.exception_terakhir = None
        poin_bersih, masalah = perbaiki_poin(hasil, poin, chunks, assessment)
        diagnosa.masalah_terakhir = list(masalah)
        diagnosa.teks_ditolak_terakhir = _ringkas_teks_ditolak(poin_bersih) if masalah else None
        if not masalah:
            diagnosa.berhasil = True
            if percobaan > 0:
                # Lolos TAPI butuh retry — near-miss. Cek guardrail mana yang paling sering menggigit
                # cuma kelihatan dari sini, bukan dari output akhir yang terlihat mulus.
                logger.info(
                    "poin %r lolos guardrail di percobaan ke-%d/%d", poin.poin_id, percobaan + 1, max_retry + 1
                )
            return poin_bersih, diagnosa

    logger.warning(
        "poin %r jatuh ke low_confidence setelah %d percobaan (sebab=%s, chunk=%d): %s",
        poin.poin_id, diagnosa.percobaan, diagnosa.sebab(), diagnosa.jumlah_chunk,
        "; ".join(diagnosa.masalah_terakhir) or "-",
    )
    return _paksa_field_wajib(template_low_confidence(poin), poin, assessment), diagnosa


def generate_poin_dengan_guardrail(
    poin: PoinKonteks,
    retriever: Retriever,
    assessment: L2Assessment,
    *,
    max_retry: int = 2,
) -> PoinOutput:
    """Entrypoint utama: generate_poin + guardrail + retry terarah + fallback low_confidence.

    APP-2026-3468: SEMUA poin (termasuk yang aman/lolos) lewat jalur LLM+guardrail penuh — poin
    aman sebelumnya short-circuit ke template_aman() (reasoning generik, sitasi selalu kosong),
    padahal reviewer tetap butuh tahu KENAPA poin ini lolos. Saran/target poin aman tetap
    ditemplate deterministik di dalam generate_poin() sendiri (lihat generator.py).

    Pembungkus tipis atas `generate_poin_terdiagnosis` — dipertahankan supaya pemanggil yang tak
    peduli sebab (test, skrip) tak perlu ikut membongkar tuple.
    """
    return generate_poin_terdiagnosis(poin, retriever, assessment, max_retry=max_retry)[0]
