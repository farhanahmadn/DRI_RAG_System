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

from app import batas_waktu
from app.adapter import cek_konsistensi_intensitas, cek_konsistensi_poin_intensitas
from app.batas_waktu import BatasWaktuTerlampaui
from app.reasoning.calculator import (
    bangun_langkah_dampak_poin,
    pilih_target_dampak_poin,
    bangun_langkah_konkret_intensitas,
    pilih_target_utama_intensitas,
)
from app.reasoning.generator import _lengkapi_sitasi_dampak, ambil_chunks_pendukung, apakah_aman, generate_poin
from app.reasoning.llm_client import LayananTidakTersediaError
from app.reasoning.templates import template_low_confidence
from app.retrieval.base import Chunk, Retriever
from app.schemas import L2Assessment, LangkahKonkretOutput, PoinKonteks, PoinOutput

logger = logging.getLogger(__name__)

# Sisa anggaran minimum (detik) agar satu percobaan generate+validasi masih layak dimulai.
_MIN_SISA_PERCOBAAN_S = 8.0

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
        _tambah(target_mitigasi.get("kdb_usulan_persen"))
        _tambah(target_mitigasi.get("kdh_usulan_persen"))
        dimensi = target_mitigasi.get("dimensi_minimum_resapan") or {}
        _tambah(dimensi.get("nilai"))
        # Dampak Rendah/Sedang: luas/KDB/KDH usulan pemohon utk saran pemeliharaan.
        pemeliharaan = (fakta.get("mitigasi") or {}).get("pemeliharaan") or {}
        for kunci in ("luas_bangunan_m2", "luas_rth_m2", "kdb_persen", "kdh_persen"):
            _tambah(pemeliharaan.get(kunci))

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
    if poin.poin_id == "dampak" and poin.kawasan_resapan is not None:
        # Status resapan diketahui -> narasi sah menyebut "Pasal 53" (berlaku atau tidak berlaku).
        sumber += " Pasal 53"
    if poin.poin_id == "dampak" and poin.rawan_kekeringan:
        sumber += " Pasal 50"
    # APP-2026-0095: `REASON` BE disuntikkan ke prompt ITBX (prompts._bangun_fakta_itbx) sehingga LLM
    # sah mengutip angkanya (mis. "Zona Pertanian (743.5 m2: Status X)"), tapi dulu tak termasuk
    # `sumber` -> guardrail menolak angka yg ia sendiri suruh LLM baca. Teks mentah, tanpa reformat.
    if isinstance(poin.fakta.get("reason"), str):
        sumber += " " + poin.fakta["reason"]
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
    # LLM menulis desimal Indonesia ("4,8", "20,89") sedangkan sumber memakai titik ("4.8"), atau
    # sebaliknya — format, bukan angka karangan (APP-2026-UJI-3: dua percobaan terbuang sia-sia).
    alt = angka.replace(",", ".") if "," in angka else angka.replace(".", ",")
    if alt != angka and re.search(rf"\b{re.escape(alt)}\b", sumber) is not None:
        return True
    return _dekat_dgn_pembulatan(angka.replace(",", "."), sumber)


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


_RE_BOOL_DATA_MATRIKS = re.compile(
    r"\s*\(?\s*kelengkapan data matriks RDTR\s*:\s*(?:True|False)\s*\)?", re.IGNORECASE)
_RE_KLB_PERSEN = re.compile(
    r"(\bKLB\b(?:(?!\bKD[BH]\b)(?:\d[.,]\d|[^.\n%])){0,45}?\d+(?:[.,]\d+)?)\s*%")


def _bersihkan_label_teknis(teks: str | None) -> str | None:
    """Ganti nama label teknis yang tersalin dari prompt dgn padanan Bahasa Indonesia biasa.
    Label tak dikenal tetap dinormalkan (huruf kecil, garis bawah -> spasi) supaya label BARU yang
    ditambahkan ke prompt nanti tidak diam-diam bocor lagi."""
    if not teks:
        return teks
    teks = _RE_LABEL_TEKNIS.sub(
        lambda m: _LABEL_TEKNIS.get(m.group(0), m.group(0).lower().replace("_", " ")), teks
    )
    # "FALLBACK_DATA_KOSONG: False" -> "kelengkapan data matriks RDTR: False": nilai Boolean mentah dan mekanisme
    # internal tidak untuk petugas (APP-2026-0022). Seluruh frasa dibuang.
    teks = _RE_BOOL_DATA_MATRIKS.sub("", teks)
    # KLB adalah rasio, bukan persen: "KLB maksimum 1 %" -> "KLB maksimum 1".
    return _RE_KLB_PERSEN.sub(lambda m: m.group(1), teks)


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

    # Cek #3 — meta.caveats WAJIB muncul di disclaimer.
    # Item permintaan user (2026-10-02): kalimat "Tingkat kepercayaan data: rendah/sedang/tinggi."
    # SENGAJA DIHAPUS dari output (sebelumnya dirakit dari meta.data_confidence_keseluruhan di sini)
    # — disclaimer seperti ini membuat sistem terkesan ragu pada datanya sendiri di mata reviewer,
    # padahal label itu TIDAK pernah mengubah apa pun secara struktural (status/target/verdict tetap
    # sama persis tak peduli label confidence-nya). meta.caveats (free-text spesifik dari BE, mis.
    # "Data ITBX sebagian estimasi") TETAP tampil — itu catatan substantif, bukan skor generik.
    meta = assessment.meta
    if meta:
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
        masalah_konsistensi = (cek_konsistensi_poin_intensitas(poin) if poin.zona_info is not None
                               else cek_konsistensi_intensitas(assessment))
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


_RE_TITIK_KOMA_DAFTAR = re.compile(r"\s*;\s*(?=\d+[.)]\s)")
_RE_TITIK_KOMA = re.compile(r"\s*;\s*")


_RE_REFERENSI_PENUTUP = re.compile(r"[ \t]*Referensi\s*:[^\n]*")
_RE_BATAS_KALIMAT = re.compile(r"[.!?]\s+[A-Z]")


def _buang_referensi_inline(teks: str) -> str:
    """Daftar "Referensi: ..." di dalam narasi duplikat field `sitasi` dan memuat rujukan yg tak
    relevan (APP-2026-0095: ITBX menyebut "Lampiran VI ketentuan intensitas"). Dibuang hanya bila
    ia kalimat PENUTUP barisnya (tak ada batas kalimat di dalamnya), supaya kalimat lain tak ikut
    terhapus."""
    def _ganti(m: re.Match) -> str:
        return m.group(0) if _RE_BATAS_KALIMAT.search(m.group(0)) else ""
    return _RE_REFERENSI_PENUTUP.sub(_ganti, teks).rstrip()


def _perbaiki_titik_koma(teks: str) -> str:
    """Titik koma adalah gaya, bukan fakta: menolak lalu meregenerasi seluruh poin untuk satu ';'
    membuang percobaan (APP-2026-0095: ITBX gagal 3x, percobaan terakhir hanya karena ';' di
    daftar bernomor). Perbaiki deterministik: '; 2.' -> '. 2.', selainnya ';' -> ','."""
    teks = _RE_TITIK_KOMA_DAFTAR.sub(". ", teks)
    return _RE_TITIK_KOMA.sub(", ", teks)


_RE_SARAN_TANPA_TINDAKAN = re.compile(
    r"tidak\s+(?:diperlukan|perlu|memerlukan)\s+(?:tindakan|langkah|mitigasi)|"
    r"tanpa\s+(?:tindakan|mitigasi)\s+(?:khusus|tambahan|lanjutan)",
    re.IGNORECASE,
)


_RE_KLAIM_RESAPAN = re.compile(
    r"zero\s+delta\s+q|pasal\s+53|"
    r"berada\s+(?:di|pada|dalam)\s+(?:kawasan|zona|daerah|area|wilayah)\s+resapan",
    re.IGNORECASE,
)
# Kalimat yang MENYATAKAN ketentuan itu tak berlaku boleh menyebut Pasal 53 / zero delta Q
# ("Ketentuan khusus kawasan resapan air (Pasal 53) tidak berlaku karena lokasi di luar kawasan
# resapan") — itu informatif bagi petugas. Yang dilarang: memakainya sbg ketentuan yang berlaku.
_RE_PENYATAAN_TIDAK_BERLAKU = re.compile(
    r"tidak\s+(?:berlaku|mengikat|diwajibkan|termasuk|berada)|"
    r"bukan\s+(?:merupakan\s+)?(?:kawasan|kewajiban)|di\s+luar\s+(?:kawasan|blok)",
    re.IGNORECASE,
)
_RE_PISAH_KALIMAT = re.compile(r"(?<=[.!?])\s+|\n+")


_RE_PERSEN_KDB_KDH = re.compile(r"\b(KDB|KDH)\b[^.\d]{0,40}?(\d+(?:[.,]\d+)?)\s*%")


def _cek_persen_intensitas_di_dampak(poin_output: PoinOutput, poin: PoinKonteks) -> list[str]:
    """Poin dampak boleh menyebut persen KDB/KDH HANYA bila itu angka target mitigasi dampak (atau
    usulan pemohon). Persen lain (mis. ambang intensitas 60%) berarti LLM mencampur dua konsep
    (APP-2026-UJI-4: "batas KDB maksimum 60%" di poin dampak yang targetnya 75%)."""
    if poin.poin_id != "dampak" or poin.status == "Tidak Dinilai":
        return []
    mit = poin.fakta.get("mitigasi") or {}
    penyesuaian = (poin.fakta.get("target_mitigasi") or {}).get("penyesuaian_lahan") or {}
    pemeliharaan = mit.get("pemeliharaan") or {}
    boleh = set()
    tm = poin.fakta.get("target_mitigasi") or {}
    for v in (penyesuaian.get("kdb_maks_persen"), penyesuaian.get("kdh_min_persen"),
              pemeliharaan.get("kdb_persen"), pemeliharaan.get("kdh_persen"),
              tm.get("kdb_usulan_persen"), tm.get("kdh_usulan_persen")):
        if v is not None:
            boleh.add(round(float(v), 2))
    teks = f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang}"
    for m in _RE_PERSEN_KDB_KDH.finditer(teks):
        angka = round(float(m.group(2).replace(",", ".")), 2)
        if angka not in boleh:
            return [
                f"poin dampak menyebut {m.group(1)} {m.group(2)}%, padahal itu bukan target mitigasi dampak "
                "(kemungkinan ambang intensitas ruang). Persen KDB/KDH di poin dampak hanya boleh target "
                "mitigasi dampak dari fakta, jangan dicampur dengan ketentuan intensitas."
            ]
    return []


_RE_SEBAB_DAMPAK_LOKASI = re.compile(
    r"(?:dampak|kategori)[^.\n]{0,80}(?:disebabkan|diakibatkan|dikarenakan|akibat\s+dari)[^.\n]{0,60}"
    r"(?:lokasi|berada|terletak)[^.\n]{0,40}resapan",
    re.IGNORECASE,
)
_RE_KEHARUSAN = re.compile(r"mengharuskan|diharuskan|diwajibkan|wajib|menuntut|\bharus\b", re.IGNORECASE)
_RE_FASILITAS_RESAPAN = re.compile(r"\bsumur\b|kolam\s+resapan", re.IGNORECASE)
_RE_SIFAT_OPSIONAL = re.compile(
    r"rekomendasi|disarankan|dianjurkan|opsi|alternatif|praktik\s+baik|bukan\s+kewajiban|"
    r"tidak\s+diwajibkan|\bdapat\b|\bboleh\b",
    re.IGNORECASE,
)
_RE_MEMANFAATKAN_PASAL53 = re.compile(
    r"memanfaatkan[^.\n]{0,40}pasal\s+53|pasal\s+53[^.\n]{0,40}memanfaatkan", re.IGNORECASE
)


def _cek_penjelasan_dampak(poin_output: PoinOutput, poin: PoinKonteks) -> list[str]:
    """APP-2026-UJI-2/-4: (a) sebab dampak dibalik jadi "disebabkan oleh lokasi di kawasan resapan";
    (b) di lokasi non-resapan, penjelasan menyebut sumur/kolam resapan sebagai keharusan padahal saran
    menyebutnya rekomendasi; (c) "tidak dapat memanfaatkan Pasal 53"."""
    if poin.poin_id != "dampak" or poin.status == "Tidak Dinilai":
        return []
    teks = f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang}"
    masalah = []
    if _RE_SEBAB_DAMPAK_LOKASI.search(teks):
        masalah.append(
            "penjelasan dampak menyebut kategori dampak 'disebabkan' oleh lokasi di kawasan resapan air. "
            "Sebab kategori dampak adalah indeks limpasan (luas bangunan dan RTH usulan), bukan lokasi."
        )
    if poin.kawasan_resapan is False:
        for kalimat in _RE_PISAH_KALIMAT.split(teks):
            if (_RE_FASILITAS_RESAPAN.search(kalimat) and _RE_KEHARUSAN.search(kalimat)
                    and not _RE_SIFAT_OPSIONAL.search(kalimat)):
                masalah.append(
                    "penjelasan dampak menyebut sumur/kolam resapan sebagai keharusan, padahal lokasi bukan "
                    "kawasan resapan air. Tulis sebagai rekomendasi atau opsi mitigasi, bukan kewajiban."
                )
                break
        if _RE_MEMANFAATKAN_PASAL53.search(teks):
            masalah.append(
                "jangan menulis lokasi 'tidak dapat memanfaatkan' Pasal 53. Tulis bahwa ketentuan itu tidak "
                "berlaku karena lokasi di luar kawasan resapan air."
            )
    return masalah


_RE_INTENSITAS_DI_ITBX = re.compile(r"\b(?:KDB|KLB|KDH)\b|koefisien\s+(?:dasar|lantai|daerah)|ambang\s+intensitas",
                                   re.IGNORECASE)


def _teks_poin(poin_output: PoinOutput) -> str:
    return (f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang} "
            f"{poin_output.rekomendasi.saran}")


def _sumber_fakta_itbx(poin: PoinKonteks, chunks: list[Chunk] | None) -> str:
    sumber = " ".join((poin.fakta.get("keterangan_ketentuan") or []) + [str(poin.fakta.get("reason") or "")])
    sumber += " " + " ".join(d.kutipan for d in poin.dasar_hukum)
    return sumber + " " + " ".join(c.teks for c in (chunks or []))


def _cek_itbx_bahas_intensitas(poin_output: PoinOutput, poin: PoinKonteks) -> list[str]:
    """APP-2026-UJI-2: ITBX menyuruh "KDB tidak melebihi 60%" padahal lokasi resapan berbatas 50%
    (60% diambil dari rentang "50-60%" di tabel yg salah dikutip). Angka batas intensitas milik poin
    intensitas; di ITBX ia hanya sumber kontradiksi yang menyebar ke narasi dan kesimpulan."""
    if poin.poin_id != "itbx":
        return []
    sumber = " ".join((poin.fakta.get("keterangan_ketentuan") or []) + [str(poin.fakta.get("reason") or "")])
    if _RE_INTENSITAS_DI_ITBX.search(_teks_poin(poin_output)) and not _RE_INTENSITAS_DI_ITBX.search(sumber):
        return [
            "poin ITBX membahas KDB/KLB/KDH atau ambang intensitas. Itu milik poin intensitas dan bisa "
            "bertentangan dengannya. Bahas HANYA klasifikasi kegiatan dan syaratnya."
        ]
    return []


_RE_KLAIM_SUDAH_TERPENUHI = re.compile(r"(?:sudah|telah)\s+(?:memenuhi|terpenuhi)", re.IGNORECASE)
_RE_KLAIM_MEMENUHI_SYARAT = re.compile(
    r"memenuhi\s+(?:syarat|batas|ketentuan)\s+(?:luas|maksimal|20)", re.IGNORECASE)
_RE_KALIMAT_BERSYARAT = re.compile(
    r"\bharus\b|\bagar\b|asalkan|selama|\bjika\b|apabila|\bperlu\b|\bwajib\b|dengan\s+syarat|supaya|"
    r"\bhingga\b|\bbila\b|\bsebelum\b|\bdapat\s+diloloskan\s+bila",
    re.IGNORECASE,
)
_RE_OTOMATIS = re.compile(r"secara\s+otomatis|diloloskan\s+otomatis|otomatis\s+oleh\s+sistem", re.IGNORECASE)


def _cek_itbx_klaim_terpenuhi(poin_output: PoinOutput, poin: PoinKonteks) -> list[str]:
    """APP-2026-UJI-3/-4: ITBX menulis pemohon "memenuhi"/"telah memenuhi" batas 20% luas blok dan
    "diloloskan otomatis oleh sistem". Payload tidak memuat rasio itu, jadi tak ada yang bisa
    dinyatakan terpenuhi; dan keputusan ada pada petugas, bukan sistem."""
    if poin.poin_id != "itbx":
        return []
    teks = _teks_poin(poin_output)
    masalah = []
    if _RE_OTOMATIS.search(teks):
        masalah.append("jangan menyebut kegiatan 'diloloskan otomatis': keputusan ada pada petugas.")
    if poin.status in ("B", "T", "TB"):
        for kalimat in _RE_PISAH_KALIMAT.split(teks):
            if _RE_KLAIM_SUDAH_TERPENUHI.search(kalimat) or (
                    _RE_KLAIM_MEMENUHI_SYARAT.search(kalimat) and not _RE_KALIMAT_BERSYARAT.search(kalimat)):
                masalah.append(
                    "poin ITBX bersyarat menyatakan syarat 'sudah/telah memenuhi'. Data tidak memuat bukti "
                    "syarat itu terpenuhi: tulis sebagai kewajiban yang HARUS dipenuhi pemohon."
                )
                break
    return masalah


_POLA_KEWAJIBAN_KARANGAN = (
    re.compile(r"(?:menambah|tambah)\s+blok", re.IGNORECASE),
    re.compile(r"(?:menyertakan|sertakan|melampirkan|lampirkan|menyiapkan|siapkan)\s+"
               r"(?:dokumen|berkas|perhitungan|bukti)", re.IGNORECASE),
    re.compile(r"dokumen\s+(?:perhitungan|pendukung)|berkas\s+permohonan", re.IGNORECASE),
)


def _cek_kewajiban_administratif_karangan(poin_output: PoinOutput, poin: PoinKonteks,
                                          chunks: list[Chunk] | None) -> list[str]:
    """APP-2026-UJI-1/-2/-4: "menyertakan dokumen perhitungan luas blok", "menambah blok", "dokumen
    pendukung (rencana tata ruang, perhitungan KDB)" tak ada di dokumen mana pun. Sah hanya bila
    frasanya tertulis di fakta atau pasal yang disodorkan."""
    teks = _teks_poin(poin_output)
    sumber = _sumber_fakta_itbx(poin, chunks)
    for pola in _POLA_KEWAJIBAN_KARANGAN:
        m = pola.search(teks)
        if m and not pola.search(sumber):
            return [
                f"saran memuat kewajiban administratif yang tak tertulis di fakta/pasal ({m.group(0)!r}). "
                "DILARANG menambah kewajiban dokumen, perhitungan, berkas, atau 'menambah blok'."
            ]
    return []


_RE_SETARA_KELIRU = re.compile(r"setara\s+dengan\s+kebutuhan", re.IGNORECASE)
_RE_ISTILAH_ITBX = re.compile(r"\bbersyarat\b", re.IGNORECASE)
_RE_LAMPIRAN_VB = re.compile(r"lampiran\s+v\.?\s*b\b", re.IGNORECASE)
_RE_ANGKA_BIASA = re.compile(r"\d+(?:[.,]\d+)?")


def _cek_intensitas_dan_istilah(poin_output: PoinOutput, poin: PoinKonteks) -> list[str]:
    """APP-2026-UJI-1/-4: (a) intensitas "memenuhi syarat" dilabeli "Bersyarat" (istilah ITBX);
    (b) "KDB 83,57% setara dengan kebutuhan lantai dasar maksimal 861,6 m2" menyamakan usulan dengan batas."""
    if poin.poin_id != "intensitas":
        return []
    teks = _teks_poin(poin_output)
    masalah = []
    if _RE_ISTILAH_ITBX.search(teks):
        masalah.append(
            "kata 'Bersyarat' adalah istilah ITBX. Status intensitas hanya 'memenuhi syarat' atau "
            "'melampaui batas'."
        )
    if _RE_SETARA_KELIRU.search(teks):
        masalah.append(
            "jangan menulis persentase usulan 'setara dengan kebutuhan' luas batas. Tulis usulan dan batas "
            "sebagai dua hal terpisah (usulan X m2 dengan KDB Y%, batas maksimal Z m2 dengan KDB W%)."
        )
    return masalah


def _cek_atribusi_lampiran_vb(poin_output: PoinOutput, chunks: list[Chunk] | None) -> list[str]:
    """APP-2026-UJI-1: syarat 20% luas blok ditulis "menurut Lampiran V.B", padahal berasal dari keterangan
    ketentuan BE (teks V.B yang dikutip tidak memuat angka itu). Angka pada kalimat yang menyebut Lampiran
    V.B harus ada di teks Lampiran V.B yang disodorkan."""
    vb = " ".join(c.teks for c in (chunks or []) if "-vb-" in (c.id or "")
                  or (c.teks or "").lstrip().startswith("Lampiran V.B"))
    if not vb:
        return []
    for kalimat in _RE_PISAH_KALIMAT.split(_teks_poin(poin_output)):
        if not _RE_LAMPIRAN_VB.search(kalimat):
            continue
        kalimat_tanpa_rujukan = _RE_LAMPIRAN_VB.sub(" ", kalimat)
        for angka in _RE_ANGKA_BIASA.findall(kalimat_tanpa_rujukan):
            # `(?!\d|[.,]\d)`: "20" TIDAK boleh dianggap ada hanya karena teks memuat "20.000".
            if re.search(rf"(?<![\d.,]){re.escape(angka)}(?!\d|[.,]\d)", vb) is None:
                return [
                    f"kalimat merujuk 'Lampiran V.B' tetapi angka {angka!r} tidak ada di teks Lampiran V.B. "
                    "Syarat dari Keterangan Ketentuan berasal dari data back-end, jangan diatribusikan ke Lampiran V.B."
                ]
    return []


def _perbaiki_bold(teks: str) -> str:
    """`**` yang tidak berpasangan (jumlah ganjil) merusak tampilan di FE (APP-2026-UJI-1: "**" menggantung
    di akhir kalimat). Buang yang terakhir."""
    if teks.count("**") % 2 == 1:
        i = teks.rfind("**")
        teks = (teks[:i] + teks[i + 2:]).rstrip()
    return teks


_POLA_SYARAT_KELOMPOK_LAIN = (
    ("jam operasional", re.compile(r"jam\s+(?:operasional|buka)|\bpukul\s+22|\b22[.:]00", re.IGNORECASE)),
    ("dokumen lingkungan", re.compile(r"dokumen\s+(?:perizinan\s+)?lingkungan", re.IGNORECASE)),
    ("pengelolaan limbah", re.compile(r"pengelolaan\s+(?:limbah|sampah)|limbah", re.IGNORECASE)),
    ("tidak mencemari", re.compile(r"mencemari|pencemaran", re.IGNORECASE)),
    ("peternakan", re.compile(r"peternakan|ternak", re.IGNORECASE)),
)


def _cek_syarat_kelompok_lain(poin_output: PoinOutput, poin: PoinKonteks) -> list[str]:
    """APP-2026-UJI: ITBX akomodasi mengutip jam operasional 22.00, dokumen lingkungan, dan "tidak
    mencemari" — syarat kelompok jasa lain/pertanian dari Lampiran V.B yang sama. Syarat sah hanya bila
    tertulis di fakta kegiatan pemohon (keterangan_ketentuan/reason BE)."""
    if poin.poin_id != "itbx":
        return []
    sumber = " ".join((poin.fakta.get("keterangan_ketentuan") or []) + [str(poin.fakta.get("reason") or "")])
    teks = f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang} {poin_output.rekomendasi.saran}"
    asing = [nama for nama, pola in _POLA_SYARAT_KELOMPOK_LAIN
             if pola.search(teks) and not pola.search(sumber)]
    if asing:
        return [
            "syarat " + ", ".join(asing) + " tidak tercantum pada ketentuan kegiatan pemohon (itu syarat "
            "kelompok kegiatan lain di Lampiran V.B). Sebut hanya syarat yang ada di Keterangan Ketentuan."
        ]
    return []


_RE_PROSEDUR_KARANGAN = re.compile(
    r"(?:mengajukan|ajukan|pengajuan)\s+(?:permohonan\s+|usulan\s+)?(?:penyesuaian|perubahan|pengecualian|"
    r"dispensasi|izin\s+khusus|zona|sub-?zona)|permohonan\s+(?:penyesuaian|perubahan|pengecualian|dispensasi)|"
    r"izin\s+khusus|dispensasi",
    re.IGNORECASE,
)
_PENANDA_PROSEDUR_DI_SUMBER = ("dispensasi", "izin khusus", "pengecualian", "permohonan penyesuaian", "perubahan zona")


def _cek_prosedur_karangan(poin_output: PoinOutput, poin: PoinKonteks, chunks: list[Chunk] | None) -> list[str]:
    """APP-2026-UJI-2: saran "ajukan permohonan penyesuaian intensitas ruang" tak ada di dokumen mana pun.
    Prosedur administratif hanya boleh bila tertulis di fakta atau teks pasal yang disodorkan."""
    teks = f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang} {poin_output.rekomendasi.saran}"
    if not _RE_PROSEDUR_KARANGAN.search(teks):
        return []
    sumber = " ".join((poin.fakta.get("keterangan_ketentuan") or []) + [str(poin.fakta.get("reason") or "")])
    sumber += " " + " ".join(c.teks for c in (chunks or []))
    sumber += " " + " ".join(d.kutipan for d in poin.dasar_hukum)
    if any(p in sumber.lower() for p in _PENANDA_PROSEDUR_DI_SUMBER):
        return []
    return [
        "saran memuat prosedur administratif (mengajukan permohonan penyesuaian/pengecualian/dispensasi/izin "
        "khusus/perubahan zona) yang tidak tertulis di fakta maupun pasal. DILARANG mengarang prosedur."
    ]


_KATA_UMUM_KEGIATAN = frozenset({"lainnya", "lain", "yang", "untuk", "dengan", "dalam", "serta", "atau"})


def _pastikan_tipe_bangunan(poin_output: PoinOutput, poin: PoinKonteks) -> PoinOutput:
    """Rapat 2026-10-16: penjelasan dampak hidrologi harus membawa tipe bangunan/kegiatan yang diusulkan
    (ia menentukan koefisien limpasan atap). Bila LLM lupa menyebutnya, kalimat deterministik ditambahkan
    ke akhir reasoning_panjang. Diperbaiki, BUKAN ditolak: menolak seluruh penjelasan hanya karena satu nama
    tidak disebut akan membakar percobaan dan berujung low_confidence."""
    if poin.poin_id != "dampak" or poin.status == "Tidak Dinilai":
        return poin_output
    tipe = poin.fakta.get("tipe_bangunan") or {}
    kegiatan = (tipe.get("kegiatan") or "").strip()
    if not kegiatan:
        return poin_output
    kata = [k for k in re.findall(r"[A-Za-z]{4,}", kegiatan.lower()) if k not in _KATA_UMUM_KEGIATAN]
    teks = f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang}".lower()
    if not kata or any(k in teks for k in kata):
        return poin_output
    kalimat = f"Tipe bangunan yang diusulkan adalah {kegiatan}"
    if tipe.get("kelas_atap"):
        kalimat += f" (kelas bangunan: {tipe['kelas_atap']})"
    kalimat += ", yang memengaruhi koefisien limpasan atap pada perhitungan dampak."
    return poin_output.model_copy(update={
        "reasoning_panjang": f"{poin_output.reasoning_panjang.rstrip()} {kalimat}"})


def _cek_klaim_resapan(poin_output: PoinOutput, poin: PoinKonteks) -> list[str]:
    """Lokasi yang BE nyatakan BUKAN kawasan resapan air tak boleh dijelaskan memakai ketentuan
    kawasan resapan (APP-2026-UJI-4)."""
    if poin.poin_id != "dampak" or poin.kawasan_resapan is not False:
        return []
    teks = f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang} {poin_output.rekomendasi.saran}"
    for kalimat in _RE_PISAH_KALIMAT.split(teks):
        if _RE_KLAIM_RESAPAN.search(kalimat) and not _RE_PENYATAAN_TIDAK_BERLAKU.search(kalimat):
            return [
                "narasi dampak memakai ketentuan/klaim kawasan resapan air (zero delta Q, Pasal 53, atau "
                "'berada di kawasan resapan') sebagai ketentuan yang berlaku, padahal lokasi BUKAN kawasan "
                "resapan air — DILARANG. Boleh menyebut bahwa ketentuan itu TIDAK berlaku. Sumur atau "
                "kolam resapan boleh disebut sebagai rekomendasi, bukan kewajiban kawasan resapan."
            ]
    return []


_RE_KLAIM_KOEFISIEN_BEDA = re.compile(
    # Hanya klaim perbandingan arah yang tak ada di fakta ("koefisien limpasan atap lebih tinggi dibanding ...").
    # "jenis bangunan memengaruhi koefisien limpasan sehingga dampaknya dapat berbeda" justru diminta prompt.
    r"koefisien\s+limpasan[^.]{0,60}lebih\s+(?:tinggi|besar|rendah|kecil)|"
    r"lebih\s+(?:tinggi|besar|rendah|kecil)[^.]{0,40}koefisien\s+limpasan",
    re.IGNORECASE,
)
_RE_HURUF_D_53 = re.compile(r"(?:ayat\s*\(?3\)?\s*)?huruf\s+d\b", re.IGNORECASE)
_RE_LAHAN_SUDAH_ADA = re.compile(r"sudah\s+ada|telah\s+ada|terbangun|eksisting", re.IGNORECASE)
_RE_KLAIM_DAMPAK_MEMENUHI = re.compile(
    # Klaim bahwa pemohon SUDAH memenuhi. "Untuk memenuhi ketentuan tersebut, pemohon harus ..." adalah
    # klausa tujuan, bukan klaim, jadi tidak dicocokkan.
    r"(?:sudah|telah)\s+(?:memenuhi|sesuai)",
    re.IGNORECASE,
)


def _cek_klaim_dampak_tanpa_dasar(poin_output: PoinOutput, poin: PoinKonteks) -> list[str]:
    """Hasil uji nyata 2026-10-16: narasi dampak mengarang 'koefisien limpasan atap berbeda antara jenis
    bangunan', memakai Pasal 53 ayat (3) huruf d sebagai kewajiban bangunan baru (huruf d hanya untuk lahan
    terbangun yang SUDAH ADA), dan menyatakan pemohon 'memenuhi syarat' kawasan resapan."""
    if poin.poin_id != "dampak":
        return []
    teks = f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang} {poin_output.rekomendasi.saran}"
    masalah: list[str] = []
    for kalimat in _RE_PISAH_KALIMAT.split(teks):
        if _RE_KLAIM_KOEFISIEN_BEDA.search(kalimat):
            masalah.append("narasi dampak membandingkan koefisien limpasan antar jenis bangunan, klaim yang "
                           "tidak ada di fakta (SYSTEM_PROMPT aturan #24). Hapus kalimat itu.")
        if _RE_HURUF_D_53.search(kalimat) and not _RE_LAHAN_SUDAH_ADA.search(kalimat):
            masalah.append("Pasal 53 ayat (3) huruf d berlaku untuk lahan terbangun yang sudah ada, bukan "
                           "kewajiban bangunan baru (SYSTEM_PROMPT aturan #24). Sebut 'lahan terbangun yang sudah ada'.")
        if _RE_KLAIM_DAMPAK_MEMENUHI.search(kalimat):
            masalah.append("narasi dampak menyatakan pemohon sudah memenuhi syarat, padahal data tidak memuat "
                           "bukti pemenuhan (SYSTEM_PROMPT aturan #24).")
    return masalah[:2]


def _cek_sumur_dampak_tinggi_non_resapan(poin_output: PoinOutput, poin: PoinKonteks) -> list[str]:
    """Aturan PM 2026-10-18: dampak Tinggi/Sangat Tinggi di LUAR kawasan resapan hanya dimitigasi dengan
    menurunkan KDB dan menaikkan KDH/RTH. Sumur/kolam resapan tak disebut sama sekali (saran dan langkah
    konkret sudah deterministik, jadi yang dicek hanya narasi LLM)."""
    if poin.poin_id != "dampak" or poin.kawasan_resapan is not False or poin.status not in ("Tinggi", "Sangat Tinggi"):
        return []
    teks = f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang}"
    if _RE_FASILITAS_RESAPAN.search(teks):
        return ["narasi dampak Tinggi di luar kawasan resapan menyebut sumur/kolam resapan, padahal mitigasinya "
                "hanya menurunkan KDB dan menaikkan KDH/RTH. Hapus semua penyebutan sumur atau kolam resapan."]
    return []


def _cek_saran_dampak_tanpa_tindakan(poin_output: PoinOutput, poin: PoinKonteks) -> list[str]:
    """Permintaan user 2026-10-14: dampak yang dinilai (termasuk Rendah) WAJIB tetap punya saran
    pencegahan/pemeliharaan konkret, bukan "tidak diperlukan tindakan"."""
    if poin.poin_id != "dampak" or poin.status == "Tidak Dinilai":
        return []
    if _RE_SARAN_TANPA_TINDAKAN.search(poin_output.rekomendasi.saran):
        return [
            "saran dampak berbunyi 'tidak diperlukan tindakan' — DILARANG. Walau dampak rendah, "
            "saran WAJIB berisi langkah pencegahan/pemeliharaan konkret (pertahankan RTH, jaga luas "
            "bangunan, sumur/kolam resapan)."
        ]
    return []


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
            target = pilih_target_dampak_poin(poin.fakta)
            langkah_konkret = bangun_langkah_dampak_poin(poin.fakta)

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

    poin_bersih = poin_bersih.model_copy(update={
        "reasoning_pendek": _perbaiki_bold(_perbaiki_titik_koma(poin_bersih.reasoning_pendek)),
        "reasoning_panjang": _perbaiki_bold(
            _perbaiki_titik_koma(_buang_referensi_inline(poin_bersih.reasoning_panjang))),
        "rekomendasi": poin_bersih.rekomendasi.model_copy(update={
            "saran": _perbaiki_bold(_perbaiki_titik_koma(poin_bersih.rekomendasi.saran)),
            "disclaimer": (
                _perbaiki_titik_koma(poin_bersih.rekomendasi.disclaimer)
                if poin_bersih.rekomendasi.disclaimer else poin_bersih.rekomendasi.disclaimer
            ),
        }),
    })

    poin_bersih = _pastikan_tipe_bangunan(poin_bersih, poin)

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
    masalah.extend(_cek_saran_dampak_tanpa_tindakan(poin_bersih, poin))
    masalah.extend(_cek_klaim_resapan(poin_bersih, poin))
    masalah.extend(_cek_klaim_dampak_tanpa_dasar(poin_bersih, poin))
    masalah.extend(_cek_sumur_dampak_tinggi_non_resapan(poin_bersih, poin))
    masalah.extend(_cek_persen_intensitas_di_dampak(poin_bersih, poin))
    masalah.extend(_cek_penjelasan_dampak(poin_bersih, poin))
    masalah.extend(_cek_syarat_kelompok_lain(poin_bersih, poin))
    masalah.extend(_cek_itbx_bahas_intensitas(poin_bersih, poin))
    masalah.extend(_cek_itbx_klaim_terpenuhi(poin_bersih, poin))
    masalah.extend(_cek_kewajiban_administratif_karangan(poin_bersih, poin, chunks))
    masalah.extend(_cek_intensitas_dan_istilah(poin_bersih, poin))
    masalah.extend(_cek_atribusi_lampiran_vb(poin_bersih, chunks))
    masalah.extend(_cek_prosedur_karangan(poin_bersih, poin, chunks))

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

    # Satu entri per percobaan yang BENAR-BENAR dijalankan, berurutan:
    #   {"percobaan": 1-based, "sebab": "berhasil"|"guardrail_menolak"|"panggilan_llm_gagal",
    #    "masalah": [label temuan guardrail], "exception": "Type: msg" | None}
    #
    # Kenapa `masalah_terakhir` saja tidak cukup (2026-10-09). Field itu menyimpan percobaan
    # TERAKHIR, jadi begitu sebuah poin lolos di percobaan ke-2 atau ke-3 ia ditimpa jadi `[]` —
    # terbukti di logs/precheck.jsonl 2026-10-02T06:44:22 (poin intensitas & dampak, percobaan=3,
    # berhasil=true, masalah_terakhir=[]): guardrail menolak keluaran LLM mentah dua kali di sana
    # dan alasannya hilang total. Padahal justru itu datanya yang menarik — ia merekam faithfulness
    # MENTAH model sebelum guardrail, dan dengan itu efek teknis guardrail jadi terukur
    # (`eval/metrik_generasi.py::laju_tolak_guardrail`), bukan diklaim.
    #
    # Percobaan yang GAGAL SAMPAI KE LLM (rate limit/timeout) dicatat dgn `masalah: []` dan
    # `exception` terisi, SENGAJA tidak dengan `masalah: [str(exc)]` walau teks itulah yang
    # dikirim balik sbg `catatan_perbaikan`: pembaca yang mengagregasi label `masalah` sedang
    # menghitung temuan guardrail, dan mencampur pesan exception ke sana akan mengarang label
    # ("RateLimitError: 429") yang bukan temuan guardrail sama sekali. Pembedaannya ada di `sebab`.
    riwayat_percobaan: list[dict] = field(default_factory=list)

    def sebab(self) -> str:
        """Satu label kasar utk dihitung agregat: kenapa poin ini jatuh ke low_confidence.

        APP-2026-INNER-01 (2026-10-02, live): ditemukan nyata di VPS — Jina API menolak SEMUA
        panggilan embed (403 "Insufficient account balance") -> `ambil_chunks_pendukung()` raise
        SEBELUM LLM sempat dipanggil sama sekali, tapi label sebelumnya ("panggilan_llm_gagal")
        menyamaratakan dgn kegagalan LLM asli (rate limit/timeout Groq) — menyesatkan diagnosis:
        user mengira ini bug reasoning/LLM, padahal akar masalahnya provider retrieval eksternal
        (tagihan/kuota Jina habis, bukan kode). Dibedakan di sini via penanda pesan
        `_provider_http.py::post_json` ("Panggilan provider '<nama>' gagal...") — SATU-SATUNYA
        sumber exception dgn format persis itu (jina-embed/jina-rerank), beda dari exception LLM
        (RateLimitError/BadRequestError/dst dari SDK Groq)."""
        if self.berhasil:
            return "berhasil"
        if self.exception_terakhir:
            if "Panggilan provider '" in self.exception_terakhir:
                return "retrieval_provider_gagal"
            if self.exception_terakhir.startswith("LayananTidakTersediaError"):
                return "llm_tidak_tersedia"
            if self.exception_terakhir.startswith("BatasWaktuTerlampaui"):
                return "melewati_batas_waktu"
            # percobaan == 0: gagal SEBELUM loop LLM mulai (ambil_chunks_pendukung raise selain
            # error provider, mis. Postgres/pgvector mati: ConnectionTimeout ke localhost:5432,
            # 2026-10-14). Bukan salah LLM, jangan dilabeli begitu.
            if self.percobaan == 0:
                return "retrieval_db_gagal"
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
    # APP-2026-0095: Groq kadang membalas 400 json_validate_failed dgn keluaran KOSONG — gangguan
    # sesaat, bukan salah isi teks. Satu kegagalan panggilan seperti itu diberi percobaan
    # TAMBAHAN di luar jatah guardrail (maks 1 per poin, supaya tak berputar tanpa batas).
    # PIIDetectedError dikecualikan: mengulang tak akan mengubah hasilnya.
    sisa_jatah = max_retry + 1
    bonus_tersedia = 1
    percobaan = -1
    while sisa_jatah > 0:
        percobaan += 1
        if batas_waktu.habis(_MIN_SISA_PERCOBAAN_S):
            diagnosa.exception_terakhir = (
                diagnosa.exception_terakhir
                or f"BatasWaktuTerlampaui: anggaran waktu habis sebelum percobaan ke-{percobaan + 1}")
            logger.warning("Anggaran waktu habis, percobaan dihentikan utk poin %r", poin.poin_id)
            break
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
            diagnosa.riwayat_percobaan.append({
                "percobaan": percobaan + 1,
                "sebab": "panggilan_llm_gagal",
                "masalah": [],  # lihat DiagnosaPoin.riwayat_percobaan: pesan exception bukan temuan guardrail
                "exception": diagnosa.exception_terakhir,
            })
            if isinstance(exc, (LayananTidakTersediaError, BatasWaktuTerlampaui)):
                # Semua kunci mati: mengulang tak mungkin berhasil sampai operator memperbaiki akun.
                logger.error("LLM tidak tersedia, percobaan dihentikan utk poin %r: %s", poin.poin_id, exc)
                break
            if bonus_tersedia and type(exc).__name__ != "PIIDetectedError":
                bonus_tersedia -= 1  # percobaan ini tak memakan jatah guardrail
            else:
                sisa_jatah -= 1
            continue

        # Percobaan ini sampai ke LLM & dapat jawaban -> exception percobaan SEBELUMNYA (kalau ada)
        # tak lagi menjadi sebab; jangan sampai salah melabeli "panggilan_llm_gagal".
        diagnosa.exception_terakhir = None
        sisa_jatah -= 1
        poin_bersih, masalah = perbaiki_poin(hasil, poin, chunks, assessment)
        diagnosa.masalah_terakhir = list(masalah)
        diagnosa.teks_ditolak_terakhir = _ringkas_teks_ditolak(poin_bersih) if masalah else None
        # Dicatat SEBELUM cabang sukses di bawah: percobaan yang lolos pun harus punya entri, kalau
        # tidak, penyebut "berapa percobaan pertama yang sampai ke guardrail" jadi tak terhitung.
        diagnosa.riwayat_percobaan.append({
            "percobaan": percobaan + 1,
            "sebab": "guardrail_menolak" if masalah else "berhasil",
            "masalah": list(masalah),
            "exception": None,
        })
        if not masalah:
            diagnosa.berhasil = True
            if percobaan > 0:
                # Lolos TAPI butuh retry — near-miss. Cek guardrail mana yang paling sering menggigit
                # cuma kelihatan dari sini, bukan dari output akhir yang terlihat mulus.
                logger.info(
                    "poin %r lolos guardrail di percobaan ke-%d/%d", poin.poin_id, percobaan + 1, max_retry + 1
                )
            # Sitasi tambahan yang pasti (Pasal 53(3) / Lampiran VI untuk dampak) ditambahkan SETELAH lolos
            # validasi: validasi sitasi hanya menerima chunk dari konteks poin, sedangkan sitasi ini dicari
            # terpisah lewat retriever (lihat generator._lengkapi_sitasi_dampak).
            poin_bersih = poin_bersih.model_copy(update={"sitasi": _lengkapi_sitasi_dampak(
                poin_bersih.sitasi, poin, {c.id: c for c in chunks}, retriever)})
            return poin_bersih, diagnosa

    logger.warning(
        "poin %r jatuh ke low_confidence setelah %d percobaan (sebab=%s, chunk=%d): %s",
        poin.poin_id, diagnosa.percobaan, diagnosa.sebab(), diagnosa.jumlah_chunk,
        "; ".join(diagnosa.masalah_terakhir) or "-",
    )
    hasil_template = _paksa_field_wajib(template_low_confidence(poin), poin, assessment)
    try:
        # Sitasi dampak yang pasti (Pasal 53(3) / Lampiran VI) tetap dilampirkan walau LLM gagal.
        hasil_template = hasil_template.model_copy(update={"sitasi": _lengkapi_sitasi_dampak(
            hasil_template.sitasi, poin, {}, retriever)})
    except Exception:  # noqa: BLE001 — sitasi tambahan opsional
        logger.warning("Sitasi tambahan fallback dampak gagal.", exc_info=True)
    return hasil_template, diagnosa


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
