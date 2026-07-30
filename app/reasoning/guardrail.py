"""Guardrail — cek murah deterministik atas PoinOutput SEBELUM diloloskan, + loop retry + fallback.

CLAUDE.md / docs/Blueprint-Adaptasi-Model-L2-Gate-Impact.md §5: "Guardrail gagal -> regenerasi
terarah (maks 1-2x) -> fallback template + tanda low_confidence. JANGAN loop tak terbatas." Semua
cek di sini murni Python (regex/perbandingan) — TIDAK ADA panggilan LLM.

Dua kategori cek, disengaja dipisah:
- **Paksaan** (`_paksa_field_wajib`): diterapkan ke SEMUA jalur keluar (aman/template/hasil-bersih)
  TANPA memicu retry — fallback ITBX data-kosong, caveat meta, wiring `cek_konsistensi_intensitas`.
  Regenerasi LLM tidak bisa memperbaiki data back-end yang tak konsisten atau caveat yang hilang;
  lebih murah & pasti benar kalau disuntik langsung via kode.
- **Masalah teks** (`perbaiki_poin`): cuma jalan di jalur non-aman (setelah panggilan LLM), memicu
  retry — reasoning/saran kosong, sitasi hilang, invers-skor, konsistensi verdict, angka di narasi.

Verifikasi entailment sitasi/verdict (NLI/LLM) DITUNDA sampai eval membuktikan perlu — lihat stub
`verifikasi_entailment_sitasi` di bawah, tidak dipanggil di alur utama.
"""

import logging
import re

from app.adapter import cek_konsistensi_intensitas
from app.reasoning.calculator import pilih_target_utama_intensitas
from app.reasoning.generator import ambil_chunks_pendukung, apakah_aman, generate_poin
from app.reasoning.templates import template_aman, template_low_confidence
from app.retrieval.base import Chunk, Retriever
from app.schemas import L2Assessment, PoinKonteks, PoinOutput

logger = logging.getLogger(__name__)

CAVEAT_FALLBACK_ITBX = (
    "diloloskan otomatis karena data matriks RDTR kosong, bukan kepatuhan terverifikasi"
)

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
            if frasa in teks:
                masalah.append(
                    f"Reasoning menyiratkan dampak tinggi ('{frasa}') padahal kategori aktual "
                    f"'{kategori}' — cek arah skor invers."
                )
                break
    elif kategori in ("Tinggi", "Sangat Tinggi"):
        for frasa in _FRASA_DAMPAK_RENDAH:
            if frasa in teks:
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
                if frasa in teks:
                    masalah.append(
                        f"Reasoning menyiratkan pelanggaran ('{frasa}') padahal status = MEMENUHI_SYARAT."
                    )
                    break
        elif poin.status == "MELAMPAUI_BATAS":
            for frasa in ("memenuhi seluruh standar", "tidak ada pelanggaran", "sudah sesuai semua"):
                if frasa in teks:
                    masalah.append(
                        f"Reasoning menyiratkan kepatuhan penuh ('{frasa}') padahal status = MELAMPAUI_BATAS."
                    )
                    break

    return masalah


# ---------------------------------------------------------------------------
# Cek #6 — Konsistensi numerik (teks)
# ---------------------------------------------------------------------------


def _angka_terlacak_ke_sumber(angka: str, poin: PoinKonteks) -> bool:
    """Investigasi ITBX APP-2026-6191: Cek #6 versi lama melarang SEMUA angka tanpa pandang sumber
    — menangkap angka ambang yang dikutip verbatim dari `keterangan_ketentuan`/`dasar_hukum` back-end
    (mis. "RTH minimal 20 dari luas persil"), padahal itu FAKTA sah, bukan halusinasi/hitungan LLM.

    Provenance check: angka BOLEH muncul di narasi HANYA kalau tercantum verbatim (word-boundary) di
    fakta sumber poin ini. `intensitas`/`dampak` tidak punya `keterangan_ketentuan` & `dasar_hukum`
    selalu kosong utk keduanya (lihat app/adapter.py) — jadi otomatis TETAP seketat sebelumnya (angka
    apa pun di situ tidak pernah terlacak ke sumber, karena memang tidak boleh ada). Ini MEMPERKETAT
    presisi cek, bukan melonggarkan: angka yang tak bisa dibuktikan asalnya tetap ditolak.
    """
    sumber = " ".join(poin.fakta.get("keterangan_ketentuan") or [])
    sumber += " " + " ".join(d.kutipan for d in poin.dasar_hukum)
    return re.search(rf"\b{re.escape(angka)}\b", sumber) is not None


def _cek_konsistensi_numerik(poin_output: PoinOutput, poin: PoinKonteks) -> list[str]:
    """SYSTEM_PROMPT (prompts.py) aturan #8 melarang LLM menyebut angka yang TIDAK bisa dilacak ke
    fakta sumber (angka final tetap dirakit kode dari calculator/back-end, tidak pernah dari sini).
    Tidak scan `sitasi[].kutipan` — kutipan pasal boleh memuat angka (nomor pasal/ayat) yang sah.
    """
    teks = f"{poin_output.reasoning_pendek} {poin_output.reasoning_panjang} {poin_output.rekomendasi.saran}"
    for match in _RE_ANGKA_MENCURIGAKAN.finditer(teks):
        angka = match.group(0)
        if not _angka_terlacak_ke_sumber(angka, poin):
            return [
                f"Reasoning/saran menyebutkan angka {angka!r} yang tidak tercantum di fakta sumber "
                "poin ini — dilarang (SYSTEM_PROMPT aturan #8). Angka harus berasal dari "
                "calculator/back-end via field terpisah, atau dikutip verbatim dari "
                "keterangan_ketentuan/dasar_hukum, bukan dihitung/dikarang LLM."
            ]
    return []


# ---------------------------------------------------------------------------
# Paksaan wajib — cek #2, #3, #4. Diterapkan ke SEMUA jalur keluar, TIDAK memicu retry.
# ---------------------------------------------------------------------------


def _paksa_field_wajib(
    poin_output: PoinOutput, poin: PoinKonteks, assessment: L2Assessment
) -> PoinOutput:
    """Pastikan field yang WAJIB benar terlepas dari LLM/template — TIDAK PERNAH mengoreksi/
    menimpa fakta back-end, hanya memastikan penanda low_confidence & catatan wajib benar-benar
    ada (Faithful, CLAUDE.md/Blueprint §5).
    """
    update: dict = {}
    disclaimer_tambahan: list[str] = []
    teks_sudah_ada = f"{poin_output.reasoning_panjang} {poin_output.rekomendasi.disclaimer or ''}"

    # Cek #2 — ITBX fallback data-kosong: paksa low_confidence + caveat wajib.
    if poin.poin_id == "itbx" and poin.fakta.get("fallback_data_kosong"):
        update["low_confidence"] = True
        if CAVEAT_FALLBACK_ITBX.lower() not in teks_sudah_ada.lower():
            disclaimer_tambahan.append(CAVEAT_FALLBACK_ITBX.capitalize() + ".")

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

    if disclaimer_tambahan:
        existing = poin_output.rekomendasi.disclaimer
        gabungan = " ".join(([existing] if existing else []) + disclaimer_tambahan)
        update["rekomendasi"] = poin_output.rekomendasi.model_copy(update={"disclaimer": gabungan})

    if update:
        poin_output = poin_output.model_copy(update=update)
    return poin_output


# ---------------------------------------------------------------------------
# Masalah teks — hanya jalur non-aman, memicu retry.
# ---------------------------------------------------------------------------


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

    target = None
    if poin.tipe_rekomendasi == "numerik":
        target = pilih_target_utama_intensitas(poin.fakta.get("target") or {})

    poin_bersih = poin_output.model_copy(
        update={
            "status": poin.status,
            "sitasi": sitasi_bersih,
            "rekomendasi": poin_output.rekomendasi.model_copy(
                update={"tipe": poin.tipe_rekomendasi, "target": target}
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
    masalah.extend(_cek_konsistensi_numerik(poin_bersih, poin))

    poin_bersih = _paksa_field_wajib(poin_bersih, poin, assessment)

    return poin_bersih, masalah


def verifikasi_entailment_sitasi(poin: PoinOutput, chunks: list[Chunk]) -> bool:
    """TODO(Fase 3+): verifikasi entailment sitasi/verdict via model NLI kecil atau panggilan LLM.

    DITUNDA sampai eval membuktikan perlu (CLAUDE.md § Guardrail). Tidak dipanggil di alur utama —
    placeholder untuk pengembangan berikutnya.
    """
    return True


def generate_poin_dengan_guardrail(
    poin: PoinKonteks,
    retriever: Retriever,
    assessment: L2Assessment,
    *,
    max_retry: int = 2,
) -> PoinOutput:
    """Entrypoint utama: generate_poin + guardrail + retry terarah + fallback low_confidence."""
    if apakah_aman(poin):
        return _paksa_field_wajib(template_aman(poin), poin, assessment)

    chunks = ambil_chunks_pendukung(poin, retriever)

    masalah: list[str] = []
    for percobaan in range(max_retry + 1):
        catatan = "; ".join(masalah) if percobaan > 0 else None
        suhu = 0.4 if percobaan > 0 else 0.0

        try:
            hasil = generate_poin(
                poin,
                retriever,
                assessment.meta,
                catatan_perbaikan=catatan,
                temperature=suhu,
            )
        except Exception as exc:  # generasi gagal dihitung sebagai percobaan gagal, bukan crash
            masalah = [str(exc)]
            continue

        poin_bersih, masalah = perbaiki_poin(hasil, poin, chunks, assessment)
        if not masalah:
            return poin_bersih

    return _paksa_field_wajib(template_low_confidence(poin), poin, assessment)
