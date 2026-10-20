"""Pemetaan keluaran internal (`OutputL3`, daftar poin datar) ke payload kontrak BE.

Mengikuti "Spesifikasi Kontrak Variabel Output AI Reasoning (L3)" v1.0.0 (2026-10-20): `poin` berupa pohon
dengan dua akar, `gate_hukum` (isi: itbx dan intensitas) dan `gate_hidrologis` (isi: dampak), setiap simpul
punya `dasar_regulasi` dan `rekomendasi` {saran, poin, langkah_konkret, alasan}.

Pemetaan ini murni transformasi bentuk dan format: tidak ada fakta, angka, atau status baru yang dibuat di sini
selain label gabungan akar (`gate_hukum`, `gate_hidrologis`) yang diturunkan dari status simpul anaknya.
Field kaya milik kita yang tidak ada di spesifikasi (sitasi lengkap, `zona`, `low_confidence`,
`langkah_terstruktur`, `catatan_global`) tetap dibawa sebagai tambahan.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.reasoning.assemble import (
    _LABEL_INTENSITAS_RINGKAS,
    _LABEL_ITBX_RINGKAS,
    _butir_saran,
    _id_dasar,
    _ke_bullet,
    _label_zona,
)
from app.reasoning.rekomendasi import KATEGORI_DAMPAK_BERSYARAT
from app.schemas import (
    KesimpulanOutput,
    LangkahKonkretOutput,
    NarasiRekomendasiOutput,
    OutputL3,
    PoinOutput,
    RekomendasiZonaOutput,
    RingkasanDampakOutput,
    RingkasanGateOutput,
    SitasiOutput,
    ZonaInfo,
)

KATEGORI_REGULASI_DAERAH = "Regulasi Daerah"
_PANJANG_DESKRIPSI = 300


class DasarRegulasiBE(BaseModel):
    judul: str
    deskripsi: str
    kategori: str = KATEGORI_REGULASI_DAERAH
    # Tambahan kita: identitas dan teks sitasi terverifikasi (sama dengan `sitasi` di poin).
    citation_id: str | None = None
    dokumen: str | None = None
    pasal: str | None = None
    halaman: int | None = None
    terverifikasi: bool | None = None


class RekomendasiBE(BaseModel):
    tipe: str | None = None
    target: float | str | None = None
    saran: str
    poin: list[str] = []
    langkah_konkret: list[str] = []
    alasan: str = ""
    # Tambahan kita
    disclaimer: str | None = None
    langkah_terstruktur: list[LangkahKonkretOutput] = []


class PoinBE(BaseModel):
    poin_id: str
    kategori: str
    status: str
    reasoning_pendek: str
    reasoning_panjang: str
    dasar_regulasi: list[DasarRegulasiBE] = []
    rekomendasi: RekomendasiBE | None = None
    sub_poin: list["PoinBE"] = []
    # Tambahan kita
    sitasi: list[SitasiOutput] = []
    zona: ZonaInfo | None = None
    low_confidence: bool = False


class AdvisoryL3(BaseModel):
    rekomendasi_sistem: str
    kesimpulan: KesimpulanOutput
    ringkasan_gate: RingkasanGateOutput
    ringkasan_dampak: RingkasanDampakOutput
    narasi_rekomendasi: NarasiRekomendasiOutput
    rekomendasi_per_zona: list[RekomendasiZonaOutput]
    poin: list[PoinBE]
    catatan_global: list[str] = []
    low_confidence_keseluruhan: bool


PoinBE.model_rebuild()

_STATUS_GATE = {"Lolos": "LOLOS", "Lolos Bersyarat": "LOLOS_BERSYARAT",
                "Lolos Terbatas": "LOLOS_TERBATAS", "Tidak Lolos": "TIDAK_LOLOS"}
_ID_LEAF_TUNGGAL = {"itbx": "itbx_evaluasi", "intensitas": "intensitas_evaluasi", "dampak": "dampak"}


# ---------------------------------------------------------------------------
# Simpul daun
# ---------------------------------------------------------------------------

def _dasar_regulasi(sitasi: list[SitasiOutput]) -> list[DasarRegulasiBE]:
    hasil = []
    for s in sitasi:
        judul = s.dokumen if not s.pasal else f"{s.dokumen} - {s.pasal if not s.pasal.isdigit() else 'Pasal ' + s.pasal}"
        kutipan = " ".join((s.kutipan or "").split())
        deskripsi = kutipan if len(kutipan) <= _PANJANG_DESKRIPSI else kutipan[: _PANJANG_DESKRIPSI - 1].rstrip() + "…"
        hasil.append(DasarRegulasiBE(
            judul=judul, deskripsi=deskripsi, citation_id=s.citation_id, dokumen=s.dokumen,
            pasal=s.pasal, halaman=s.halaman, terverifikasi=s.terverifikasi,
        ))
    return hasil


def _rekomendasi_be(p: PoinOutput) -> RekomendasiBE:
    butir = _butir_saran(p.rekomendasi.saran)
    return RekomendasiBE(
        tipe=p.rekomendasi.tipe, target=p.rekomendasi.target,
        saran=_ke_bullet(butir) if len(butir) > 1 else (butir[0] if butir else p.rekomendasi.saran),
        poin=butir, langkah_konkret=[l.deskripsi for l in p.rekomendasi.langkah_konkret],
        alasan=p.reasoning_pendek, disclaimer=p.rekomendasi.disclaimer,
        langkah_terstruktur=list(p.rekomendasi.langkah_konkret),
    )


_LABEL_STATUS_ITBX = {"I": "Diizinkan (I)", "T": "Terbatas (T)", "B": "Bersyarat (B)",
                      "TB": "Terbatas dan Bersyarat (TB)", "X": "Tidak Diizinkan (X)"}


def _status_daun(p: PoinOutput) -> str:
    """Format status simpul daun mengikuti contoh BE: ITBX "Bersyarat (B)", dampak "SEDANG", intensitas
    "MELAMPAUI_BATAS". Status internal (kode asli) tetap dipakai di seluruh pipeline."""
    dasar = _id_dasar(p.poin_id)
    if dasar == "itbx":
        return _LABEL_STATUS_ITBX.get(p.status, p.status)
    if dasar == "dampak":
        return "TIDAK_DINILAI" if p.status == "Tidak Dinilai" else p.status.upper()
    if p.status == "Tidak Dinilai":
        return "TIDAK_DINILAI"
    return p.status


def _daun(p: PoinOutput, banyak_zona: bool) -> PoinBE:
    poin_id = p.poin_id if banyak_zona and p.zona else _ID_LEAF_TUNGGAL.get(_id_dasar(p.poin_id), p.poin_id)
    return PoinBE(
        poin_id=poin_id, kategori=p.kategori, status=_status_daun(p),
        reasoning_pendek=p.reasoning_pendek, reasoning_panjang=p.reasoning_panjang,
        dasar_regulasi=_dasar_regulasi(p.sitasi), rekomendasi=_rekomendasi_be(p),
        sitasi=p.sitasi, zona=p.zona, low_confidence=p.low_confidence,
    )


def _gabung_dasar(anak: list[PoinBE]) -> list[DasarRegulasiBE]:
    """Dasar regulasi simpul induk = gabungan dasar anak-anaknya, tanpa duplikat (kunci: citation_id)."""
    unik: dict[str, DasarRegulasiBE] = {}
    for a in anak:
        for d in a.dasar_regulasi:
            unik.setdefault(d.citation_id or d.judul, d)
    return list(unik.values())


def _gabung_rekomendasi(anak: list[PoinBE], alasan: str) -> RekomendasiBE:
    """Rekomendasi simpul induk: gabungan butir dan langkah anak-anaknya (urut, tanpa duplikat)."""
    butir = list(dict.fromkeys(b for a in anak if a.rekomendasi for b in a.rekomendasi.poin))
    langkah = list(dict.fromkeys(l for a in anak if a.rekomendasi for l in a.rekomendasi.langkah_konkret))
    return RekomendasiBE(
        saran=_ke_bullet(butir) if len(butir) > 1 else (butir[0] if butir else ""),
        poin=butir, langkah_konkret=langkah, alasan=alasan,
    )


# ---------------------------------------------------------------------------
# Akar
# ---------------------------------------------------------------------------

def _status_gate_hukum(output: OutputL3) -> str:
    """Status akar gate_hukum: dari status gate BE, tetapi ITBX X/T tetap menentukan (spesifikasi bagian 4)."""
    itbx = {p.status for p in output.poin if _id_dasar(p.poin_id) == "itbx"}
    if "X" in itbx:
        return "TIDAK_LOLOS"
    if itbx & {"T", "TB"} and output.ringkasan_gate.final_gate_status != "Tidak Lolos":
        return "LOLOS_TERBATAS"
    return _STATUS_GATE.get(output.ringkasan_gate.final_gate_status, output.ringkasan_gate.final_gate_status.upper())


def _klausa_status(output: OutputL3) -> str:
    """Kalimat status ITBX dan intensitas untuk ringkasan_gate (spesifikasi: narasi status ITBX dan kepatuhan intensitas)."""
    bagian = []
    zona = list(dict.fromkeys(_label_zona(p) for p in output.poin if _id_dasar(p.poin_id) in ("itbx", "intensitas")))
    for label in zona:
        itbx = next((p for p in output.poin if _label_zona(p) == label and _id_dasar(p.poin_id) == "itbx"), None)
        inten = next((p for p in output.poin if _label_zona(p) == label and _id_dasar(p.poin_id) == "intensitas"), None)
        isi = []
        if itbx:
            isi.append(f"status ITBX {_LABEL_ITBX_RINGKAS.get(itbx.status, itbx.status)}")
        if inten:
            isi.append(f"intensitas {_LABEL_INTENSITAS_RINGKAS.get(inten.status, inten.status.lower())}")
        if isi:
            teks = ", ".join(isi)
            bagian.append(f"{label}: {teks}" if label else teks[0].upper() + teks[1:])
    return (". ".join(bagian) + ".") if bagian else ""


def _akar_gate_hukum(output: OutputL3, daun: list[tuple[PoinOutput, PoinBE]], banyak_zona: bool,
                     kalimat_gate: str) -> PoinBE:
    gate = [(p, b) for p, b in daun if _id_dasar(p.poin_id) in ("itbx", "intensitas")]
    if banyak_zona:
        anak: list[PoinBE] = []
        for z in output.rekomendasi_per_zona:
            sub = [b for p, b in gate if p.zona and p.zona.kode_zona == z.kode_zona]
            kode = z.kode_zona or "zona"
            anak.append(PoinBE(
                poin_id=f"zona_{kode}",
                kategori=f"Penilaian Tata Ruang: {z.nama_zona}" + (f" ({kode})" if z.kode_zona else ""),
                status=z.status_keputusan or "", reasoning_pendek=z.pertimbangan or z.alasan,
                reasoning_panjang=z.alasan,
                rekomendasi=RekomendasiBE(saran=z.saran, poin=list(z.rekomendasi_poin),
                                          langkah_konkret=[l for b in sub if b.rekomendasi for l in b.rekomendasi.langkah_konkret],
                                          alasan=z.alasan),
                sub_poin=sub, zona=ZonaInfo(nama_zona=z.nama_zona, kode_zona=z.kode_zona,
                                            luas_m2=z.luas_m2, persentase=z.persentase),
                low_confidence=z.low_confidence,
            ))
        leaf_rekom = [b for _, b in gate]
    else:
        anak = [b for _, b in gate]
        leaf_rekom = anak
    panjang = " ".join(
        (f"{_label_zona(p)}: " if banyak_zona and _label_zona(p) else "") + b.reasoning_pendek for p, b in gate)
    return PoinBE(
        poin_id="gate_hukum", kategori="Penilaian Legal Tata Ruang", status=_status_gate_hukum(output),
        reasoning_pendek=kalimat_gate, reasoning_panjang=panjang or kalimat_gate,
        dasar_regulasi=_gabung_dasar(leaf_rekom),
        rekomendasi=_gabung_rekomendasi(leaf_rekom, kalimat_gate), sub_poin=anak,
        sitasi=[], low_confidence=any(b.low_confidence for b in leaf_rekom),
    )


def _akar_gate_hidrologis(output: OutputL3, daun: list[tuple[PoinOutput, PoinBE]]) -> PoinBE | None:
    dampak = [(p, b) for p, b in daun if _id_dasar(p.poin_id) == "dampak"]
    if not dampak:
        return None
    p, b = dampak[0]
    if p.status == "Tidak Dinilai":
        status = "TIDAK_DINILAI"
    elif p.status in KATEGORI_DAMPAK_BERSYARAT:
        status = "PERLU_MITIGASI"
    else:
        status = "MEMENUHI_SYARAT"
    return PoinBE(
        poin_id="gate_hidrologis", kategori="Penilaian Lingkungan (Hidrologi)", status=status,
        reasoning_pendek=b.reasoning_pendek, reasoning_panjang=b.reasoning_panjang,
        dasar_regulasi=_gabung_dasar([b]),
        rekomendasi=_gabung_rekomendasi([b], b.reasoning_pendek), sub_poin=[b],
        low_confidence=b.low_confidence,
    )


# ---------------------------------------------------------------------------
# Titik masuk
# ---------------------------------------------------------------------------

def ke_kontrak_be(output: OutputL3) -> AdvisoryL3:
    """OutputL3 internal -> payload kontrak BE v1.0.0 (pohon `poin`, format saran, dasar_regulasi)."""
    banyak_zona = any(p.zona is not None for p in output.poin)
    daun = [(p, _daun(p, banyak_zona)) for p in output.poin]

    klausa = _klausa_status(output)
    gate_ringkas = output.ringkasan_gate.model_copy(update={
        "kalimat": f"{output.ringkasan_gate.kalimat} {klausa}".strip()})

    akar = [_akar_gate_hukum(output, daun, banyak_zona, gate_ringkas.kalimat)]
    hidro = _akar_gate_hidrologis(output, daun)
    if hidro:
        akar.append(hidro)

    return AdvisoryL3(
        rekomendasi_sistem=output.rekomendasi_sistem, kesimpulan=output.kesimpulan,
        ringkasan_gate=gate_ringkas, ringkasan_dampak=output.ringkasan_dampak,
        narasi_rekomendasi=output.narasi_rekomendasi, rekomendasi_per_zona=output.rekomendasi_per_zona,
        poin=akar, catatan_global=output.catatan_global,
        low_confidence_keseluruhan=output.low_confidence_keseluruhan,
    )
