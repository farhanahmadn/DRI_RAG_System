"""Perbaikan hasil evaluasi 4 payload uji hidrologi (APP-2026-UJI-1..4, 2026-10-15)."""

import json
from pathlib import Path

import pytest

from app.reasoning import llm_client as llm_client_module
from app.reasoning.assemble import _rakit_kesimpulan
from app.reasoning.generator import (
    _berlaku_utk_subzona,
    _sesuaikan_dgn_status_resapan,
    chunk_pengisi_dampak,
    generate_poin,
)
from app.reasoning.guardrail import (
    _angka_terlacak_ke_sumber,
    _cek_penjelasan_dampak,
    _cek_prosedur_karangan,
    _cek_syarat_kelompok_lain,
)
from app.reasoning.llm_client import _normalisasi_karakter
from app.reasoning.prompts import _angka_prompt
from app.retrieval.base import Chunk
from app.retrieval.mock import MockRetriever
from app.schemas import DasarHukum, L2Assessment, PoinKonteks, PoinOutput, RekomendasiOutput

FIXTURES = Path(__file__).parent / "fixtures"


def _poin(**kw) -> PoinKonteks:
    d = dict(poin_id="dampak", kategori="Dampak", tipe_rekomendasi="numerik-mitigasi", status="Tinggi",
             fakta={"dinilai": True})
    d.update(kw)
    return PoinKonteks(**d)


def _out(poin_id="dampak", panjang="x", saran="saran", pendek="p") -> PoinOutput:
    return PoinOutput(poin_id=poin_id, kategori="K", status="Tinggi", reasoning_pendek=pendek,
                      reasoning_panjang=panjang, sitasi=[],
                      rekomendasi=RekomendasiOutput(tipe="kategorikal", saran=saran), low_confidence=False)


def _chunk(cid, pasal, teks, level="ayat"):
    return Chunk(id=cid, level=level, dokumen="Perbup", pasal=pasal, teks=teks)


# ---------------------------------------------------------------- A. sebab dampak
class TestSebabDampak:
    def test_sebab_dibalik_ditolak(self):
        teks = "**Tinggi** dampak lingkungan hidrologi disebabkan oleh lokasi di kawasan resapan air."
        for status in (True, False, None):
            m = _cek_penjelasan_dampak(_out(panjang=teks), _poin(kawasan_resapan=status))
            assert any("disebabkan" in x for x in m), status

    def test_kewajiban_karena_lokasi_resapan_bukan_sebab_dampak(self):
        teks = ("Kategori dampak rendah menunjukkan tidak ada kebutuhan mitigasi korektif, namun karena lokasi "
                "berada di kawasan resapan air, pemohon harus mematuhi ketentuan khusus.")
        assert not _cek_penjelasan_dampak(_out(panjang=teks), _poin(kawasan_resapan=True))


# ---------------------------------------------------------------- B. keharusan sumur di non-resapan
class TestKeharusanSumurNonResapan:
    def test_mengharuskan_sumur_ditolak(self):
        teks = ("Rincian mitigasi mengharuskan luas bangunan maksimal 1077 m², RTH minimal 359 m², serta "
                "sumur atau kolam resapan dengan dimensi minimum 0.62 m³, sesuai ketentuan RDTR.")
        m = _cek_penjelasan_dampak(_out(panjang=teks), _poin(kawasan_resapan=False))
        assert any("keharusan" in x for x in m)

    def test_rekomendasi_sumur_diizinkan(self):
        teks = "Sumur atau kolam resapan disarankan sebagai opsi mitigasi, bukan kewajiban."
        assert not _cek_penjelasan_dampak(_out(panjang=teks), _poin(kawasan_resapan=False))

    def test_di_kawasan_resapan_keharusan_boleh(self):
        teks = "Pemohon wajib menyediakan sumur atau kolam resapan sesuai Pasal 53."
        assert not _cek_penjelasan_dampak(_out(panjang=teks), _poin(kawasan_resapan=True))

    def test_tidak_dapat_memanfaatkan_pasal_53_ditolak(self):
        teks = "Lokasi tidak berada di kawasan resapan sehingga tidak dapat memanfaatkan ketentuan Pasal 53."
        m = _cek_penjelasan_dampak(_out(pendek=teks), _poin(kawasan_resapan=False))
        assert any("memanfaatkan" in x for x in m)


# ---------------------------------------------------------------- C. syarat kelompok kegiatan lain
class TestSyaratKelompokLain:
    def _itbx(self, ketentuan):
        return PoinKonteks(poin_id="itbx", kategori="ITBX", tipe_rekomendasi="kategorikal", status="B",
                           fakta={"lolos": True, "reason": "Lolos karena Bersyarat (B)",
                                  "keterangan_ketentuan": ketentuan})

    def test_syarat_asing_ditolak(self):
        out = _out("itbx", saran="Verifikasi jam operasional maksimal 22.00 dan siapkan dokumen lingkungan.")
        m = _cek_syarat_kelompok_lain(out, self._itbx(["luas akomodasi maksimal 20% dari luas blok"]))
        assert m and "jam operasional" in m[0] and "dokumen lingkungan" in m[0]

    def test_syarat_yang_ada_di_ketentuan_diizinkan(self):
        out = _out("itbx", saran="Patuhi jam operasional maksimal 22.00.")
        assert not _cek_syarat_kelompok_lain(out, self._itbx(["jam operasional dibatasi maksimal pukul 22.00"]))

    def test_poin_lain_tak_kena(self):
        assert not _cek_syarat_kelompok_lain(_out("dampak", saran="jam operasional 22.00"), _poin())


# ---------------------------------------------------------------- D. prosedur karangan
class TestProsedurKarangan:
    def test_prosedur_karangan_ditolak(self):
        out = _out("intensitas", saran="Jika tidak memungkinkan, ajukan permohonan penyesuaian intensitas ruang.")
        assert _cek_prosedur_karangan(out, _poin(poin_id="intensitas"), [])

    def test_ditolak_juga_izin_khusus(self):
        out = _out("itbx", saran="Minta izin khusus kepada dinas.")
        assert _cek_prosedur_karangan(out, _poin(poin_id="itbx"), [])

    def test_diizinkan_bila_tertulis_di_pasal(self):
        out = _out("intensitas", saran="Ajukan permohonan penyesuaian intensitas ruang.")
        chunk = _chunk("c", "9", "Penyesuaian dapat diberikan melalui permohonan penyesuaian oleh bupati.")
        assert not _cek_prosedur_karangan(out, _poin(poin_id="intensitas"), [chunk])

    def test_saran_biasa_lolos(self):
        assert not _cek_prosedur_karangan(_out("intensitas", saran="Kurangi luas lantai dasar hingga 718 m²."),
                                          _poin(poin_id="intensitas"), [])


# ---------------------------------------------------------------- E. sitasi dampak
class TestSitasiDampak:
    def test_pengisi_dikenali(self):
        assert chunk_pengisi_dampak(_chunk("p1-a66", "1", "66. Jaringan Drainase Lokal adalah ..."))
        assert chunk_pengisi_dampak(_chunk("rdtr-sleman-tengah-vi-r-3", "", "tabel", level="tabel"))
        assert chunk_pengisi_dampak(_chunk("x", "", "Lampiran VI — Ketentuan Intensitas ..."))
        assert not chunk_pengisi_dampak(_chunk("p53-a3", "53", "(3) Ketentuan khusus kawasan resapan air"))

    def test_ayat_pasal_53_sesuai_subzona(self):
        a2 = _chunk("p53-a2", "53", "(2) ... Sub-zona Taman Kecamatan dengan kode RTH-3, ... meliputi: a.")
        a3 = _chunk("p53-a3", "53", "(3) ... Sub-zona Perumahan Kepadatan Sedang dengan kode R-3, ... meliputi: a.")
        a1 = _chunk("p53-a1", "53", "(1) Ketentuan khusus kawasan resapan air terdapat di: a. SWP B Blok I.B.1")
        assert not _berlaku_utk_subzona(a2, "R-3")
        assert _berlaku_utk_subzona(a3, "R-3")
        assert _berlaku_utk_subzona(a1, "R-3")
        assert _berlaku_utk_subzona(a2, None)  # sub-zona tak diketahui -> tak menyaring

    def test_alur_penyaringan_resapan(self):
        class _R:
            def search(self, *a, **k):
                return []

        a2 = _chunk("p53-a2", "53", "(2) Ketentuan khusus kawasan resapan air ... dengan kode RTH-3, ... meliputi")
        a3 = _chunk("p53-a3", "53", "(3) Ketentuan khusus kawasan resapan air ... dengan kode R-3, ... meliputi")
        isi = _chunk("p1-a66", "1", "66. Jaringan Drainase Lokal adalah")
        poin = _poin(kawasan_resapan=True, zona="Zona Perumahan", zona_subzone="R-3")
        ids = [c.id for c in _sesuaikan_dgn_status_resapan([isi, a2, a3], poin, _R(), 3)]
        assert ids == ["p53-a3"]
        # Non-resapan: Pasal 53 dan pengisi hilang -> tidak ada sitasi pengisi.
        poin_non = _poin(kawasan_resapan=False, zona="Zona Perumahan", zona_subzone="R-3")
        assert _sesuaikan_dgn_status_resapan([isi, a2, a3], poin_non, _R(), 3) == []


# ---------------------------------------------------------------- F. label teknis, penomoran, catatan batas
class TestLabelDanPenomoran:
    def test_label_teknis_diganti(self):
        h = _normalisasi_karakter({"a": "Intensitas MELAMPAUI_BATAS dan MEMENUHI_SYARAT, TIDAK_DIEVALUASI."})
        assert h["a"] == "Intensitas melampaui batas dan memenuhi syarat, tidak dievaluasi."

    def test_ruang_terbuka_terbuka_diperbaiki(self):
        assert _normalisasi_karakter("tambah ruang terbuka terbuka (RTH)") == "tambah ruang terbuka hijau (RTH)"

    def test_nomor_di_awal_butir_kesimpulan_dibuang(self, monkeypatch):
        monkeypatch.setattr(llm_client_module, "generate", lambda *a, **k: {
            "langkah_berdampak": ["1. Pastikan luas akomodasi tidak melebihi 20%.", "2) Pertahankan RTH."],
            "catatan_lokasi": None})
        poin = [_out("itbx", saran="Pastikan luas akomodasi tidak melebihi 20%. Pertahankan RTH.")]
        poin[0] = poin[0].model_copy(update={"poin_id": "itbx"})
        hasil = _rakit_kesimpulan(poin, "Setuju Bersyarat")
        assert hasil.langkah_berdampak == ["Pastikan luas akomodasi tidak melebihi 20%.", "Pertahankan RTH."]


# ---------------------------------------------------------------- G. kutipan jangkar dari BE
def test_kutipan_jangkar_selalu_dari_be(monkeypatch):
    monkeypatch.setattr(llm_client_module, "generate", lambda *a, **k: {
        "reasoning_pendek": "p", "reasoning_panjang": "q", "saran": "s", "disclaimer": None,
        "sitasi": [{"citation_id": "anchor-0", "kutipan": "RDTR Sleman Ketentuan: KDB maks 99% (diubah LLM)"}]})
    poin = PoinKonteks(
        poin_id="intensitas", kategori="Intensitas", tipe_rekomendasi="numerik", status="MEMENUHI_SYARAT",
        fakta={"dinilai": True},
        dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="Ketentuan Intensitas Ruang (R-3)",
                                kutipan="KDB maks 60%, KLB maks 4.8, KDH min 25%")])
    hasil = generate_poin(poin, MockRetriever(), None)
    anchor = [s for s in hasil.sitasi if s.citation_id == "anchor-0"][0]
    assert anchor.kutipan == "KDB maks 60%, KLB maks 4.8, KDH min 25%"


# ---------------------------------------------------------------- H. format angka
def test_float_bulat_tanpa_nol_desimal():
    assert _angka_prompt(800.0) == "800"
    assert _angka_prompt(1077.0) == "1077"
    assert _angka_prompt(1.8) == "1.8"


# ---------------------------------------------------------------- I. koma desimal
def test_koma_desimal_dianggap_format_bukan_karangan():
    poin = PoinKonteks(
        poin_id="intensitas", kategori="Intensitas", tipe_rekomendasi="numerik", status="MELAMPAUI_BATAS",
        fakta={"parameter": {"klb": {"usulan": 2.51, "ambang_maks": 4.8, "memenuhi": True, "satuan": "rasio"},
                             "kdb": {"usulan": 20.89, "ambang_maks": 60.0, "memenuhi": True, "satuan": "persen"}}})
    assert _angka_terlacak_ke_sumber("4,8", poin)
    assert _angka_terlacak_ke_sumber("20,89", poin)
    assert not _angka_terlacak_ke_sumber("4,9", poin)
