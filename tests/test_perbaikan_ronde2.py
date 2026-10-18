"""Perbaikan ronde 2 hasil evaluasi APP-2026-UJI-1..4 (2026-10-16): ITBX tak boleh membahas intensitas,
tak boleh menyatakan syarat sudah terpenuhi, tak boleh menambah kewajiban administratif karangan."""

import pytest

from app.reasoning.guardrail import (
    _cek_itbx_bahas_intensitas,
    _cek_itbx_klaim_terpenuhi,
    _cek_kewajiban_administratif_karangan,
)
from app.reasoning.prompts import SYSTEM_PROMPT
from app.retrieval.base import Chunk
from app.schemas import PoinKonteks, PoinOutput, RekomendasiOutput


def _itbx(status="B", ketentuan=None, reason="Lolos Bersyarat (B)"):
    return PoinKonteks(
        poin_id="itbx", kategori="ITBX", tipe_rekomendasi="kategorikal", status=status,
        fakta={"lolos": True, "reason": reason,
               "keterangan_ketentuan": ketentuan or ["luas keseluruhan kegiatan akomodasi makan minum maksimal 20% dari luas blok"]})


def _out(pendek="p", panjang="q", saran="s") -> PoinOutput:
    return PoinOutput(poin_id="itbx", kategori="K", status="B", reasoning_pendek=pendek, reasoning_panjang=panjang,
                      sitasi=[], rekomendasi=RekomendasiOutput(tipe="kategorikal", saran=saran), low_confidence=False)


class TestItbxTidakMembahasIntensitas:
    def test_kdb_di_itbx_ditolak(self):
        m = _cek_itbx_bahas_intensitas(_out(saran="Verifikasi perhitungan KDB tidak melebihi 60% sesuai RDTR."), _itbx())
        assert m and "intensitas" in m[0]

    def test_ambang_intensitas_ditolak(self):
        assert _cek_itbx_bahas_intensitas(_out(panjang="Perlu memperhatikan ambang intensitas ruang."), _itbx())

    def test_boleh_bila_ada_di_ketentuan_be(self):
        poin = _itbx(ketentuan=["KDB maksimal 40% untuk kegiatan ini"])
        assert not _cek_itbx_bahas_intensitas(_out(saran="Pastikan KDB tidak melebihi 40%."), poin)

    def test_poin_lain_tak_kena(self):
        poin = PoinKonteks(poin_id="intensitas", kategori="I", tipe_rekomendasi="numerik", status="MELAMPAUI_BATAS", fakta={})
        assert not _cek_itbx_bahas_intensitas(_out(saran="KDB 83% melebihi 60%"), poin)


class TestKlaimTerpenuhi:
    def test_memenuhi_syarat_luas_ditolak(self):
        m = _cek_itbx_klaim_terpenuhi(_out(pendek="Kegiatan ini memenuhi syarat luas maksimal 20% blok."), _itbx())
        assert m and "HARUS dipenuhi" in m[0]

    def test_telah_memenuhi_ditolak(self):
        m = _cek_itbx_klaim_terpenuhi(
            _out(panjang="**Memenuhi ketentuan luas 20%**: pemohon telah memenuhi batas tersebut."), _itbx())
        assert m

    def test_otomatis_ditolak_apa_pun_statusnya(self):
        for status in ("B", "I"):
            m = _cek_itbx_klaim_terpenuhi(
                _out(panjang="Kegiatan dapat diloloskan secara otomatis oleh sistem."), _itbx(status=status))
            assert any("otomatis" in x for x in m), status

    def test_kalimat_bersyarat_diizinkan(self):
        teks = "Pemohon harus memenuhi ketentuan luas maksimal 20% dari luas blok agar kegiatan dapat diloloskan."
        assert not _cek_itbx_klaim_terpenuhi(_out(panjang=teks), _itbx())

    def test_status_diizinkan_boleh_menyebut_memenuhi(self):
        assert not _cek_itbx_klaim_terpenuhi(_out(panjang="Kegiatan memenuhi syarat luas yang berlaku."), _itbx(status="I"))


class TestKewajibanAdministratifKarangan:
    @pytest.mark.parametrize("teks", [
        "Pemohon harus menyertakan dokumen perhitungan luas blok dalam berkas permohonan.",
        "Jika melebihi batas, pemohon harus mengurangi luas atau menambah blok.",
        "Siapkan dokumen pendukung (rencana tata ruang).",
        "Sertakan dokumen perhitungan KDB.",
    ])
    def test_ditolak(self, teks):
        assert _cek_kewajiban_administratif_karangan(_out(saran=teks), _itbx(), [])

    def test_diizinkan_bila_tertulis_di_pasal(self):
        chunk = Chunk(id="c", level="ayat", dokumen="Perbup", pasal="9",
                      teks="Pemohon wajib menyertakan dokumen perhitungan luas blok.")
        assert not _cek_kewajiban_administratif_karangan(
            _out(saran="Sertakan dokumen perhitungan luas blok."), _itbx(), [chunk])

    def test_saran_biasa_lolos(self):
        assert not _cek_kewajiban_administratif_karangan(
            _out(saran="Pastikan luas kegiatan akomodasi tidak melebihi 20% dari luas blok."), _itbx(), [])


class TestPromptDanTeksPasal53:
    def test_aturan_prompt_baru_ada(self):
        assert "ITBX HANYA SOAL KLASIFIKASI KEGIATAN" in SYSTEM_PROMPT
        assert "BELUM TERBUKTI TERPENUHI" in SYSTEM_PROMPT
        assert "JANGAN MENAMBAH KEWAJIBAN ADMINISTRATIF" in SYSTEM_PROMPT

    def test_sumur_resapan_diwajibkan_di_lahan_terbangun_eksisting(self):
        from app.reasoning.prompts import _bangun_fakta_dampak
        p = PoinKonteks(poin_id="dampak", kategori="D", tipe_rekomendasi="numerik-mitigasi", status="Tinggi",
                        fakta={"dinilai": True}, kawasan_resapan=True)
        teks = "\n".join(_bangun_fakta_dampak(p))
        assert "LAHAN TERBANGUN YANG SUDAH ADA" in teks and "JANGAN menyebut huruf d sebagai dasar kewajiban" in teks

    def test_langkah_pemeliharaan_resapan_menyebut_zero_delta_q_dan_lahan_eksisting(self):
        from app.reasoning.calculator import bangun_langkah_dampak_poin
        fakta = {"mitigasi": {"perlu_mitigasi": False, "arah": ["x"], "pemeliharaan": {}}, "kawasan_resapan": True}
        deskripsi = bangun_langkah_dampak_poin(fakta)[-1]["deskripsi"]
        assert "zero delta Q" in deskripsi and "lahan terbangun yang sudah ada" in deskripsi
