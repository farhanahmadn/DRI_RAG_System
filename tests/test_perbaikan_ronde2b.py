"""Ronde 2 bagian kedua: kalimat intensitas keliru (#3), istilah/format (#4), sitasi pengisi (#6),
atribusi Lampiran V.B (#7), dan pencatatan penyebab sintesis fallback (#8)."""

import json
from pathlib import Path

from app.logging_util import log_precheck
from app.reasoning import assemble
from app.reasoning import llm_client
from app.reasoning.generator import _saring_pengisi_poin, chunk_pengisi_dampak
from app.reasoning.guardrail import (
    _cek_atribusi_lampiran_vb,
    _cek_intensitas_dan_istilah,
    _perbaiki_bold,
)
from app.reasoning.prompts import _bangun_fakta_intensitas
from app.retrieval.base import Chunk
from app.schemas import L2Assessment, PoinKonteks, PoinOutput, RekomendasiOutput

VB_NYATA = (
    "Lampiran V.B — Penjelasan Ketentuan Kegiatan (T/B) Sub-zona Perumahan Kepadatan Sedang (R-3):\n"
    "2. jam operasional (jam buka) dibatasi sampai dengan maksimal pukul 22.00 kecuali dengan izin instansi;\n"
    "4. khusus peternakan domba dan kambing jumlah ternak maksimal 100 ekor;\n"
    "5. khusus peternakan unggas jumlah ternak maksimal 20.000 ekor dan/atau telah terbangun")


def _chunk(cid, teks, pasal=""):
    return Chunk(id=cid, level="ayat", dokumen="Perbup", pasal=pasal, teks=teks)


def _out(poin_id="intensitas", pendek="p", panjang="q", saran="s") -> PoinOutput:
    return PoinOutput(poin_id=poin_id, kategori="K", status="MEMENUHI_SYARAT", reasoning_pendek=pendek,
                      reasoning_panjang=panjang, sitasi=[],
                      rekomendasi=RekomendasiOutput(tipe="numerik", saran=saran), low_confidence=False)


def _poin(poin_id="intensitas", **kw):
    d = dict(poin_id=poin_id, kategori="K", tipe_rekomendasi="numerik", status="MEMENUHI_SYARAT", fakta={})
    d.update(kw)
    return PoinKonteks(**d)


class TestIstilahDanSetara:
    def test_bersyarat_di_intensitas_ditolak(self):
        m = _cek_intensitas_dan_istilah(_out(panjang="pemohon diklasifikasikan **Bersyarat** untuk intensitas."), _poin())
        assert m and "istilah ITBX" in m[0]

    def test_bersyarat_di_dampak_tidak_dipersoalkan(self):
        assert not _cek_intensitas_dan_istilah(_out("dampak", panjang="status ITBX Bersyarat"), _poin("dampak"))

    def test_setara_dengan_kebutuhan_ditolak(self):
        m = _cek_intensitas_dan_istilah(
            _out(panjang="KDB 83.57% setara dengan kebutuhan lantai dasar maksimal 861.6 m²."), _poin())
        assert any("setara" in x for x in m)

    def test_kalimat_benar_lolos(self):
        teks = "Luas lantai dasar usulan 1200 m² (KDB 83.57%), batas maksimal 861.6 m² (KDB 60%)."
        assert not _cek_intensitas_dan_istilah(_out(panjang=teks), _poin())

    def test_prompt_memisahkan_usulan_dan_batas(self):
        p = _poin(status="MELAMPAUI_BATAS", fakta={
            "parameter": {"kdb": {"usulan": 83.57, "ambang_maks": 60.0, "ambang_min": None, "memenuhi": False, "satuan": "persen"},
                          "kdh": {"usulan": 13.93, "ambang_maks": None, "ambang_min": 25.0, "memenuhi": False, "satuan": "persen"}},
            "luas_tapak_m2": 1200.0, "luas_rth_usulan_m2": 200.0,
            "target": {"kdb": {"target_kdb": 60.0, "selisih": 23.57, "footprint_maks_m2": 861.6},
                       "kdh": {"target_kdh": 25.0, "selisih": 11.07, "rth_dibutuhkan_m2": 359.0}}})
        teks = "\n".join(_bangun_fakta_intensitas(p))
        assert "Luas lantai dasar USULAN pemohon: 1200 m2 (KDB 83.57%)" in teks
        assert "MAKSIMAL yang diperbolehkan: 861.6 m2 (KDB 60%)" in teks
        assert "RTH USULAN pemohon: 200 m2 (KDH 13.93%)" in teks and "RTH MINIMAL yang dibutuhkan: 359 m2" in teks
        assert "JANGAN menulis persentase usulan 'setara'" in teks


class TestBoldTakBerpasangan:
    def test_ganjil_dibuang_yang_terakhir(self):
        assert _perbaiki_bold("**Memenuhi** dan **lanjut.") == "**Memenuhi** dan lanjut."
        assert _perbaiki_bold("**Memenuhi** dan lanjut.**") == "**Memenuhi** dan lanjut."

    def test_genap_tak_berubah(self):
        assert _perbaiki_bold("**a** b **c**") == "**a** b **c**"
        assert _perbaiki_bold("tanpa bold") == "tanpa bold"


class TestAtribusiLampiranVB:
    def test_angka_20_persen_diatribusikan_ke_vb_ditolak(self):
        """Teks V.B memuat "20.000 ekor" tapi BUKAN "20 %" luas blok."""
        out = _out("itbx", panjang="Menurut Lampiran V.B, luas akomodasi makan minum maksimal 20 % dari luas blok.")
        m = _cek_atribusi_lampiran_vb(out, [_chunk("rdtr-sleman-tengah-vb-r-3", VB_NYATA)])
        assert m and "'20'" in m[0]

    def test_angka_yang_memang_ada_di_vb_lolos(self):
        out = _out("itbx", panjang="Menurut Lampiran V.B, jam operasional dibatasi maksimal pukul 22.00.")
        assert not _cek_atribusi_lampiran_vb(out, [_chunk("rdtr-sleman-tengah-vb-r-3", VB_NYATA)])

    def test_tanpa_rujukan_vb_tak_dicek(self):
        out = _out("itbx", panjang="Luas akomodasi makan minum maksimal 20 % dari luas blok.")
        assert not _cek_atribusi_lampiran_vb(out, [_chunk("rdtr-sleman-tengah-vb-r-3", VB_NYATA)])

    def test_tanpa_chunk_vb_tak_dicek(self):
        out = _out("itbx", panjang="Menurut Lampiran V.B, maksimal 20 %.")
        assert not _cek_atribusi_lampiran_vb(out, [])


class TestSitasiPengisi:
    def test_itbx_tanpa_tabel_intensitas(self):
        vi = _chunk("rdtr-sleman-tengah-vi-r-3", "Lampiran VI — Ketentuan Intensitas ... KDB maksimum 50-60%")
        vb = _chunk("rdtr-sleman-tengah-vb-r-3", VB_NYATA)
        sisa = _saring_pengisi_poin([vi, vb], _poin("itbx", status="B"))
        assert [c.id for c in sisa] == ["rdtr-sleman-tengah-vb-r-3"]

    def test_intensitas_tanpa_syarat_kegiatan(self):
        vi = _chunk("rdtr-sleman-tengah-vi-r-3", "Lampiran VI — Ketentuan Intensitas ...")
        vb = _chunk("rdtr-sleman-tengah-vb-r-3", VB_NYATA)
        assert [c.id for c in _saring_pengisi_poin([vb, vi], _poin("intensitas"))] == ["rdtr-sleman-tengah-vi-r-3"]

    def test_penyaringan_tak_pernah_mengosongkan(self):
        vi = _chunk("rdtr-sleman-tengah-vi-r-3", "Lampiran VI — tabel")
        assert _saring_pengisi_poin([vi], _poin("itbx", status="B")) == [vi]

    def test_pasal_44_pengisi_di_dampak(self):
        assert chunk_pengisi_dampak(_chunk("p44-a2", "(2) Luas minimal bidang tanah ... di luar resapan air", pasal="44"))
        assert not chunk_pengisi_dampak(_chunk("p53-a3", "(3) Ketentuan khusus kawasan resapan air", pasal="53"))


class TestCatatanSintesis:
    def test_narasi_llm_berhasil_dicatat(self, monkeypatch):
        monkeypatch.setattr(llm_client, "generate", lambda *a, **k: {
            "paragraf_gate_intensitas": "Gate baik.", "paragraf_dampak": "Dampak rendah."})
        catatan: dict = {}
        assemble._rakit_narasi_rekomendasi([], "Setuju", catatan)
        assert catatan == {"narasi": "llm"}

    def test_alasan_fallback_narasi_dicatat(self, monkeypatch):
        monkeypatch.setattr(llm_client, "generate", lambda *a, **k: {
            "paragraf_gate_intensitas": "KDB harus 777.", "paragraf_dampak": "Dampak."})
        catatan: dict = {}
        assemble._rakit_narasi_rekomendasi([], "Setuju", catatan)
        assert catatan["narasi"].startswith("fallback: ValueError") and "777" in catatan["narasi"]

    def test_alasan_fallback_kesimpulan_dicatat(self, monkeypatch):
        def _gagal(*a, **k):
            raise RuntimeError("Groq mati")
        monkeypatch.setattr(llm_client, "generate", _gagal)
        catatan: dict = {}
        assemble._rakit_kesimpulan([], "Setuju", catatan)
        assert "fallback: RuntimeError: Groq mati" == catatan["kesimpulan"]

    def test_log_menulis_kunci_sintesis(self, tmp_path):
        from app.schemas import (KesimpulanOutput, NarasiRekomendasiOutput, OutputL3,
                                 RingkasanDampakOutput, RingkasanGateOutput)
        payload = json.loads((Path(__file__).parent / "fixtures" / "l2_sample_lolos.json").read_text(encoding="utf-8"))
        a = L2Assessment.model_validate(payload["data"])
        resp = OutputL3(
            ringkasan_gate=RingkasanGateOutput(final_gate_status="Lolos", decisive_stage=None, kalimat="k"),
            ringkasan_dampak=RingkasanDampakOutput(impact_category="Rendah", kalimat="k"),
            poin=[], rekomendasi_sistem="Setuju",
            narasi_rekomendasi=NarasiRekomendasiOutput(paragraf_gate_intensitas="x", paragraf_dampak="x"),
            kesimpulan=KesimpulanOutput(langkah_berdampak=[], catatan_lokasi=None),
            catatan_global=[], low_confidence_keseluruhan=False)
        path = tmp_path / "log.jsonl"
        log_precheck(a, resp, sintesis={"narasi": "fallback: ValueError: x"}, log_path=path)
        log_precheck(a, resp, log_path=path)
        baris = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        assert baris[0]["sintesis"] == {"narasi": "fallback: ValueError: x"}
        assert baris[1]["sintesis"] == {}


class TestAngkaSintesisFormatDesimal:
    """Hasil uji nyata 2026-10-16: sintesis narasi ditolak karena "861,6" vs "861.6"."""

    def _poin(self, saran):
        return [_out("intensitas", saran=saran)]

    def test_koma_dan_titik_dianggap_sama(self):
        poin = self._poin("Kurangi luas lantai dasar hingga 861.6 m² (KDB 60%).")
        assert assemble._angka_tak_terlacak_ke_saran("maksimal 861,6 m²", poin) is None
        poin = self._poin("Luas lantai dasar maksimal 861,6 m².")
        assert assemble._angka_tak_terlacak_ke_saran("maksimal 861.6 m²", poin) is None

    def test_angka_karangan_tetap_ditolak(self):
        poin = self._poin("Kurangi luas lantai dasar hingga 861.6 m².")
        assert assemble._angka_tak_terlacak_ke_saran("maksimal 862,6 m²", poin) is not None
