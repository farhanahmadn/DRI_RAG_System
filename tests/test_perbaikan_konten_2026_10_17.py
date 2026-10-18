"""Perbaikan konten hasil uji nyata 2026-10-16: klaim dampak tanpa dasar, fallback sintesis yang rapi,
prompt sistem yang lebih ringkas."""

from app.reasoning import prompts
from app.reasoning.assemble import _rakit_kesimpulan_fallback, _rakit_narasi_rekomendasi_fallback
from app.reasoning.calculator import sarankan_arah_mitigasi_dampak  # noqa: F401  (pastikan modul termuat)
from app.reasoning.guardrail import _cek_klaim_dampak_tanpa_dasar
from app.schemas import PoinKonteks, PoinOutput, RekomendasiOutput


def _out(poin_id, saran, panjang="q", status="Tinggi") -> PoinOutput:
    return PoinOutput(poin_id=poin_id, kategori="K", status=status, reasoning_pendek="p",
                      reasoning_panjang=panjang, sitasi=[],
                      rekomendasi=RekomendasiOutput(tipe="numerik", saran=saran), low_confidence=False)


def _dampak() -> PoinKonteks:
    return PoinKonteks(poin_id="dampak", kategori="K", tipe_rekomendasi="numerik", status="Tinggi", fakta={})


class TestKlaimDampak:
    def test_koefisien_beda_ditolak(self):
        o = _out("dampak", "s", "Koefisien limpasan atap berbeda antara bangunan industri dan pergudangan.")
        assert _cek_klaim_dampak_tanpa_dasar(o, _dampak())

    def test_huruf_d_untuk_bangunan_baru_ditolak(self):
        o = _out("dampak", "s", "Sediakan sumur sesuai Pasal 53 ayat 3 huruf d.")
        assert _cek_klaim_dampak_tanpa_dasar(o, _dampak())

    def test_huruf_d_lahan_terbangun_lolos(self):
        o = _out("dampak", "s", "Pasal 53 ayat 3 huruf d mengatur sumur pada lahan terbangun yang sudah ada.")
        assert not _cek_klaim_dampak_tanpa_dasar(o, _dampak())

    def test_klaim_memenuhi_ditolak(self):
        o = _out("dampak", "s", "Pemohon telah memenuhi syarat tersebut.")
        assert _cek_klaim_dampak_tanpa_dasar(o, _dampak())

    def test_poin_lain_diabaikan(self):
        o = _out("itbx", "s", "Koefisien limpasan atap berbeda.")
        p = PoinKonteks(poin_id="itbx", kategori="K", tipe_rekomendasi="kategorikal", status="B", fakta={})
        assert not _cek_klaim_dampak_tanpa_dasar(o, p)


class TestFallbackSintesis:
    def test_butir_bernomor_dipecah_dan_baku_dibuang(self):
        poin = [
            _out("itbx", "1. Pastikan jam buka.\n2. Sediakan bukti."),
            _out("intensitas", "Tidak diperlukan tindakan khusus. Poin ini telah memenuhi ketentuan."),
            _out("dampak", "1. Pertahankan RTH 260 m².\n2. Jaga bangunan 140 m²."),
        ]
        k = _rakit_kesimpulan_fallback(poin)
        assert k.langkah_berdampak == [
            "Pastikan jam buka.", "Sediakan bukti.", "Pertahankan RTH 260 m².", "Jaga bangunan 140 m²."]
        n = _rakit_narasi_rekomendasi_fallback(poin)
        assert "\n" not in n.paragraf_gate_intensitas and "1." not in n.paragraf_gate_intensitas
        assert "Tidak diperlukan" not in n.paragraf_gate_intensitas
        assert n.paragraf_dampak == "Pertahankan RTH 260 m². Jaga bangunan 140 m²."

    def test_semua_baku_tetap_dipertahankan_satu(self):
        poin = [_out("itbx", "Tidak diperlukan tindakan khusus."), _out("intensitas", "Tidak diperlukan tindakan khusus.")]
        assert len(_rakit_kesimpulan_fallback(poin).langkah_berdampak) == 1


def test_system_prompt_lebih_ringkas():
    assert len(prompts.SYSTEM_PROMPT) < 7000


def test_panggilan_llm_poin_berurutan(monkeypatch):
    import threading
    import time

    from app.reasoning import llm_client

    aktif, puncak, kunci = 0, 0, threading.Lock()

    def palsu(prompt, json_schema, **kw):
        nonlocal aktif, puncak
        with kunci:
            aktif += 1
            puncak = max(puncak, aktif)
        time.sleep(0.05)
        with kunci:
            aktif -= 1
        return {}

    monkeypatch.setattr(llm_client, "_generate_tanpa_antre", palsu)
    monkeypatch.setattr(llm_client, "_antrean_llm", threading.BoundedSemaphore(1))
    ts = [threading.Thread(target=llm_client.generate, args=("p", {}), kwargs={"schema_name": "poin"})
          for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert puncak == 1


def test_sintesis_tidak_diantre(monkeypatch):
    from app.reasoning import llm_client

    monkeypatch.setattr(llm_client, "_generate_tanpa_antre", lambda p, s, **kw: {"ok": 1})
    with llm_client._antrean_llm:  # antrean poin terkunci, sintesis tetap jalan
        assert llm_client.generate("p", {}, schema_name="kesimpulan") == {"ok": 1}


class TestPinKunciDanSintesisAuto:
    def _rot(self, n=3):
        from app.reasoning.llm_client import _RotasiKunciGroq
        return _RotasiKunciGroq([f"k{i}" for i in range(n)], None)

    def test_pin_memetakan_poin_ke_kunci_tetap(self):
        r = self._rot()
        assert [r.kunci_awal(i) for i in (0, 1, 2, 0)] == [0, 1, 2, 0]

    def test_pin_maju_bila_kunci_karantina(self):
        r = self._rot()
        r.karantina(1, 60, "uji")
        assert r.kunci_awal(1) == 2

    def test_tanpa_pin_tetap_round_robin(self):
        r = self._rot()
        assert [r.kunci_awal() for _ in range(4)] == [0, 1, 2, 0]

    def test_konteks_poin_membawa_pin(self):
        from app.reasoning import assemble, llm_client
        assert assemble._konteks_dgn_pin(2).run(llm_client._pin_kunci.get) == 2
        assert llm_client._pin_kunci.get() is None

    def test_sintesis_off_dilewati(self, monkeypatch):
        from app.reasoning import assemble
        monkeypatch.setenv("SINTESIS_LLM", "off")
        assert assemble._alasan_lewati_sintesis() is not None

    def test_sintesis_auto_lewati_saat_waktu_sempit_atau_429(self, monkeypatch):
        from app import batas_waktu
        from app.reasoning import assemble, llm_client
        monkeypatch.setenv("SINTESIS_LLM", "auto")
        assert assemble._alasan_lewati_sintesis() is None  # tanpa anggaran, tanpa 429
        with batas_waktu.anggaran(10.0):
            assert "sisa waktu" in assemble._alasan_lewati_sintesis()
        llm_client._catat_rate_limit()
        assert "429" in assemble._alasan_lewati_sintesis()
        monkeypatch.setattr(llm_client, "_waktu_rate_limit_terakhir", 0.0)

    def test_sintesis_on_selalu_mencoba(self, monkeypatch):
        from app.reasoning import assemble, llm_client
        monkeypatch.setenv("SINTESIS_LLM", "on")
        llm_client._catat_rate_limit()
        assert assemble._alasan_lewati_sintesis() is None
        monkeypatch.setattr(llm_client, "_waktu_rate_limit_terakhir", 0.0)


class TestAturanPM:
    def test_sumur_di_dampak_tinggi_non_resapan_ditolak(self):
        from app.reasoning.guardrail import _cek_sumur_dampak_tinggi_non_resapan
        poin = PoinKonteks(poin_id="dampak", kategori="K", tipe_rekomendasi="numerik-mitigasi",
                           status="Tinggi", fakta={}, kawasan_resapan=False)
        assert _cek_sumur_dampak_tinggi_non_resapan(_out("dampak", "s", "Sediakan sumur resapan."), poin)
        assert not _cek_sumur_dampak_tinggi_non_resapan(_out("dampak", "s", "Turunkan KDB."), poin)
        resapan = poin.model_copy(update={"kawasan_resapan": True})
        assert not _cek_sumur_dampak_tinggi_non_resapan(_out("dampak", "s", "Sediakan sumur resapan."), resapan)
        rendah = poin.model_copy(update={"status": "Rendah"})
        assert not _cek_sumur_dampak_tinggi_non_resapan(_out("dampak", "s", "Sediakan sumur resapan."), rendah)

    def test_arah_mitigasi_tanpa_sumur_di_non_resapan(self):
        from app.schemas import ImpactAssessment
        i = ImpactAssessment(dinilai=True, impact_category="TINGGI", runoff_change_index=2.7)
        assert "sumur resapan" not in sarankan_arah_mitigasi_dampak(i, False)["arah"]
        assert "sumur resapan" in sarankan_arah_mitigasi_dampak(i, True)["arah"]

    def test_sitasi_dampak_dilengkapi_deterministik(self):
        from app.reasoning.generator import _lengkapi_sitasi_dampak
        from app.retrieval.base import Chunk
        c53 = Chunk(id="rdtr-sleman-tengah-p53-a3", level="ayat", dokumen="Perbup", pasal="53", teks="zero delta Q")
        cvi = Chunk(id="rdtr-sleman-tengah-vi-r-3", level="tabel", dokumen="Perbup", pasal="", teks="KDB")
        peta = {c53.id: c53, cvi.id: cvi}
        resapan = PoinKonteks(poin_id="dampak", kategori="K", tipe_rekomendasi="numerik-mitigasi",
                              status="Tinggi", fakta={}, kawasan_resapan=True)
        assert [x.citation_id for x in _lengkapi_sitasi_dampak([], resapan, peta)] == [c53.id]
        bukan = resapan.model_copy(update={"kawasan_resapan": False, "fakta": {"target_mitigasi": {"a": 1}}})
        assert [x.citation_id for x in _lengkapi_sitasi_dampak([], bukan, peta)] == [cvi.id]
        rendah_bukan = bukan.model_copy(update={"fakta": {}})
        assert _lengkapi_sitasi_dampak([], rendah_bukan, peta) == []


class TestFallbackDeterministik:
    def _dampak(self, resapan):
        fakta = {"dinilai": True, "target_mitigasi": {
            "runoff_change_index_maks": 2.5, "kategori_target": "Sedang",
            "penyesuaian_lahan": {"luas_bangunan_maks_m2": 1077.0, "luas_rth_min_m2": 359.0,
                                  "kdb_maks_persen": 75, "kdh_min_persen": 25},
            "dimensi_minimum_resapan": {"nilai": 0.62, "satuan": "m³"}, "kawasan_resapan": resapan},
            "kawasan_resapan": resapan}
        return PoinKonteks(poin_id="dampak", kategori="K", tipe_rekomendasi="numerik-mitigasi",
                           status="Tinggi", fakta=fakta, kawasan_resapan=resapan)

    def test_dampak_tinggi_fallback_memakai_saran_pm(self):
        from app.reasoning.templates import template_low_confidence
        r = template_low_confidence(self._dampak(True))
        assert r.low_confidence and r.rekomendasi.saran.startswith("1. Wajib menurunkan")
        assert "2. Wajib memastikan air hujan tidak melimpas" in r.rekomendasi.saran
        b = template_low_confidence(self._dampak(False))
        assert "sumur" not in b.rekomendasi.saran and b.low_confidence
        assert "Perlu peninjauan manual" not in b.rekomendasi.saran

    def test_itbx_tanpa_langkah_tetap_peninjauan_manual(self):
        from app.reasoning.templates import template_low_confidence
        p = PoinKonteks(poin_id="itbx", kategori="K", tipe_rekomendasi="kategorikal", status="B", fakta={})
        assert "peninjauan manual" in template_low_confidence(p).rekomendasi.saran
