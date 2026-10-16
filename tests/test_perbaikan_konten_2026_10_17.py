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
