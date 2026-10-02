import json
import time
from pathlib import Path

from app.adapter import adaptasi
from app.reasoning import assemble as assemble_module
from app.reasoning.guardrail import DiagnosaPoin
from app.reasoning.prompts import CAVEAT_SUBZONA_TAK_TERKONFIRMASI
from app.reasoning.assemble import (
    CAVEAT_DI_LUAR_CAKUPAN,
    _rakit_catatan_global,
    _rakit_kalimat_gate,
    _rakit_kesimpulan,
    _rakit_kesimpulan_fallback,
    _rakit_ringkasan_dampak,
    jalankan_precheck,
)
from app.retrieval.mock import MockRetriever
from app.schemas import L2Assessment, MetaL2, PoinOutput, RekomendasiOutput

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _muat_assessment(nama_file: str) -> L2Assessment:
    payload = json.loads((FIXTURES_DIR / nama_file).read_text(encoding="utf-8"))
    return L2Assessment.model_validate(payload["data"])


def _poin_output(**overrides) -> PoinOutput:
    defaults = dict(
        poin_id="itbx",
        kategori="Klasifikasi Kegiatan (ITBX)",
        status="I",
        reasoning_pendek="x",
        reasoning_panjang="x",
        sitasi=[],
        rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Tidak diperlukan tindakan khusus."),
        low_confidence=False,
    )
    defaults.update(overrides)
    return PoinOutput(**defaults)


def _diagnosa_stub(poin_id: str, berhasil: bool = True) -> DiagnosaPoin:
    """Diagnosa minimal utk stub — assemble sekarang menerima (PoinOutput, DiagnosaPoin)."""
    return DiagnosaPoin(poin_id=poin_id, berhasil=berhasil, percobaan=1, jumlah_chunk=1)


def _patch_guardrail(monkeypatch, poin_by_id: dict[str, PoinOutput]):
    dipanggil = []

    def _stub(poin, retriever, assessment, *, max_retry=2):
        dipanggil.append(poin.poin_id)
        return poin_by_id[poin.poin_id], _diagnosa_stub(poin.poin_id)

    monkeypatch.setattr(assemble_module, "generate_poin_terdiagnosis", _stub)
    return dipanggil


def _patch_log(monkeypatch):
    panggilan = []

    # **kwargs: assemble sekarang meneruskan `diagnostik=` (lihat guardrail.DiagnosaPoin).
    def _stub(request, response, **kwargs):
        panggilan.append((request, response, kwargs))

    monkeypatch.setattr(assemble_module, "log_precheck", _stub)
    return panggilan


def _patch_kesimpulan_llm(monkeypatch, hasil_dict=None, raise_exc=None):
    def _stub(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        if raise_exc:
            raise raise_exc
        return hasil_dict

    monkeypatch.setattr(assemble_module.llm_client, "generate", _stub)


# --- _rakit_kalimat_gate: fungsi murni atas primitif — testable tanpa fabrikasi L2Assessment penuh.


class TestRakitKalimatGate:
    def test_lolos(self):
        kalimat = _rakit_kalimat_gate("Lolos", None, itbx_fallback=False)
        assert "lolos pemeriksaan" in kalimat.lower()
        assert "bersyarat" not in kalimat.lower()

    def test_lolos_bersyarat_tanpa_decisive_stage(self):
        kalimat = _rakit_kalimat_gate("Lolos Bersyarat", None, itbx_fallback=False)
        assert "lolos bersyarat" in kalimat.lower()

    def test_lolos_bersyarat_dengan_decisive_stage(self):
        kalimat = _rakit_kalimat_gate("Lolos Bersyarat", "intensitas", itbx_fallback=False)
        assert "intensitas bangunan" in kalimat.lower()

    def test_tidak_lolos_tanpa_decisive_stage(self):
        kalimat = _rakit_kalimat_gate("Tidak Lolos", None, itbx_fallback=False)
        assert "tidak lolos" in kalimat.lower()

    def test_tidak_lolos_dengan_decisive_stage_itbx(self):
        kalimat = _rakit_kalimat_gate("Tidak Lolos", "itbx", itbx_fallback=False)
        assert "tidak lolos" in kalimat.lower()
        assert "klasifikasi kegiatan" in kalimat.lower()

    def test_decisive_stage_tak_dikenal_ditampilkan_apa_adanya(self):
        kalimat = _rakit_kalimat_gate("Lolos Bersyarat", "tahap-baru-belum-dikenal", itbx_fallback=False)
        assert "tahap-baru-belum-dikenal" in kalimat


class TestRakitKalimatGateFallbackItbx:
    """Blueprint §5.2: ITBX yang lolos via fallback data-kosong BUKAN kepatuhan terverifikasi —
    narasi TIDAK BOLEH overclaim ke arah mana pun, final_gate_status sendiri TETAP apa adanya."""

    def test_lolos_dengan_fallback_tidak_overclaim_memenuhi_ketentuan(self):
        kalimat = _rakit_kalimat_gate("Lolos", None, itbx_fallback=True)
        assert "memenuhi seluruh ketentuan" not in kalimat.lower()
        assert "belum terverifikasi" in kalimat.lower()

    def test_lolos_bersyarat_dengan_fallback_tetap_sebut_caveat(self):
        kalimat = _rakit_kalimat_gate("Lolos Bersyarat", "intensitas", itbx_fallback=True)
        assert "lolos bersyarat" in kalimat.lower()  # kalimat dasar tetap ada
        assert "belum terverifikasi" in kalimat.lower()  # caveat ditambahkan

    def test_tidak_lolos_dengan_fallback_tidak_overclaim_pelanggaran_mutlak(self):
        # Bukti dari fixture nyata l2_sample_tidak_lolos.json: status X + reason ambigu ("dilarang
        # atau tidak ditemukan") — mengklaim "pelanggaran mutlak" tanpa catatan juga overclaim.
        kalimat = _rakit_kalimat_gate("Tidak Lolos", "itbx", itbx_fallback=True)
        assert "tidak lolos" in kalimat.lower()
        assert "bukan kepastian pelanggaran" in kalimat.lower()

    def test_tidak_lolos_dengan_fallback_tidak_sebut_mutlak(self):
        # APP-2026-3335: framing "mutlak" (kalimat dasar) dan "bukan kepastian" (caveat) TIDAK
        # BOLEH muncul bersamaan — kontradiktif. Kalau fallback berlaku, "mutlak" dihapus.
        kalimat = _rakit_kalimat_gate("Tidak Lolos", "itbx", itbx_fallback=True)
        assert "mutlak" not in kalimat.lower()

    def test_tidak_lolos_tanpa_fallback_tetap_sebut_mutlak(self):
        # Regresi: tanpa fallback, framing "mutlak" tetap dipakai (verdict genuinely tegas).
        kalimat = _rakit_kalimat_gate("Tidak Lolos", "itbx", itbx_fallback=False)
        assert "mutlak" in kalimat.lower()

    def test_tanpa_fallback_kalimat_tak_berubah(self):
        # itbx_fallback=False -> kalimat identik dgn versi non-fallback (regresi tak disengaja).
        assert _rakit_kalimat_gate("Lolos", None, itbx_fallback=False) == (
            "Permohonan lolos pemeriksaan gate hukum — kegiatan dan intensitas bangunan "
            "memenuhi seluruh ketentuan yang berlaku."
        )


class TestRakitCatatanGlobal:
    def test_tanpa_meta_dan_tanpa_fallback_kosong(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(update={"meta": None})
        assert _rakit_catatan_global(assessment, itbx_fallback=False, poin_list=[]) == []

    def test_meta_caveats_muncul(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(update={"meta": MetaL2(caveats=["Data ITBX sebagian estimasi"])})
        catatan = _rakit_catatan_global(assessment, itbx_fallback=False, poin_list=[])
        assert catatan == ["Data ITBX sebagian estimasi"]

    def test_fallback_itbx_ditambahkan(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(update={"meta": None})
        catatan = _rakit_catatan_global(assessment, itbx_fallback=True, poin_list=[])
        assert len(catatan) == 1
        assert "diloloskan otomatis" in catatan[0].lower()

    def test_meta_caveats_dan_fallback_digabung(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(update={"meta": MetaL2(caveats=["Catatan lain"])})
        catatan = _rakit_catatan_global(assessment, itbx_fallback=True, poin_list=[])
        assert catatan[0] == "Catatan lain"
        assert "diloloskan otomatis" in catatan[1].lower()

    def test_tanpa_fallback_tapi_ada_poin_low_confidence_tetap_muncul(self):
        # Fix #2: poin bisa low_confidence lewat jalur lain (mis. guardrail kehabisan retry karena
        # reasoning terus melanggar aturan teks — bukan fallback ITBX, bukan meta.caveats). Sebelum
        # perbaikan ini, catatan_global akan tetap kosong padahal low_confidence_keseluruhan=True.
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(update={"meta": None})
        poin_list = [_poin_output(poin_id="itbx", low_confidence=True)]
        catatan = _rakit_catatan_global(assessment, itbx_fallback=False, poin_list=poin_list)
        assert len(catatan) == 1
        assert "poin: itbx" in catatan[0]
        assert "peninjauan manual" in catatan[0]

    def test_beberapa_poin_low_confidence_disebutkan_semua(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(update={"meta": None})
        poin_list = [
            _poin_output(poin_id="itbx", low_confidence=True),
            _poin_output(poin_id="intensitas", low_confidence=False),
            _poin_output(poin_id="dampak", low_confidence=True),
        ]
        catatan = _rakit_catatan_global(assessment, itbx_fallback=False, poin_list=poin_list)
        assert "poin: itbx, dampak" in catatan[0]

    def test_fallback_itbx_status_x_caveat_tidak_memuat_diloloskan(self):
        # APP-2026-3335: fixture nyata dgn ITBX status X — caveat global TIDAK BOLEH memakai kata
        # "diloloskan" (kontradiktif dgn verdict Tidak Lolos yang sebenarnya).
        assessment = _muat_assessment("l2_sample_tidak_lolos.json")
        assert assessment.gate_hukum.tahapan.itbx.status == "X"
        catatan = _rakit_catatan_global(assessment, itbx_fallback=True, poin_list=[])
        assert len(catatan) == 1
        assert "diloloskan" not in catatan[0].lower()
        assert "perlu verifikasi manual" in catatan[0].lower()

    def test_semua_sumber_caveat_digabung(self):
        # meta.caveats + fallback ITBX + poin low_confidence generik, ketiganya muncul sekaligus.
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(update={"meta": MetaL2(caveats=["Catatan lain"])})
        poin_list = [_poin_output(poin_id="intensitas", low_confidence=True)]
        catatan = _rakit_catatan_global(assessment, itbx_fallback=True, poin_list=poin_list)
        assert catatan[0] == "Catatan lain"
        assert "diloloskan otomatis" in catatan[1].lower()
        assert "poin: intensitas" in catatan[2]


class TestRakitRingkasanDampak:
    def test_belum_dinilai(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment.impact_assessment.dinilai = False
        assessment.impact_assessment.impact_category = None

        hasil = _rakit_ringkasan_dampak(assessment)
        assert hasil.impact_category is None
        assert hasil.impact_score is None
        assert "belum dinilai" in hasil.kalimat.lower()

    def test_kategori_ternormalisasi_dan_catatan_invers(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        hasil = _rakit_ringkasan_dampak(assessment)

        assert hasil.impact_category == "Sedang"
        assert hasil.impact_score == 65
        assert "sedang" in hasil.kalimat.lower()
        assert "invers" in hasil.kalimat.lower()

    def test_kalimat_pakai_label_dampak_terhadap_lingkungan_hidrologi(self):
        # Item permintaan user 2026-09-21: "Dampak Tata Guna Lahan" (lama) -> label baru.
        assessment = _muat_assessment("l2_sample_lolos.json")
        hasil = _rakit_ringkasan_dampak(assessment)
        assert "Dampak terhadap lingkungan (hidrologi)" in hasil.kalimat
        assert "tata guna lahan" not in hasil.kalimat.lower()


class TestRakitKesimpulan:
    def test_sukses_bersih_dipakai_apa_adanya(self, monkeypatch):
        _patch_kesimpulan_llm(
            monkeypatch,
            hasil_dict={"langkah_berdampak": ["Lengkapi dokumen X.", "Konsultasi ke dinas terkait."], "catatan_lokasi": None},
        )
        poin_list = [_poin_output()]
        hasil = _rakit_kesimpulan(poin_list, "Setuju")

        assert hasil.langkah_berdampak == ["Lengkapi dokumen X.", "Konsultasi ke dinas terkait."]
        assert hasil.catatan_lokasi is None

    def test_exception_llm_fallback_deterministik(self, monkeypatch):
        _patch_kesimpulan_llm(monkeypatch, raise_exc=RuntimeError("gagal panggilan LLM"))
        poin_list = [_poin_output(rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Saran asli poin."))]

        hasil = _rakit_kesimpulan(poin_list, "Setuju")

        assert hasil.langkah_berdampak == ["Saran asli poin."]
        assert hasil.catatan_lokasi is None

    def test_llm_sebut_angka_fallback_bukan_diloloskan(self, monkeypatch):
        _patch_kesimpulan_llm(
            monkeypatch,
            hasil_dict={"langkah_berdampak": ["Kurangi KDB hingga 60 persen."], "catatan_lokasi": None},
        )
        poin_list = [_poin_output(rekomendasi=RekomendasiOutput(tipe="numerik", saran="Saran asli poin."))]

        hasil = _rakit_kesimpulan(poin_list, "Setuju Bersyarat")

        assert hasil.langkah_berdampak == ["Saran asli poin."]  # fallback, bukan teks LLM ber-angka


class TestRakitKesimpulanFallback:
    def test_saran_kosong_difilter(self):
        poin_list = [
            _poin_output(rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Saran valid.")),
            _poin_output(poin_id="intensitas", rekomendasi=RekomendasiOutput(tipe="numerik", saran="  ")),
        ]
        hasil = _rakit_kesimpulan_fallback(poin_list)
        assert hasil.langkah_berdampak == ["Saran valid."]

    def test_urutan_dipertahankan(self):
        poin_list = [
            _poin_output(poin_id="itbx", rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Saran 1.")),
            _poin_output(poin_id="intensitas", rekomendasi=RekomendasiOutput(tipe="numerik", saran="Saran 2.")),
            _poin_output(poin_id="dampak", rekomendasi=RekomendasiOutput(tipe="numerik-mitigasi", saran="Saran 3.")),
        ]
        hasil = _rakit_kesimpulan_fallback(poin_list)
        assert hasil.langkah_berdampak == ["Saran 1.", "Saran 2.", "Saran 3."]


def _patch_narasi_llm(monkeypatch, hasil_dict=None, raise_exc=None):
    def _stub(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        if raise_exc:
            raise raise_exc
        return hasil_dict

    monkeypatch.setattr(assemble_module.llm_client, "generate", _stub)


class TestRakitNarasiRekomendasi:
    """Item permintaan user 2026-09-21: narasi 2 paragraf (ITBX+Intensitas / Dampak) — SATU
    panggilan LLM TAMBAHAN, pola identik `TestRakitKesimpulan` di atas (fallback deterministik
    kalau LLM gagal/melanggar aturan)."""

    def test_sukses_bersih_dipakai_apa_adanya(self, monkeypatch):
        _patch_narasi_llm(
            monkeypatch,
            hasil_dict={
                "paragraf_gate_intensitas": "ITBX dan intensitas bangunan memenuhi ketentuan.",
                "paragraf_dampak": "Dampak terhadap lingkungan tergolong Rendah.",
            },
        )
        poin_list = [_poin_output()]
        hasil = assemble_module._rakit_narasi_rekomendasi(poin_list, "Setuju")

        assert hasil.paragraf_gate_intensitas == "ITBX dan intensitas bangunan memenuhi ketentuan."
        assert hasil.paragraf_dampak == "Dampak terhadap lingkungan tergolong Rendah."

    def test_exception_llm_fallback_deterministik(self, monkeypatch):
        _patch_narasi_llm(monkeypatch, raise_exc=RuntimeError("gagal panggilan LLM"))
        poin_list = [
            _poin_output(poin_id="itbx", rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Saran ITBX.")),
            _poin_output(poin_id="intensitas", rekomendasi=RekomendasiOutput(tipe="numerik", saran="Saran intensitas.")),
            _poin_output(poin_id="dampak", rekomendasi=RekomendasiOutput(tipe="numerik-mitigasi", saran="Saran dampak.")),
        ]

        hasil = assemble_module._rakit_narasi_rekomendasi(poin_list, "Setuju")

        assert hasil.paragraf_gate_intensitas == "Saran ITBX. Saran intensitas."
        assert hasil.paragraf_dampak == "Saran dampak."

    def test_llm_sebut_angka_fallback_bukan_diloloskan(self, monkeypatch):
        _patch_narasi_llm(
            monkeypatch,
            hasil_dict={
                "paragraf_gate_intensitas": "KDB harus turun hingga 60 persen.",
                "paragraf_dampak": "Dampak tergolong Rendah.",
            },
        )
        poin_list = [
            _poin_output(poin_id="itbx", rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Saran asli ITBX.")),
            _poin_output(poin_id="dampak", rekomendasi=RekomendasiOutput(tipe="numerik-mitigasi", saran="Saran asli dampak.")),
        ]

        hasil = assemble_module._rakit_narasi_rekomendasi(poin_list, "Setuju Bersyarat")

        assert hasil.paragraf_gate_intensitas == "Saran asli ITBX."  # fallback, bukan teks LLM ber-angka
        assert hasil.paragraf_dampak == "Saran asli dampak."

    def test_llm_pakai_titik_koma_fallback_deterministik(self, monkeypatch):
        _patch_narasi_llm(
            monkeypatch,
            hasil_dict={
                "paragraf_gate_intensitas": "ITBX memenuhi ketentuan; intensitas juga memenuhi.",
                "paragraf_dampak": "Dampak tergolong Rendah.",
            },
        )
        poin_list = [
            _poin_output(poin_id="itbx", rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Saran asli ITBX.")),
            _poin_output(poin_id="dampak", rekomendasi=RekomendasiOutput(tipe="numerik-mitigasi", saran="Saran asli dampak.")),
        ]

        hasil = assemble_module._rakit_narasi_rekomendasi(poin_list, "Setuju")

        assert ";" not in hasil.paragraf_gate_intensitas
        assert hasil.paragraf_gate_intensitas == "Saran asli ITBX."

    def test_llm_paragraf_kosong_fallback_deterministik(self, monkeypatch):
        _patch_narasi_llm(
            monkeypatch,
            hasil_dict={"paragraf_gate_intensitas": "", "paragraf_dampak": "Dampak Rendah."},
        )
        poin_list = [_poin_output(poin_id="itbx", rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Saran asli."))]

        hasil = assemble_module._rakit_narasi_rekomendasi(poin_list, "Setuju")

        assert hasil.paragraf_gate_intensitas == "Saran asli."


class TestJalankanPrecheckEndToEnd:
    """Mock llm_client.generate (cepat/gratis) + MockRetriever. 3 fixture nyata: lolos,
    lolos_bersyarat, tidak_lolos (fixture tidak_lolos ditambah belakangan — sebelumnya ditunda,
    lihat riwayat git; TestRakitKalimatGate tetap dipertahankan utk cakupan cabang teks murni)."""

    def _stub_llm_generic(self, prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        if schema_name == "kesimpulan":
            return {"langkah_berdampak": ["Tindak lanjuti sesuai saran per-poin."], "catatan_lokasi": None}
        if schema_name == "narasi_rekomendasi":
            return {
                "paragraf_gate_intensitas": "Klasifikasi kegiatan dan intensitas bangunan sudah sesuai ketentuan.",
                "paragraf_dampak": "Dampak terhadap lingkungan tergolong dalam kategori yang dapat diterima.",
            }
        return {
            "reasoning_pendek": "Ringkasan singkat poin ini.",
            "reasoning_panjang": "Penjelasan lebih lengkap mengenai poin ini berdasarkan data yang tersedia.",
            "sitasi": [],
            "saran": "Ikuti prosedur yang berlaku.",
            "disclaimer": None,
        }

    def test_fixture_lolos_tiga_poin_dan_rekomendasi_tak_dihitung_ulang(self, monkeypatch):
        monkeypatch.setattr(assemble_module.llm_client, "generate", self._stub_llm_generic)
        _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_lolos.json")
        output = jalankan_precheck(assessment, MockRetriever())

        assert {p.poin_id for p in output.poin} == {"itbx", "intensitas", "dampak"}
        assert output.ringkasan_gate.final_gate_status == "Lolos"
        assert output.rekomendasi_sistem == adaptasi(assessment).rekomendasi_sistem == "Setuju"
        assert output.narasi_rekomendasi.paragraf_gate_intensitas.strip() != ""
        assert output.narasi_rekomendasi.paragraf_dampak.strip() != ""

    def test_fixture_lolos_bersyarat_rekomendasi_tak_dihitung_ulang(self, monkeypatch):
        monkeypatch.setattr(assemble_module.llm_client, "generate", self._stub_llm_generic)
        _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_amplop_6191.json")
        output = jalankan_precheck(assessment, MockRetriever())

        assert output.ringkasan_gate.final_gate_status == "Lolos Bersyarat"
        assert output.rekomendasi_sistem == adaptasi(assessment).rekomendasi_sistem == "Setuju Bersyarat"
        poin_intensitas = next(p for p in output.poin if p.poin_id == "intensitas")
        assert poin_intensitas.status == "MELAMPAUI_BATAS"

    def test_fixture_tidak_lolos_rekomendasi_tak_dihitung_ulang(self, monkeypatch):
        monkeypatch.setattr(assemble_module.llm_client, "generate", self._stub_llm_generic)
        _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_tidak_lolos.json")
        output = jalankan_precheck(assessment, MockRetriever())

        assert output.ringkasan_gate.final_gate_status == "Tidak Lolos"
        assert output.ringkasan_gate.decisive_stage == "itbx"
        assert output.rekomendasi_sistem == adaptasi(assessment).rekomendasi_sistem == "Tidak Setuju"

    def test_fixture_tidak_lolos_itbx_x_fallback_dipaksa_low_confidence_dan_caveat(self, monkeypatch):
        # reason back-end ("dilarang (X) atau tidak ditemukan") ambigu -> heuristik fallback_data_kosong
        # aktif -> guardrail WAJIB paksa low_confidence + caveat, walau LLM mengembalikan teks bersih.
        monkeypatch.setattr(assemble_module.llm_client, "generate", self._stub_llm_generic)
        _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_tidak_lolos.json")
        output = jalankan_precheck(assessment, MockRetriever())

        poin_itbx = next(p for p in output.poin if p.poin_id == "itbx")
        assert poin_itbx.status == "X"
        assert poin_itbx.low_confidence is True
        # APP-2026-3335: status X TIDAK BOLEH dapat caveat "diloloskan otomatis" (kontradiktif dgn
        # verdict Tidak Lolos) — caveat netral "perlu verifikasi manual" dipakai sebagai gantinya.
        assert "diloloskan" not in poin_itbx.rekomendasi.disclaimer.lower()
        assert "perlu verifikasi manual" in poin_itbx.rekomendasi.disclaimer.lower()
        assert output.low_confidence_keseluruhan is True
        assert not any("diloloskan" in c.lower() for c in output.catatan_global)
        assert any("perlu verifikasi manual" in c.lower() for c in output.catatan_global)

    def test_fixture_lolos_dengan_itbx_fallback_sintetis_ringkasan_tidak_overclaim(self, monkeypatch):
        # Fixture SINTETIS (mutasi l2_sample_lolos.json, bukan file baru) — itbx.reason mengandung
        # "kolom matriks RDTR kosong" -> heuristik fallback_data_kosong aktif, TAPI final_gate_status
        # tetap "Lolos" (TIDAK diubah, cuma narasi & flag global yg disesuaikan — Blueprint §5.2/§5.4).
        monkeypatch.setattr(assemble_module.llm_client, "generate", self._stub_llm_generic)
        _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_lolos.json")
        itbx = assessment.gate_hukum.tahapan.itbx.model_copy(
            update={"reason": "Lolos karena kolom matriks RDTR kosong untuk kegiatan ini."}
        )
        tahapan = assessment.gate_hukum.tahapan.model_copy(update={"itbx": itbx})
        gate = assessment.gate_hukum.model_copy(update={"tahapan": tahapan})
        assessment = assessment.model_copy(update={"gate_hukum": gate})

        output = jalankan_precheck(assessment, MockRetriever())

        assert output.ringkasan_gate.final_gate_status == "Lolos"  # verdict TETAP apa adanya
        assert "memenuhi seluruh ketentuan" not in output.ringkasan_gate.kalimat.lower()
        assert "belum terverifikasi" in output.ringkasan_gate.kalimat.lower()
        assert output.low_confidence_keseluruhan is True
        assert any("diloloskan otomatis" in c.lower() for c in output.catatan_global)

    def test_satu_poin_low_confidence_tanpa_fallback_tetap_muncul_di_catatan_global(self, monkeypatch):
        # Fix #2: itbx APP-2026-6191 pernah teramati jatuh ke low_confidence murni karena guardrail
        # Cek #6 (angka di narasi) kehabisan retry — reasoning-nya wajar mengutip angka ambang dari
        # keterangan_ketentuan (mis. "RTH minimal 20 persen"), BUKAN fallback ITBX/meta.caveats.
        # Sebelum perbaikan ini, catatan_global akan tetap kosong padahal low_confidence_keseluruhan
        # True — stub di bawah mereproduksi persis pola itu utk poin itbx saja.
        def _stub_itbx_selalu_sebut_angka(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
            if schema_name == "kesimpulan":
                return {"langkah_berdampak": ["Tindak lanjuti sesuai saran per-poin."], "catatan_lokasi": None}
            if "poin_id: itbx" in prompt:
                return {
                    "reasoning_pendek": "Kegiatan diperbolehkan terbatas dengan syarat RTH minimal 20 persen.",
                    "reasoning_panjang": (
                        "Kegiatan Warung diperbolehkan terbatas dengan syarat menyediakan RTH "
                        "minimal 20 persen dari luas persil sesuai ketentuan zona."
                    ),
                    "sitasi": [],
                    "saran": "Penuhi syarat RTH minimal sesuai ketentuan.",
                    "disclaimer": None,
                }
            # intensitas/dampak: sitasi valid (bukan kosong) supaya cek "chunk tersedia tapi tidak
            # disitasi" tidak ikut menandai poin ini low_confidence — isolasi kasus ke itbx saja.
            # citation_id "rdtr-p1-a107" (zona=None, general/tak terikat zona) SENGAJA dipakai, bukan
            # "...-vi-c1" (zona C-1) — fixture ini rdtr_zone="Zona Perumahan" (kode R), filter
            # zona_prefix (perbaikan APP-2026-6191) sekarang membuang chunk C-1 dari hasil search()
            # fallback krn beda keluarga zona (perilaku yg BENAR) — "rdtr-p1-a107" dijamin selalu
            # muncul di ketiga poin (top_k_dukungan=3) krn tak terikat zona apa pun.
            return {
                "reasoning_pendek": "Ringkasan singkat poin ini.",
                "reasoning_panjang": "Penjelasan lebih lengkap mengenai poin ini berdasarkan data yang tersedia.",
                "sitasi": [{"citation_id": "rdtr-p1-a107", "kutipan": "diabaikan, wajib verbatim chunk"}],
                "saran": "Ikuti prosedur yang berlaku.",
                "disclaimer": None,
            }

        monkeypatch.setattr(assemble_module.llm_client, "generate", _stub_itbx_selalu_sebut_angka)
        _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_amplop_6191.json")
        output = jalankan_precheck(assessment, MockRetriever())

        poin_itbx = next(p for p in output.poin if p.poin_id == "itbx")
        assert poin_itbx.low_confidence is True  # kehabisan retry, jatuh ke template_low_confidence
        poin_lain = [p for p in output.poin if p.poin_id != "itbx"]
        assert all(not p.low_confidence for p in poin_lain)  # bukan fallback ITBX, bukan meta.caveats

        assert output.low_confidence_keseluruhan is True
        assert output.catatan_global != []
        assert any("poin: itbx" in c and "peninjauan manual" in c for c in output.catatan_global)

    def test_fixture_tidak_lolos_dampak_belum_dinilai(self, monkeypatch):
        monkeypatch.setattr(assemble_module.llm_client, "generate", self._stub_llm_generic)
        _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_tidak_lolos.json")
        output = jalankan_precheck(assessment, MockRetriever())

        assert output.ringkasan_dampak.impact_category is None
        assert "belum dinilai" in output.ringkasan_dampak.kalimat.lower()
        poin_dampak = next(p for p in output.poin if p.poin_id == "dampak")
        assert poin_dampak.status == "Tidak Dinilai"

    def test_log_precheck_dipanggil_sekali(self, monkeypatch):
        monkeypatch.setattr(assemble_module.llm_client, "generate", self._stub_llm_generic)
        panggilan = _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_lolos.json")
        output = jalankan_precheck(assessment, MockRetriever())

        assert len(panggilan) == 1
        logged_request, logged_response, kwargs = panggilan[0]
        assert logged_request == assessment
        assert logged_response == output
        # Diagnostik ikut ke log (bukan ke OutputL3) — satu entri per poin, lengkap dgn label sebab.
        diagnostik = kwargs["diagnostik"]
        assert [d["poin_id"] for d in diagnostik] == ["itbx", "intensitas", "dampak"]
        assert all("sebab" in d for d in diagnostik)

    def test_diagnostik_tidak_bocor_ke_output_l3(self, monkeypatch):
        """Diagnostik itu data operasional — `OutputL3` adalah kontrak dgn back-end/reviewer dan
        tidak boleh membengkak olehnya."""
        monkeypatch.setattr(assemble_module.llm_client, "generate", self._stub_llm_generic)
        _patch_log(monkeypatch)

        output = jalankan_precheck(_muat_assessment("l2_sample_lolos.json"), MockRetriever())

        serialisasi = output.model_dump(mode="json")
        assert "diagnostik" not in serialisasi
        assert all("diagnostik" not in p for p in serialisasi["poin"])


class TestPemrosesanParalel:
    def test_poin_diproses_paralel_bukan_sekuensial(self, monkeypatch):
        durasi_tidur = 0.2

        def _stub_lambat(poin, retriever, assessment, *, max_retry=2):
            time.sleep(durasi_tidur)
            return _poin_output(poin_id=poin.poin_id, kategori=poin.kategori), _diagnosa_stub(poin.poin_id)

        monkeypatch.setattr(assemble_module, "generate_poin_terdiagnosis", _stub_lambat)
        _patch_kesimpulan_llm(monkeypatch, hasil_dict={"langkah_berdampak": [], "catatan_lokasi": None})
        _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_lolos.json")

        mulai = time.perf_counter()
        output = jalankan_precheck(assessment, MockRetriever())
        durasi_total = time.perf_counter() - mulai

        assert len(output.poin) == 3
        # Sekuensial akan >= 3 * durasi_tidur (~0.6s). Paralel harus jauh di bawah itu.
        assert durasi_total < 3 * durasi_tidur * 0.7

    def test_urutan_poin_sesuai_urutan_adapter(self, monkeypatch):
        def _stub_durasi_terbalik(poin, retriever, assessment, *, max_retry=2):
            durasi = {"itbx": 0.3, "intensitas": 0.15, "dampak": 0.05}[poin.poin_id]
            time.sleep(durasi)
            return _poin_output(poin_id=poin.poin_id, kategori=poin.kategori), _diagnosa_stub(poin.poin_id)

        monkeypatch.setattr(assemble_module, "generate_poin_terdiagnosis", _stub_durasi_terbalik)
        _patch_kesimpulan_llm(monkeypatch, hasil_dict={"langkah_berdampak": [], "catatan_lokasi": None})
        _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_lolos.json")
        output = jalankan_precheck(assessment, MockRetriever())

        assert [p.poin_id for p in output.poin] == ["itbx", "intensitas", "dampak"]

    def test_satu_poin_gagal_tak_terduga_tidak_menggagalkan_batch(self, monkeypatch):
        def _stub_campuran(poin, retriever, assessment, *, max_retry=2):
            if poin.poin_id == "intensitas":
                raise RuntimeError("bug tak terduga di guardrail")
            return _poin_output(poin_id=poin.poin_id, kategori=poin.kategori), _diagnosa_stub(poin.poin_id)

        monkeypatch.setattr(assemble_module, "generate_poin_terdiagnosis", _stub_campuran)
        _patch_kesimpulan_llm(monkeypatch, hasil_dict={"langkah_berdampak": [], "catatan_lokasi": None})
        _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_lolos.json")
        output = jalankan_precheck(assessment, MockRetriever())  # TIDAK BOLEH raise

        poin_by_id = {p.poin_id: p for p in output.poin}
        assert poin_by_id["itbx"].low_confidence is False
        assert poin_by_id["dampak"].low_confidence is False
        assert poin_by_id["intensitas"].low_confidence is True


class TestPagarCakupanWilayah:
    """Permohonan di luar delineasi wilayah yang dilayani tetap dijawab, tapi ditandai jelas —
    caveat di catatan_global + low_confidence_keseluruhan=True. Dua fixture nyata dipakai sbg
    pasangan: 6191 di luar delineasi Sleman Tengah, 2428 di dalam."""

    BBOX = "-7.8352,-7.6621,110.2804,110.4483"  # Perbup 80/2023 Pasal 3

    def _siapkan(self, monkeypatch, bbox: str | None):
        """Stub LLM + log_precheck. WAJIB: tanpa ini `jalankan_precheck` memanggil Groq sungguhan
        (lambat, kena rate limit) DAN menulis ke logs/precheck.jsonl yang berisi data nyata."""
        monkeypatch.setattr(
            assemble_module.llm_client, "generate", TestJalankanPrecheckEndToEnd()._stub_llm_generic
        )
        _patch_log(monkeypatch)
        if bbox is None:
            monkeypatch.delenv("CAKUPAN_BBOX", raising=False)
        else:
            monkeypatch.setenv("CAKUPAN_BBOX", bbox)

    def test_di_luar_cakupan_memicu_caveat_dan_low_confidence(self, monkeypatch):
        self._siapkan(monkeypatch, self.BBOX)
        assessment = _muat_assessment("l2_sample_amplop_6191.json")

        hasil = jalankan_precheck(assessment, MockRetriever())

        assert CAVEAT_DI_LUAR_CAKUPAN in hasil.catatan_global
        assert hasil.low_confidence_keseluruhan is True

    def test_di_dalam_cakupan_tanpa_caveat(self, monkeypatch):
        self._siapkan(monkeypatch, self.BBOX)
        assessment = _muat_assessment("l2_sample_amplop_2428.json")

        hasil = jalankan_precheck(assessment, MockRetriever())

        assert CAVEAT_DI_LUAR_CAKUPAN not in hasil.catatan_global

    def test_guard_nonaktif_tak_mengubah_apa_pun(self, monkeypatch):
        self._siapkan(monkeypatch, None)
        assessment = _muat_assessment("l2_sample_amplop_6191.json")

        hasil = jalankan_precheck(assessment, MockRetriever())

        assert CAVEAT_DI_LUAR_CAKUPAN not in hasil.catatan_global

    def test_caveat_tidak_mengklaim_yang_lolos_sudah_terverifikasi(self):
        """Bbox lebih besar dari poligon aslinya — guard cuma bisa memastikan "di LUAR". Kalimatnya
        tak boleh menyiratkan bahwa permohonan tanpa caveat sudah terbukti di dalam delineasi."""
        assert "di luar delineasi" in CAVEAT_DI_LUAR_CAKUPAN
        assert "ditinjau manual" in CAVEAT_DI_LUAR_CAKUPAN

    def test_caveat_menyatu_dgn_catatan_global_lain(self, monkeypatch):
        # Caveat cakupan TIDAK boleh menimpa/menghapus catatan low_confidence per-poin yang sudah ada.
        monkeypatch.setenv("CAKUPAN_BBOX", self.BBOX)
        assessment = _muat_assessment("l2_sample_amplop_6191.json")
        poin_list = [_poin_output(poin_id="intensitas", low_confidence=True)]

        catatan = _rakit_catatan_global(assessment, False, poin_list, True)

        assert CAVEAT_DI_LUAR_CAKUPAN in catatan
        assert any("intensitas" in c for c in catatan)


class TestCaveatSubzonaTakTerkonfirmasi:
    """Lapis DETERMINISTIK dari pasangan wajib pembukaan gating query tajam.

    Prompt juga membawa caveat yang sama (generator.caveat_subzona) supaya narasinya tak mengklaim
    lebih dari yang diketahui, tapi narasi bergantung pada LLM yang bisa gagal atau jatuh ke template
    low_confidence. Kelas ini menjaga lapis yang TIDAK bergantung LLM: pembaca tetap melihat batasnya.
    """

    def test_muncul_saat_subzona_kosong_dan_ada_poin_intensitas(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(update={"meta": None})
        poin_list = [_poin_output(poin_id="intensitas")]

        catatan = _rakit_catatan_global(assessment, False, poin_list, False, True)

        assert CAVEAT_SUBZONA_TAK_TERKONFIRMASI in catatan

    def test_tidak_muncul_saat_subzona_terkonfirmasi(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(update={"meta": None})
        poin_list = [_poin_output(poin_id="intensitas")]

        catatan = _rakit_catatan_global(assessment, False, poin_list, False, False)

        assert CAVEAT_SUBZONA_TAK_TERKONFIRMASI not in catatan

    def test_tidak_muncul_tanpa_poin_intensitas(self):
        """Tanpa poin intensitas tak ada ambang KDB/KLB/KDH yang dikutip — caveat cuma derau."""
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(update={"meta": None})
        poin_list = [_poin_output(poin_id="itbx"), _poin_output(poin_id="dampak")]

        catatan = _rakit_catatan_global(assessment, False, poin_list, False, True)

        assert CAVEAT_SUBZONA_TAK_TERKONFIRMASI not in catatan

    def test_menyatu_dgn_catatan_lain_bukan_menimpa(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(update={"meta": None})
        poin_list = [_poin_output(poin_id="intensitas", low_confidence=True)]

        catatan = _rakit_catatan_global(assessment, False, poin_list, False, True)

        assert CAVEAT_SUBZONA_TAK_TERKONFIRMASI in catatan
        assert any("peninjauan manual" in c for c in catatan)

    def test_kalimat_menyebut_sebabnya_dan_apa_yang_harus_dilakukan(self):
        """Caveat yang cuma bilang "tidak pasti" membuat pembaca menebak. Sebutkan asal angkanya
        (tabel tingkat keluarga) dan tindakan yang diminta (verifikasi sub-zona sebenarnya)."""
        assert "keluarga zona" in CAVEAT_SUBZONA_TAK_TERKONFIRMASI
        assert "diverifikasi" in CAVEAT_SUBZONA_TAK_TERKONFIRMASI
        assert "KDB/KLB/KDH" in CAVEAT_SUBZONA_TAK_TERKONFIRMASI

    # --- kabel dari jalankan_precheck: flag dihitung di sana, bukan di _rakit_catatan_global ---
    def _stub_llm(self, prompt, json_schema, *, schema_name="response", system=None,
                  temperature=0.0, max_tokens=1024):
        if schema_name == "kesimpulan":
            return {"langkah_berdampak": ["Tindak lanjuti sesuai saran per-poin."], "catatan_lokasi": None}
        if schema_name == "narasi_rekomendasi":
            return {"paragraf_gate_intensitas": "Intensitas bangunan sesuai ketentuan.",
                    "paragraf_dampak": "Dampak lingkungan dapat diterima."}
        return {"reasoning_pendek": "Ringkasan singkat.",
                "reasoning_panjang": "Penjelasan lebih lengkap mengenai poin ini.",
                "sitasi": [], "saran": "Ikuti prosedur yang berlaku.", "disclaimer": None}

    def test_e2e_caveat_muncul_saat_be_tak_kirim_subzona(self, monkeypatch):
        monkeypatch.setattr(assemble_module.llm_client, "generate", self._stub_llm)
        _patch_log(monkeypatch)

        output = jalankan_precheck(_muat_assessment("l2_sample_lolos.json"), MockRetriever())

        assert CAVEAT_SUBZONA_TAK_TERKONFIRMASI in output.catatan_global

    def test_e2e_caveat_absen_saat_be_mengirim_subzona(self, monkeypatch):
        monkeypatch.setattr(assemble_module.llm_client, "generate", self._stub_llm)
        _patch_log(monkeypatch)

        # Fixture ini membawa rdtr_subzone="R-3" -> filter exact, tak ada yang perlu disangkal.
        output = jalankan_precheck(_muat_assessment("l2_sample_amplop_5067.json"), MockRetriever())

        assert CAVEAT_SUBZONA_TAK_TERKONFIRMASI not in output.catatan_global

    def test_e2e_caveat_absen_saat_intensitas_tidak_dinilai(self, monkeypatch):
        monkeypatch.setattr(assemble_module.llm_client, "generate", self._stub_llm)
        _patch_log(monkeypatch)

        # rdtr_subzone kosong, dan BE tak mengirim penilaian intensitas. Adapter TETAP merakit
        # poin intensitas (status "Tidak Dinilai") — jadi menyaring per poin_id saja tidak cukup;
        # yang menentukan adalah ada-tidaknya ambang yang dikutip.
        output = jalankan_precheck(
            _muat_assessment("l2_sample_itbx_x_tanpa_intensitas.json"), MockRetriever())

        poin_intensitas = next(p for p in output.poin if p.poin_id == "intensitas")
        assert poin_intensitas.status == "Tidak Dinilai"
        assert CAVEAT_SUBZONA_TAK_TERKONFIRMASI not in output.catatan_global
