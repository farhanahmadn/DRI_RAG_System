import json
import time
from pathlib import Path

from app.adapter import adaptasi
from app.reasoning import assemble as assemble_module
from app.reasoning.assemble import (
    _rakit_kalimat_gate,
    _rakit_kesimpulan,
    _rakit_kesimpulan_fallback,
    _rakit_ringkasan_dampak,
    jalankan_precheck,
)
from app.retrieval.mock import MockRetriever
from app.schemas import L2Assessment, PoinOutput, RekomendasiOutput

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


def _patch_guardrail(monkeypatch, poin_by_id: dict[str, PoinOutput]):
    dipanggil = []

    def _stub(poin, retriever, assessment, *, max_retry=2):
        dipanggil.append(poin.poin_id)
        return poin_by_id[poin.poin_id]

    monkeypatch.setattr(assemble_module, "generate_poin_dengan_guardrail", _stub)
    return dipanggil


def _patch_log(monkeypatch):
    panggilan = []
    monkeypatch.setattr(
        assemble_module, "log_precheck", lambda request, response: panggilan.append((request, response))
    )
    return panggilan


def _patch_kesimpulan_llm(monkeypatch, hasil_dict=None, raise_exc=None):
    def _stub(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        if raise_exc:
            raise raise_exc
        return hasil_dict

    monkeypatch.setattr(assemble_module.llm_client, "generate", _stub)


# --- _rakit_kalimat_gate: fungsi murni, cakup "Tidak Lolos" TANPA fabrikasi L2Assessment --------
# CATATAN: kasus "Tidak Lolos" end-to-end (lewat jalankan_precheck sungguhan) DITUNDA — belum ada
# fixture asli utk kasus ini (dikonfirmasi user: tunggu contoh nyata dari back-end, jangan
# fabrikasi payload ITBX/Intensitas). Cabang teksnya tetap tercakup di sini via fungsi primitif.


class TestRakitKalimatGate:
    def test_lolos(self):
        kalimat = _rakit_kalimat_gate("Lolos", None)
        assert "lolos pemeriksaan" in kalimat.lower()
        assert "bersyarat" not in kalimat.lower()

    def test_lolos_bersyarat_tanpa_decisive_stage(self):
        kalimat = _rakit_kalimat_gate("Lolos Bersyarat", None)
        assert "lolos bersyarat" in kalimat.lower()

    def test_lolos_bersyarat_dengan_decisive_stage(self):
        kalimat = _rakit_kalimat_gate("Lolos Bersyarat", "intensitas")
        assert "intensitas bangunan" in kalimat.lower()

    def test_tidak_lolos_tanpa_decisive_stage(self):
        kalimat = _rakit_kalimat_gate("Tidak Lolos", None)
        assert "tidak lolos" in kalimat.lower()

    def test_tidak_lolos_dengan_decisive_stage_itbx(self):
        kalimat = _rakit_kalimat_gate("Tidak Lolos", "itbx")
        assert "tidak lolos" in kalimat.lower()
        assert "klasifikasi kegiatan" in kalimat.lower()

    def test_decisive_stage_tak_dikenal_ditampilkan_apa_adanya(self):
        kalimat = _rakit_kalimat_gate("Lolos Bersyarat", "tahap-baru-belum-dikenal")
        assert "tahap-baru-belum-dikenal" in kalimat


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


class TestJalankanPrecheckEndToEnd:
    """Mock llm_client.generate (cepat/gratis) + MockRetriever. HANYA 2 fixture nyata yang ada —
    kasus "Tidak Lolos" DITUNDA sampai fixture asli tersedia (lihat TestRakitKalimatGate utk
    cakupan cabang teksnya tanpa fabrikasi data)."""

    def _stub_llm_generic(self, prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        if schema_name == "kesimpulan":
            return {"langkah_berdampak": ["Tindak lanjuti sesuai saran per-poin."], "catatan_lokasi": None}
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

    def test_fixture_lolos_bersyarat_rekomendasi_tak_dihitung_ulang(self, monkeypatch):
        monkeypatch.setattr(assemble_module.llm_client, "generate", self._stub_llm_generic)
        _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_lolos_bersyarat.json")
        output = jalankan_precheck(assessment, MockRetriever())

        assert output.ringkasan_gate.final_gate_status == "Lolos Bersyarat"
        assert output.rekomendasi_sistem == adaptasi(assessment).rekomendasi_sistem == "Setuju Bersyarat"
        poin_intensitas = next(p for p in output.poin if p.poin_id == "intensitas")
        assert poin_intensitas.status == "MELAMPAUI_BATAS"

    def test_log_precheck_dipanggil_sekali(self, monkeypatch):
        monkeypatch.setattr(assemble_module.llm_client, "generate", self._stub_llm_generic)
        panggilan = _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_lolos.json")
        output = jalankan_precheck(assessment, MockRetriever())

        assert len(panggilan) == 1
        logged_request, logged_response = panggilan[0]
        assert logged_request == assessment
        assert logged_response == output


class TestPemrosesanParalel:
    def test_poin_diproses_paralel_bukan_sekuensial(self, monkeypatch):
        durasi_tidur = 0.2

        def _stub_lambat(poin, retriever, assessment, *, max_retry=2):
            time.sleep(durasi_tidur)
            return _poin_output(poin_id=poin.poin_id, kategori=poin.kategori)

        monkeypatch.setattr(assemble_module, "generate_poin_dengan_guardrail", _stub_lambat)
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
            return _poin_output(poin_id=poin.poin_id, kategori=poin.kategori)

        monkeypatch.setattr(assemble_module, "generate_poin_dengan_guardrail", _stub_durasi_terbalik)
        _patch_kesimpulan_llm(monkeypatch, hasil_dict={"langkah_berdampak": [], "catatan_lokasi": None})
        _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_lolos.json")
        output = jalankan_precheck(assessment, MockRetriever())

        assert [p.poin_id for p in output.poin] == ["itbx", "intensitas", "dampak"]

    def test_satu_poin_gagal_tak_terduga_tidak_menggagalkan_batch(self, monkeypatch):
        def _stub_campuran(poin, retriever, assessment, *, max_retry=2):
            if poin.poin_id == "intensitas":
                raise RuntimeError("bug tak terduga di guardrail")
            return _poin_output(poin_id=poin.poin_id, kategori=poin.kategori)

        monkeypatch.setattr(assemble_module, "generate_poin_dengan_guardrail", _stub_campuran)
        _patch_kesimpulan_llm(monkeypatch, hasil_dict={"langkah_berdampak": [], "catatan_lokasi": None})
        _patch_log(monkeypatch)

        assessment = _muat_assessment("l2_sample_lolos.json")
        output = jalankan_precheck(assessment, MockRetriever())  # TIDAK BOLEH raise

        poin_by_id = {p.poin_id: p for p in output.poin}
        assert poin_by_id["itbx"].low_confidence is False
        assert poin_by_id["dampak"].low_confidence is False
        assert poin_by_id["intensitas"].low_confidence is True
