import json
import time
from pathlib import Path

from app.adapter import adaptasi
from app.reasoning import assemble as assemble_module
from app.reasoning.assemble import (
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
    """Mock llm_client.generate (cepat/gratis) + MockRetriever. 3 fixture nyata: lolos,
    lolos_bersyarat, tidak_lolos (fixture tidak_lolos ditambah belakangan — sebelumnya ditunda,
    lihat riwayat git; TestRakitKalimatGate tetap dipertahankan utk cakupan cabang teks murni)."""

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
            return {
                "reasoning_pendek": "Ringkasan singkat poin ini.",
                "reasoning_panjang": "Penjelasan lebih lengkap mengenai poin ini berdasarkan data yang tersedia.",
                "sitasi": [{"citation_id": "rdtr-lampiran-vi-c1", "kutipan": "diabaikan, wajib verbatim chunk"}],
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
