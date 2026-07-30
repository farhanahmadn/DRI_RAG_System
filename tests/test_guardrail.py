import json
from pathlib import Path

from app.reasoning import llm_client as llm_client_module
from app.reasoning.guardrail import (
    _cari_band_untuk_index,
    _cek_invers_skor,
    _cek_konsistensi_numerik,
    _cek_konsistensi_verdict,
    _paksa_field_wajib,
    generate_poin_dengan_guardrail,
    perbaiki_poin,
    verifikasi_entailment_sitasi,
)
from app.retrieval.base import Chunk
from app.retrieval.mock import MockRetriever
from app.schemas import L2Assessment, MetaL2, PoinKonteks, PoinOutput, RekomendasiOutput, SitasiOutput

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _muat_assessment(nama_file: str) -> L2Assessment:
    payload = json.loads((FIXTURES_DIR / nama_file).read_text(encoding="utf-8"))
    return L2Assessment.model_validate(payload["data"])


def _poin(**overrides) -> PoinKonteks:
    defaults = dict(
        poin_id="itbx",
        kategori="Klasifikasi Kegiatan (ITBX)",
        tipe_rekomendasi="kategorikal",
        status="I",
        fakta={"lolos": True, "reason": "Lolos karena kegiatan Diizinkan (I)"},
        dasar_hukum=[],
    )
    defaults.update(overrides)
    return PoinKonteks(**defaults)


def _poin_output(**overrides) -> PoinOutput:
    defaults = dict(
        poin_id="itbx",
        kategori="Klasifikasi Kegiatan (ITBX)",
        status="I",
        reasoning_pendek="x",
        reasoning_panjang="x",
        sitasi=[],
        rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="x"),
        low_confidence=False,
    )
    defaults.update(overrides)
    return PoinOutput(**defaults)


class _RetrieverYangMelarangDipanggil:
    def search(self, query, filters, top_k=5):
        raise AssertionError("search() tidak boleh dipanggil untuk poin yang aman")

    def get_by_reference(self, referensi):
        raise AssertionError("get_by_reference() tidak boleh dipanggil untuk poin yang aman")

    def get_parent(self, chunk_id):
        raise AssertionError("get_parent() tidak boleh dipanggil untuk poin yang aman")


class TestCariBandUntukIndex:
    def test_format_kurang_dari_dgn_prefix(self):
        bands = {"Rendah": "index < 1.5", "Sedang": "1.5-2.5", "Tinggi": "2.5-4.0", "Sangat Tinggi": "> 4.0"}
        assert _cari_band_untuk_index(1.0, bands) == "Rendah"

    def test_format_rentang(self):
        bands = {"Rendah": "index < 1.5", "Sedang": "1.5-2.5", "Tinggi": "2.5-4.0", "Sangat Tinggi": "> 4.0"}
        assert _cari_band_untuk_index(1.7833333333333334, bands) == "Sedang"

    def test_format_lebih_dari(self):
        bands = {"Rendah": "index < 1.5", "Sedang": "1.5-2.5", "Tinggi": "2.5-4.0", "Sangat Tinggi": "> 4.0"}
        assert _cari_band_untuk_index(5.0, bands) == "Sangat Tinggi"

    def test_tidak_ada_yang_cocok(self):
        assert _cari_band_untuk_index(1.0, {"Sedang": "1.5-2.5"}) is None


class TestCekInversSkor:
    def _poin_dampak(self, status="Sedang", index=1.78, bands=None) -> PoinKonteks:
        return _poin(
            poin_id="dampak",
            tipe_rekomendasi="numerik-mitigasi",
            status=status,
            fakta={
                "runoff_change_index": index,
                "threshold_bands": bands
                or {"Rendah": "index < 1.5", "Sedang": "1.5-2.5", "Tinggi": "2.5-4.0", "Sangat Tinggi": "> 4.0"},
            },
        )

    def test_bukan_poin_dampak_tidak_dicek(self):
        assert _cek_invers_skor(_poin_output(), _poin()) == []

    def test_teks_terbalik_kategori_rendah_sedang(self):
        poin = self._poin_dampak(status="Sedang")
        output = _poin_output(
            poin_id="dampak",
            reasoning_pendek="Skor 65 berarti risiko tinggi bagi lingkungan.",
            reasoning_panjang="x",
        )
        masalah = _cek_invers_skor(output, poin)
        assert any("dampak tinggi" in m or "risiko tinggi" in m for m in masalah)

    def test_teks_terbalik_kategori_tinggi(self):
        poin = self._poin_dampak(status="Tinggi", index=3.0)
        output = _poin_output(poin_id="dampak", reasoning_panjang="Kondisi ini tergolong dampak rendah.")
        masalah = _cek_invers_skor(output, poin)
        assert any("dampak rendah" in m for m in masalah)

    def test_teks_benar_tidak_ada_masalah(self):
        poin = self._poin_dampak(status="Sedang")
        output = _poin_output(
            poin_id="dampak", reasoning_pendek="Dampak tata guna lahan tergolong sedang.", reasoning_panjang="x"
        )
        assert _cek_invers_skor(output, poin) == []

    def test_data_index_dan_kategori_konsisten_fixture_nyata(self):
        # index=1.7833 (fixture nyata) jatuh di band Sedang, cocok dgn kategori "Sedang".
        poin = self._poin_dampak(status="Sedang", index=1.7833333333333334)
        output = _poin_output(poin_id="dampak")
        assert _cek_invers_skor(output, poin) == []

    def test_data_index_dan_kategori_tidak_konsisten(self):
        # index jatuh di band Tinggi tapi kategori bilang Sedang -> data back-end sendiri tak konsisten.
        poin = self._poin_dampak(status="Sedang", index=3.0)
        output = _poin_output(poin_id="dampak")
        masalah = _cek_invers_skor(output, poin)
        assert any("tak konsisten" in m for m in masalah)


class TestCekKonsistensiVerdict:
    def test_itbx_x_tanpa_verdict_larangan_ditegaskan(self):
        poin = _poin(status="X", fakta={"lolos": False, "reason": "x"})
        output = _poin_output(status="X", reasoning_panjang="Kegiatan ini diizinkan di zona tersebut.")
        masalah = _cek_konsistensi_verdict(output, poin)
        assert any("larangan" in m for m in masalah)

    def test_itbx_x_menyebut_kegiatan_alternatif_diizinkan_tidak_false_positive(self):
        # Regresi bug nyata (ditemukan lewat eval/run_eval live): SYSTEM_PROMPT aturan #7
        # mewajibkan LLM menyebut kegiatan ALTERNATIF yang diizinkan di zona ini saat status=X —
        # kata "diizinkan" WAJAR muncul merujuk kegiatan lain, bukan kegiatan yang diusulkan.
        # Cek negatif lama ('diizinkan' dilarang muncul sama sekali) false-positive di kasus ini.
        poin = _poin(status="X", fakta={"lolos": False, "reason": "x"})
        output = _poin_output(
            status="X",
            reasoning_panjang=(
                "Kegiatan Industri Besar/Pabrik dilarang di zona perumahan. Sebagai gantinya, "
                "kegiatan seperti Rumah Tunggal dan Warung dapat diizinkan di zona ini."
            ),
        )
        assert _cek_konsistensi_verdict(output, poin) == []

    def test_itbx_i_tidak_dicek(self):
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x"})
        output = _poin_output(status="I", reasoning_panjang="Kegiatan ini diizinkan di zona tersebut.")
        assert _cek_konsistensi_verdict(output, poin) == []

    def test_intensitas_memenuhi_syarat_tapi_teks_bilang_melanggar(self):
        poin = _poin(poin_id="intensitas", tipe_rekomendasi="numerik", status="MEMENUHI_SYARAT", fakta={})
        output = _poin_output(poin_id="intensitas", reasoning_panjang="KDB melanggar ambang yang berlaku.")
        masalah = _cek_konsistensi_verdict(output, poin)
        assert any("MEMENUHI_SYARAT" in m for m in masalah)

    def test_intensitas_melampaui_batas_tapi_teks_bilang_memenuhi_semua(self):
        poin = _poin(poin_id="intensitas", tipe_rekomendasi="numerik", status="MELAMPAUI_BATAS", fakta={})
        output = _poin_output(poin_id="intensitas", reasoning_panjang="Bangunan ini memenuhi seluruh standar yang berlaku.")
        masalah = _cek_konsistensi_verdict(output, poin)
        assert any("MELAMPAUI_BATAS" in m for m in masalah)

    def test_teks_konsisten_tidak_ada_masalah(self):
        poin = _poin(poin_id="intensitas", tipe_rekomendasi="numerik", status="MELAMPAUI_BATAS", fakta={})
        output = _poin_output(poin_id="intensitas", reasoning_panjang="KDB yang diusulkan melampaui ambang maksimum zona ini.")
        assert _cek_konsistensi_verdict(output, poin) == []


class TestCekKonsistensiNumerik:
    def test_ada_angka_di_reasoning(self):
        output = _poin_output(reasoning_panjang="KDB usulan adalah 70 persen.")
        assert _cek_konsistensi_numerik(output) != []

    def test_ada_angka_desimal_di_saran(self):
        output = _poin_output(rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Kurangi hingga 60.5 persen."))
        assert _cek_konsistensi_numerik(output) != []

    def test_tanpa_angka_tidak_ada_masalah(self):
        output = _poin_output(
            reasoning_pendek="KDB melampaui ambang maksimum.",
            reasoning_panjang="Usulan KDB melampaui batas yang berlaku di zona ini.",
            rekomendasi=RekomendasiOutput(tipe="numerik", saran="Kurangi proporsi luas bangunan."),
        )
        assert _cek_konsistensi_numerik(output) == []

    def test_angka_tunggal_tidak_dianggap_mencurigakan(self):
        # angka 1 digit (mis. referensi umum) tidak memicu false-positive.
        output = _poin_output(reasoning_panjang="Ketentuan diatur pada ayat 1 peraturan terkait.")
        assert _cek_konsistensi_numerik(output) == []


class TestPaksaFieldWajib:
    def _assessment_tanpa_meta(self) -> L2Assessment:
        assessment = _muat_assessment("l2_sample_lolos.json")
        return assessment.model_copy(update={"meta": None})

    def test_fallback_itbx_paksa_low_confidence_dan_caveat(self):
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x", "fallback_data_kosong": True})
        output = _poin_output(status="I", low_confidence=False)
        hasil = _paksa_field_wajib(output, poin, self._assessment_tanpa_meta())

        assert hasil.low_confidence is True
        assert "diloloskan otomatis" in hasil.rekomendasi.disclaimer.lower()

    def test_fallback_itbx_tidak_dobel_kalau_caveat_sudah_ada(self):
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x", "fallback_data_kosong": True})
        output = _poin_output(
            status="I",
            reasoning_panjang="Diloloskan otomatis karena data matriks RDTR kosong, bukan kepatuhan terverifikasi.",
        )
        hasil = _paksa_field_wajib(output, poin, self._assessment_tanpa_meta())
        assert hasil.rekomendasi.disclaimer is None

    def test_tanpa_fallback_tidak_dipaksa(self):
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x", "fallback_data_kosong": False})
        output = _poin_output(status="I", low_confidence=False)
        hasil = _paksa_field_wajib(output, poin, self._assessment_tanpa_meta())
        assert hasil.low_confidence is False
        assert hasil.rekomendasi.disclaimer is None

    def test_meta_caveat_dan_data_confidence_disuntik(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(
            update={"meta": MetaL2(data_confidence_keseluruhan="Medium", caveats=["Data ITBX sebagian estimasi"])}
        )
        poin = _poin()
        output = _poin_output()
        hasil = _paksa_field_wajib(output, poin, assessment)

        assert "Medium" in hasil.rekomendasi.disclaimer
        assert "Data ITBX sebagian estimasi" in hasil.rekomendasi.disclaimer

    def test_cek_konsistensi_intensitas_men_trigger_low_confidence(self):
        assessment = _muat_assessment("l2_sample_amplop_6191.json")
        # rusak jadi tak konsisten: status dipaksa MEMENUHI_SYARAT tapi param kdb tetap memenuhi=False.
        intensitas = assessment.gate_hukum.tahapan.intensitas.model_copy(update={"status": "MEMENUHI_SYARAT"})
        tahapan = assessment.gate_hukum.tahapan.model_copy(update={"intensitas": intensitas})
        gate = assessment.gate_hukum.model_copy(update={"tahapan": tahapan})
        assessment = assessment.model_copy(update={"gate_hukum": gate})

        poin = _poin(poin_id="intensitas", tipe_rekomendasi="numerik", status="MEMENUHI_SYARAT", fakta={})
        output = _poin_output(poin_id="intensitas", status="MEMENUHI_SYARAT", low_confidence=False)
        hasil = _paksa_field_wajib(output, poin, assessment)

        assert hasil.low_confidence is True
        assert hasil.status == "MEMENUHI_SYARAT"  # TIDAK ditimpa, tetap apa adanya

    def test_intensitas_konsisten_tidak_dipaksa(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        poin = _poin(poin_id="intensitas", tipe_rekomendasi="numerik", status="MEMENUHI_SYARAT", fakta={})
        output = _poin_output(poin_id="intensitas", status="MEMENUHI_SYARAT", low_confidence=False)
        hasil = _paksa_field_wajib(output, poin, assessment)
        assert hasil.low_confidence is False


class TestPerbaikiPoin:
    def test_status_dan_tipe_dipaksa_dari_poin(self):
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x"})
        output = _poin_output(status="SALAH", rekomendasi=RekomendasiOutput(tipe="numerik", saran="x"))
        poin_bersih, masalah = perbaiki_poin(output, poin, [], _muat_assessment("l2_sample_lolos.json"))
        assert poin_bersih.status == "I"
        assert poin_bersih.rekomendasi.tipe == "kategorikal"

    def test_sitasi_citation_id_tak_dikenal_difilter(self):
        chunk = Chunk(id="c1", level="pasal", teks="x", dokumen="RDTR Sleman")
        output = _poin_output(
            sitasi=[
                SitasiOutput(citation_id="c1", dokumen="RDTR Sleman", pasal="", halaman=0, kutipan="x", terverifikasi=True),
                SitasiOutput(citation_id="tak-dikenal", dokumen="x", pasal="", halaman=0, kutipan="x", terverifikasi=True),
            ]
        )
        poin_bersih, _ = perbaiki_poin(output, _poin(), [chunk], _muat_assessment("l2_sample_lolos.json"))
        assert len(poin_bersih.sitasi) == 1
        assert poin_bersih.sitasi[0].citation_id == "c1"

    def test_reasoning_kosong_jadi_masalah(self):
        output = _poin_output(reasoning_pendek="", reasoning_panjang="x")
        _, masalah = perbaiki_poin(output, _poin(), [], _muat_assessment("l2_sample_lolos.json"))
        assert masalah


class TestGeneratePoinDenganGuardrail:
    def test_poin_aman_pakai_template_tanpa_retrieval_atau_llm(self, monkeypatch):
        dipanggil = {"llm": 0}
        monkeypatch.setattr(
            llm_client_module, "generate", lambda *a, **k: dipanggil.__setitem__("llm", dipanggil["llm"] + 1)
        )

        poin = _poin(status="I", fakta={"lolos": True, "reason": "x"})
        assessment = _muat_assessment("l2_sample_lolos.json")
        hasil = generate_poin_dengan_guardrail(poin, _RetrieverYangMelarangDipanggil(), assessment)

        assert hasil.status == "I"
        assert dipanggil["llm"] == 0

    def test_llm_bersih_percobaan_pertama_langsung_return(self, monkeypatch):
        panggilan = {"n": 0}

        def _stub(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
            panggilan["n"] += 1
            return {
                "reasoning_pendek": "KDB melampaui ambang maksimum.",
                "reasoning_panjang": "Usulan KDB melampaui ambang maksimum yang berlaku di zona ini.",
                "sitasi": [{"citation_id": "rdtr-lampiran-vi-c1", "kutipan": "KDB maksimum 80%."}],
                "saran": "Kurangi proporsi luas bangunan terhadap luas lahan.",
                "disclaimer": None,
            }

        monkeypatch.setattr(llm_client_module, "generate", _stub)

        poin = _poin(
            poin_id="intensitas",
            # kategori tak lagi menentukan query fallback utk poin_id="intensitas" (selalu "kdb",
            # lihat generator.py::_QUERY_FALLBACK_PER_POIN) — "kdb" MEMANG match rdtr-lampiran-vi-c1
            # di MockRetriever, jadi stub sitasi di atas harus mengutip chunk itu (bukan kosong).
            tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS",
            fakta={"target": {"kdb": {"target_kdb": 60.0, "selisih": 10.0}}},
        )
        assessment = _muat_assessment("l2_sample_amplop_6191.json")
        hasil = generate_poin_dengan_guardrail(poin, MockRetriever(), assessment)

        assert panggilan["n"] == 1
        assert hasil.rekomendasi.target == 60.0
        assert hasil.low_confidence is False

    def test_llm_gagal_lalu_bersih_diperbaiki_via_retry(self, monkeypatch):
        panggilan = {"n": 0}

        def _stub(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
            panggilan["n"] += 1
            if panggilan["n"] == 1:
                return {
                    "reasoning_pendek": "KDB usulan adalah 70 persen.",  # angka -> ditolak cek #6
                    "reasoning_panjang": "x",
                    "sitasi": [{"citation_id": "rdtr-lampiran-vi-c1", "kutipan": "KDB maksimum 80%."}],
                    "saran": "x",
                    "disclaimer": None,
                }
            return {
                "reasoning_pendek": "KDB melampaui ambang maksimum.",
                "reasoning_panjang": "Usulan KDB melampaui ambang maksimum yang berlaku di zona ini.",
                "sitasi": [{"citation_id": "rdtr-lampiran-vi-c1", "kutipan": "KDB maksimum 80%."}],
                "saran": "Kurangi proporsi luas bangunan terhadap luas lahan.",
                "disclaimer": None,
            }

        monkeypatch.setattr(llm_client_module, "generate", _stub)

        poin = _poin(
            poin_id="intensitas",
            # kategori tak lagi menentukan query fallback utk poin_id="intensitas" (selalu "kdb").
            tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS",
            fakta={"target": {"kdb": {"target_kdb": 60.0, "selisih": 10.0}}},
        )
        assessment = _muat_assessment("l2_sample_amplop_6191.json")
        hasil = generate_poin_dengan_guardrail(poin, MockRetriever(), assessment)

        assert panggilan["n"] == 2
        assert hasil.low_confidence is False
        assert hasil.reasoning_pendek == "KDB melampaui ambang maksimum."

    def test_llm_selalu_gagal_jatuh_ke_template_low_confidence(self, monkeypatch):
        panggilan = {"n": 0}

        def _stub(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
            panggilan["n"] += 1
            return {
                "reasoning_pendek": "KDB usulan adalah 70 persen.",  # selalu mengandung angka -> selalu gagal
                "reasoning_panjang": "x",
                "sitasi": [],
                "saran": "x",
                "disclaimer": None,
            }

        monkeypatch.setattr(llm_client_module, "generate", _stub)

        poin = _poin(
            poin_id="intensitas",
            kategori="Intensitas Bangunan (KDB/KLB/KDH)",
            tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS",
            fakta={"target": {"kdb": {"target_kdb": 60.0, "selisih": 10.0}}},
        )
        assessment = _muat_assessment("l2_sample_amplop_6191.json")
        hasil = generate_poin_dengan_guardrail(poin, MockRetriever(), assessment, max_retry=2)

        assert panggilan["n"] == 3  # max_retry=2 -> 3 percobaan total, JANGAN loop tak terbatas
        assert hasil.low_confidence is True
        assert hasil.rekomendasi.target == 60.0  # tetap dari calculator meski LLM gagal terus

    def test_exception_generate_poin_dihitung_sbg_percobaan_gagal(self, monkeypatch):
        panggilan = {"n": 0}

        def _stub_raise(*args, **kwargs):
            panggilan["n"] += 1
            raise RuntimeError("simulasi gagal panggilan LLM")

        monkeypatch.setattr(llm_client_module, "generate", _stub_raise)

        poin = _poin(status="B", fakta={"lolos": True, "reason": "x"})
        assessment = _muat_assessment("l2_sample_lolos.json")
        hasil = generate_poin_dengan_guardrail(poin, MockRetriever(), assessment, max_retry=1)

        assert panggilan["n"] == 2  # max_retry=1 -> 2 percobaan
        assert hasil.low_confidence is True


def test_verifikasi_entailment_sitasi_stub_selalu_true():
    assert verifikasi_entailment_sitasi(_poin_output(), []) is True
