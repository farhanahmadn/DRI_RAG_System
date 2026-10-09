import json
from pathlib import Path

import pytest

from app.reasoning import guardrail as guardrail_module
from app.reasoning import llm_client as llm_client_module
from app.reasoning.guardrail import (
    DiagnosaPoin,
    _bersihkan_label_teknis,
    _teks_chunk_disitasi,
    _bersihkan_disclaimer_fallback_palsu,
    _cari_band_untuk_index,
    _cek_citation_id_bocor,
    _cek_invers_skor,
    _cek_konsistensi_numerik,
    _cek_konsistensi_verdict,
    _cek_tanda_baca_dilarang,
    _dekat_dgn_pembulatan,
    _gabung_kalimat,
    _paksa_field_wajib,
    caveat_fallback_itbx,
    generate_poin_dengan_guardrail,
    generate_poin_terdiagnosis,
    perbaiki_poin,
    verifikasi_entailment_sitasi,
)
from app.retrieval.base import Chunk
from app.retrieval.mock import MockRetriever
from app.schemas import DasarHukum, L2Assessment, LangkahKonkretOutput, MetaL2, PoinKonteks, PoinOutput, RekomendasiOutput, SitasiOutput

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
        assert _cek_konsistensi_numerik(output, _poin()) != []

    def test_ada_angka_desimal_di_saran(self):
        output = _poin_output(rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Kurangi hingga 60.5 persen."))
        assert _cek_konsistensi_numerik(output, _poin()) != []

    def test_tanpa_angka_tidak_ada_masalah(self):
        output = _poin_output(
            reasoning_pendek="KDB melampaui ambang maksimum.",
            reasoning_panjang="Usulan KDB melampaui batas yang berlaku di zona ini.",
            rekomendasi=RekomendasiOutput(tipe="numerik", saran="Kurangi proporsi luas bangunan."),
        )
        assert _cek_konsistensi_numerik(output, _poin()) == []

    def test_angka_tunggal_tidak_dianggap_mencurigakan(self):
        # angka 1 digit (mis. referensi umum) tidak memicu false-positive.
        output = _poin_output(reasoning_panjang="Ketentuan diatur pada ayat 1 peraturan terkait.")
        assert _cek_konsistensi_numerik(output, _poin()) == []

    def test_angka_terlacak_ke_keterangan_ketentuan_tidak_ditolak(self):
        # Investigasi ITBX APP-2026-6191: angka ambang yang dikutip verbatim dari keterangan_ketentuan
        # (fakta sah back-end) TIDAK boleh dianggap pelanggaran — beda dari angka dikarang/dihitung LLM.
        poin = _poin(
            status="T",
            fakta={
                "lolos": True,
                "reason": "x",
                "keterangan_ketentuan": ["Dibatasi maksimum 20 dari luas blok dan atau total subzona."],
            },
        )
        output = _poin_output(
            reasoning_panjang="Kegiatan diperbolehkan terbatas, dibatasi maksimum 20 dari luas blok.",
        )
        assert _cek_konsistensi_numerik(output, poin) == []

    def test_angka_tak_terlacak_tetap_ditolak_meski_ada_keterangan_ketentuan(self):
        # Faithfulness TIDAK dilonggarkan secara umum — angka LAIN yang tak ada di sumber (mis.
        # dikarang/dihitung LLM sendiri) tetap ditolak walau poin ini punya keterangan_ketentuan.
        poin = _poin(
            status="T",
            fakta={
                "lolos": True,
                "reason": "x",
                "keterangan_ketentuan": ["Dibatasi maksimum 20 dari luas blok dan atau total subzona."],
            },
        )
        output = _poin_output(reasoning_panjang="KDB usulan sebesar 47.5 persen dari luas lahan.")
        assert _cek_konsistensi_numerik(output, poin) != []

    def test_kbli_itbx_terlacak_bukan_dianggap_karangan(self):
        # Bug ditemukan live (APP-2026-6191): kbli_diusulkan="0111" itu FAKTA sah (persis di
        # fakta['kbli_diusulkan']), tapi SEBELUM diperbaiki, itbx tak pernah dimasukkan
        # _angka_fakta_poin sama sekali -> LLM yang menyebut "KBLI 0111" SELALU ditolak, memicu
        # retry sia-sia (bahkan exhaust sampai low_confidence kalau kuota LLM habis di tengah retry).
        poin = _poin(
            status="T",
            fakta={"lolos": True, "reason": "x", "kbli_diusulkan": "0111", "kegiatan_diusulkan": "WARUNG"},
        )
        output = _poin_output(
            reasoning_panjang="Pemohon mengusulkan kegiatan WARUNG (KBLI 0111) di zona perumahan.",
        )
        assert _cek_konsistensi_numerik(output, poin) == []

    def test_intensitas_tanpa_keterangan_ketentuan_tetap_ketat_seperti_semula(self):
        # fakta={} kosong (tak ada parameter/target sama sekali) -> tidak ada angka ground truth
        # utk dilacak, provenance check tetap melarang SEMUA angka seperti perilaku lama.
        poin = _poin(poin_id="intensitas", tipe_rekomendasi="numerik", status="MELAMPAUI_BATAS", fakta={})
        output = _poin_output(poin_id="intensitas", reasoning_panjang="KDB usulan adalah 70 persen.")
        assert _cek_konsistensi_numerik(output, poin) != []

    def test_angka_terlacak_ke_dasar_hukum_kutipan(self):
        from app.schemas import DasarHukum

        poin = _poin(
            status="B",
            fakta={"lolos": True, "reason": "x"},
            dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="1", kutipan="KDB maksimum 80 persen.")],
        )
        output = _poin_output(reasoning_panjang="Ketentuan zona ini mengatur KDB maksimum 80 persen.")
        assert _cek_konsistensi_numerik(output, poin) == []

    def _poin_intensitas_dgn_fakta(self) -> PoinKonteks:
        # Fixture nyata APP-2026-6191: KDB usulan=90, ambang_maks=60 -> LOLOS BERSYARAT/MELAMPAUI_BATAS.
        return _poin(
            poin_id="intensitas",
            tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS",
            fakta={
                "parameter": {
                    "kdb": {"usulan": 90, "ambang_maks": 60, "ambang_min": None, "memenuhi": False, "satuan": "persen"},
                },
                "target": {"kdb": {"target_kdb": 60.0, "selisih": 30.0, "footprint_maks_m2": 510.0}},
                "luas_tapak_m2": 400,
            },
        )

    def test_diagnosis_intensitas_angka_usulan_ambang_terlacak_lolos(self):
        # Fix flaky-fallback intensitas: angka usulan/ambang GROUND TRUTH dari fakta poin ini sendiri
        # (bukan dari keterangan_ketentuan/dasar_hukum, yang memang selalu kosong utk intensitas)
        # kini boleh disebut verbatim di narasi.
        poin = self._poin_intensitas_dgn_fakta()
        output = _poin_output(
            poin_id="intensitas",
            reasoning_panjang="KDB usulan 90 persen melampaui batas maksimum 60 persen yang berlaku di zona ini.",
        )
        assert _cek_konsistensi_numerik(output, poin) == []

    def test_diagnosis_intensitas_angka_target_terlacak_lolos(self):
        poin = self._poin_intensitas_dgn_fakta()
        output = _poin_output(
            poin_id="intensitas",
            rekomendasi=RekomendasiOutput(tipe="numerik", saran="Turunkan KDB hingga 60.0 persen untuk memenuhi ambang."),
        )
        assert _cek_konsistensi_numerik(output, poin) == []

    def test_diagnosis_intensitas_angka_ngawur_tetap_ditolak(self):
        # Angka yang TIDAK cocok dengan fakta ground truth manapun (halusinasi/salah kutip dari
        # pasal zona lain, mis. "50" bukan ambang_maks fixture ini yang sebenarnya 60) tetap ditolak
        # — memperkuat presisi, bukan melonggarkan.
        poin = self._poin_intensitas_dgn_fakta()
        output = _poin_output(
            poin_id="intensitas",
            reasoning_panjang="KDB maksimal 50 persen sesuai ketentuan zona yang berlaku.",
        )
        assert _cek_konsistensi_numerik(output, poin) != []

    def test_diagnosis_intensitas_angka_tak_terkait_sama_sekali_tetap_ditolak(self):
        poin = self._poin_intensitas_dgn_fakta()
        output = _poin_output(poin_id="intensitas", reasoning_panjang="Selisihnya mencapai 12345 persen.")
        assert _cek_konsistensi_numerik(output, poin) != []

    def test_kasus_nyata_app_2026_8376_angka_dibulatkan_terlacak_lolos(self):
        # Fixture nyata: kdh.usulan=29.411764705882355 (presisi penuh float back-end) — LLM WAJAR
        # menulis "29.41" di narasi (tak ada yang menulis 15 digit desimal dlm kalimat), tapi
        # percobaan pertama SEBELUM diperbaiki selalu ditolak sbg "angka karangan" krn match string
        # persis gagal — memicu retry sia-sia (kadang exhaust jadi low_confidence kalau nasib buruk).
        poin = _poin(
            poin_id="intensitas",
            tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS",
            fakta={
                "parameter": {
                    "kdh": {"usulan": 29.411764705882355, "ambang_maks": None, "ambang_min": 88.0,
                            "memenuhi": False, "satuan": "persen"},
                },
            },
        )
        output = _poin_output(
            poin_id="intensitas",
            reasoning_panjang="KDH usulan 29.41% di bawah ambang minimum 88% yang berlaku di zona ini.",
        )
        assert _cek_konsistensi_numerik(output, poin) == []

    def test_diagnosis_dampak_angka_impact_score_terlacak_lolos(self):
        poin = _poin(
            poin_id="dampak",
            tipe_rekomendasi="numerik-mitigasi",
            status="Tinggi",
            fakta={"impact_score": 40, "runoff_change_index": 2.85, "c_before": 0.3, "c_after": 0.855},
        )
        output = _poin_output(
            poin_id="dampak",
            reasoning_panjang="Skor dampak 40 menunjukkan kategori Tinggi, dengan runoff_change_index 2.85.",
        )
        assert _cek_konsistensi_numerik(output, poin) == []

    def test_diagnosis_dampak_angka_ngawur_tetap_ditolak(self):
        poin = _poin(
            poin_id="dampak",
            tipe_rekomendasi="numerik-mitigasi",
            status="Tinggi",
            fakta={"impact_score": 40, "runoff_change_index": 2.85},
        )
        output = _poin_output(poin_id="dampak", reasoning_panjang="Dampaknya diperkirakan mencapai 99 persen dari total kawasan.")
        assert _cek_konsistensi_numerik(output, poin) != []

    def _poin_dampak_rekomendasi_mitigasi_be(self) -> PoinKonteks:
        return _poin(
            poin_id="dampak",
            tipe_rekomendasi="numerik-mitigasi",
            status="Tinggi",
            fakta={
                "impact_score": 40,
                "runoff_change_index": 2.923,
                "target_mitigasi": {
                    "runoff_change_index_maks": 2.5, "kategori_target": "Sedang", "index_saat_ini": 2.923,
                    "penyesuaian_lahan": {
                        "luas_bangunan_maks_m2": 11551.06, "luas_rth_min_m2": 3850.35,
                        "kdb_maks_persen": 75, "kdh_min_persen": 25,
                        "catatan": "Agar indeks runoff turun ke <= 2.5 (kategori Sedang), luas bangunan maksimal adalah 11.551,06 m² dan RTH minimal 3.850,35 m².",
                    },
                    "dimensi_minimum_resapan": {"nilai": 19.01, "satuan": "m³"},
                    "saran_be": (
                        "Untuk menurunkan dampak dari TINGGI menjadi SEDANG (indeks <= 2.5): pemohon "
                        "disarankan menyesuaikan luas lantai dasar bangunan menjadi maksimal 11.551,06 m² "
                        "(KDB maks 75%) dan menyediakan RTH minimal 3.850,35 m² (KDH min 25%), atau "
                        "menyediakan fasilitas sumur/kolam resapan air hujan dengan dimensi kapasitas "
                        "minimum 19.01 m³."
                    ),
                },
            },
        )

    def test_diagnosis_dampak_saran_be_format_indonesia_terlacak_lolos(self):
        # APP-2026-8025: `saran_be` (echo verbatim di rekomendasi.saran, generator.py::
        # _saran_mitigasi_dampak) ditulis format Indonesia (titik ribuan, koma desimal) — regex
        # ekstraksi angka guardrail tokenize BEDA dari representasi float Python. HARUS tetap lolos
        # (bukan ditolak sbg "angka karangan") krn ground truth-nya memang BE, bukan LLM.
        poin = self._poin_dampak_rekomendasi_mitigasi_be()
        output = _poin_output(
            poin_id="dampak",
            reasoning_panjang="Dampak Tinggi krn indeks limpasan 2.923 melebihi ambang 2.5.",
            rekomendasi=RekomendasiOutput(
                tipe="numerik-mitigasi",
                saran=poin.fakta["target_mitigasi"]["saran_be"],  # echo verbatim spt generator.py
            ),
        )
        assert _cek_konsistensi_numerik(output, poin) == []

    def test_diagnosis_dampak_angka_penyesuaian_lahan_plain_format_terlacak_lolos(self):
        # Angka polos (bukan format Indonesia) dari rincian BE juga harus terlacak via
        # _angka_fakta_poin (jaga-jaga LLM menulis reasoning_panjang sendiri dgn format plain).
        poin = self._poin_dampak_rekomendasi_mitigasi_be()
        output = _poin_output(
            poin_id="dampak",
            reasoning_panjang=(
                "Perlu menurunkan luas bangunan ke 11551.06 m2 dan menaikkan RTH ke 3850.35 m2, "
                "atau sumur resapan dimensi 19.01 m3."
            ),
        )
        assert _cek_konsistensi_numerik(output, poin) == []

    def test_fix1_nomor_pasal_dari_sitasi_llm_terlacak_lolos(self):
        # Fix #1 (opsional, retry sia-sia "Pasal 62"): nomor pasal yang BENAR-BENAR disitasi LLM
        # (poin_output.sitasi[].pasal) sah muncul di narasi — bukan dikarang, memang dirujuk.
        poin = self._poin_intensitas_dgn_fakta()
        output = _poin_output(
            poin_id="intensitas",
            reasoning_panjang="Sesuai Pasal 62 Ayat 4, KDB usulan 90 persen melampaui ambang maksimum 60 persen.",
            sitasi=[
                SitasiOutput(
                    citation_id="rdtr-sleman-tengah-p62-a4",
                    dokumen="RDTR Sleman Tengah",
                    pasal="62",
                    halaman=58,
                    kutipan="(4) ...",
                    terverifikasi=True,
                )
            ],
        )
        assert _cek_konsistensi_numerik(output, poin) == []

    def test_fix1_nomor_pasal_dari_dasar_hukum_poin_terlacak_lolos(self):
        from app.schemas import DasarHukum

        poin = _poin(
            poin_id="intensitas",
            tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS",
            fakta={},
            dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="62", kutipan="x")],
        )
        output = _poin_output(poin_id="intensitas", reasoning_panjang="Ketentuan ini merujuk Pasal 62 tentang intensitas.")
        assert _cek_konsistensi_numerik(output, poin) == []

    def test_fix1_angka_ngawur_bukan_nomor_pasal_manapun_tetap_ditolak(self):
        # Angka yang bukan nomor pasal manapun (disitasi atau dasar_hukum) TETAP ditolak — Fix #1
        # tidak melonggarkan cek utk angka non-pasal.
        poin = self._poin_intensitas_dgn_fakta()
        output = _poin_output(
            poin_id="intensitas",
            reasoning_panjang="Sesuai Pasal 62, selisihnya mencapai 77 persen.",
            sitasi=[
                SitasiOutput(
                    citation_id="rdtr-sleman-tengah-p62-a4",
                    dokumen="RDTR Sleman Tengah",
                    pasal="62",
                    halaman=58,
                    kutipan="(4) ...",
                    terverifikasi=True,
                )
            ],
        )
        assert _cek_konsistensi_numerik(output, poin) != []

    def test_nomor_dokumen_dari_sitasi_llm_terlacak_lolos(self):
        # Investigasi live APP-2026-3468: dokumen RAG asli bernama "Peraturan Bupati Sleman Nomor
        # 80 Tahun 2023...". LLM menyebut "Nomor 80 Tahun 2023" merujuk sumber yang BENAR-BENAR
        # disitasi -- bukan dikarang -- angkanya sah, jangan ditolak.
        poin = _poin(
            poin_id="dampak",
            kategori="Dampak Tata Guna Lahan",
            tipe_rekomendasi="numerik-mitigasi",
            status="Sedang",
            fakta={"mitigasi": {"perlu_mitigasi": False, "arah": []}},
            dasar_hukum=[],
        )
        output = _poin_output(
            poin_id="dampak",
            reasoning_panjang=(
                "Sesuai Peraturan Bupati Sleman Nomor 80 Tahun 2023, kawasan ini termasuk zona resapan air."
            ),
            sitasi=[
                SitasiOutput(
                    citation_id="rdtr-sleman-tengah-p53-a3",
                    dokumen="Peraturan Bupati Sleman Nomor 80 Tahun 2023 tentang RDTR Kawasan Sleman Tengah",
                    pasal="53",
                    halaman=40,
                    kutipan="x",
                    terverifikasi=True,
                )
            ],
        )
        assert _cek_konsistensi_numerik(output, poin) == []

    def test_nomor_dokumen_dari_dasar_hukum_poin_terlacak_lolos(self):
        from app.schemas import DasarHukum

        poin = _poin(
            poin_id="itbx",
            fakta={"lolos": True, "reason": "x"},
            dasar_hukum=[
                DasarHukum(dokumen="Peraturan Bupati Sleman Nomor 80 Tahun 2023", pasal="Matriks ITBX", kutipan="x")
            ],
        )
        output = _poin_output(
            poin_id="itbx",
            reasoning_panjang="Merujuk Peraturan Bupati Sleman Nomor 80 Tahun 2023 tentang RDTR.",
        )
        assert _cek_konsistensi_numerik(output, poin) == []

    def test_nomor_dokumen_ngawur_bukan_dokumen_manapun_tetap_ditolak(self):
        poin = _poin(
            poin_id="dampak",
            kategori="Dampak Tata Guna Lahan",
            tipe_rekomendasi="numerik-mitigasi",
            status="Sedang",
            fakta={"mitigasi": {"perlu_mitigasi": False, "arah": []}},
            dasar_hukum=[],
        )
        output = _poin_output(
            poin_id="dampak",
            reasoning_panjang="Sesuai Peraturan Nomor 99 Tahun 2023, kawasan ini termasuk zona resapan air.",
            sitasi=[
                SitasiOutput(
                    citation_id="rdtr-sleman-tengah-p53-a3",
                    dokumen="Peraturan Bupati Sleman Nomor 80 Tahun 2023 tentang RDTR Kawasan Sleman Tengah",
                    pasal="53",
                    halaman=40,
                    kutipan="x",
                    terverifikasi=True,
                )
            ],
        )
        assert _cek_konsistensi_numerik(output, poin) != []


class TestDekatDenganPembulatan:
    def test_pembulatan_2_desimal_cocok(self):
        assert _dekat_dgn_pembulatan("29.41", "kdh usulan 29.411764705882355 persen") is True

    def test_pembulatan_bilangan_bulat_cocok(self):
        assert _dekat_dgn_pembulatan("29", "kdh usulan 29.411764705882355 persen") is True

    def test_angka_persis_tanpa_pembulatan_tetap_cocok(self):
        assert _dekat_dgn_pembulatan("60.0", "target 60.0") is True

    def test_angka_beda_tidak_cocok(self):
        assert _dekat_dgn_pembulatan("25.41", "kdh usulan 29.411764705882355 persen") is False

    def test_string_bukan_angka_return_false(self):
        assert _dekat_dgn_pembulatan("bukan-angka", "kdh usulan 29.41 persen") is False

    def test_sumber_kosong_return_false(self):
        assert _dekat_dgn_pembulatan("29.41", "") is False


class TestGabungKalimat:
    def test_tambah_titik_kalau_belum_ada(self):
        # Bug live APP-2026-6191: fragmen tanpa titik trailing (mis. disclaimer LLM) nyambung ke
        # fragmen berikutnya tanpa pemisah kalau cuma " ".join(...) polos.
        hasil = _gabung_kalimat(["kalimat pertama tanpa titik", "kalimat kedua."])
        assert hasil == "kalimat pertama tanpa titik. kalimat kedua."

    def test_tidak_dobel_titik_kalau_sudah_ada(self):
        hasil = _gabung_kalimat(["kalimat pertama.", "kalimat kedua."])
        assert hasil == "kalimat pertama. kalimat kedua."

    def test_tanda_tanya_dan_seru_tidak_ditimpa(self):
        hasil = _gabung_kalimat(["Sudah benar?", "Ya!", "kalimat ketiga"])
        assert hasil == "Sudah benar? Ya! kalimat ketiga"

    def test_fragmen_kosong_dilewati(self):
        assert _gabung_kalimat(["", None, "satu-satunya isi"]) == "satu-satunya isi"

    def test_list_kosong_return_string_kosong(self):
        assert _gabung_kalimat([]) == ""

    def test_satu_fragmen_tak_berubah(self):
        assert _gabung_kalimat(["cuma satu tanpa titik"]) == "cuma satu tanpa titik"


class TestBersihkanDisclaimerFallbackPalsu:
    def test_buang_kalimat_yang_mirip_caveat_fallback(self):
        # Kasus nyata APP-2026-6191: reason bersih ("Lolos karena kegiatan Terbatas (T)...") tidak
        # memicu deteksi_fallback_itbx (dikonfirmasi False), tapi LLM tetap menulis disclaimer mirip
        # caveat fallback atas inisiatifnya sendiri — kalimat itu harus terbuang, SISANYA tetap ada.
        disclaimer = (
            "Penentuan status ini didasarkan pada data matriks RDTR yang mungkin belum lengkap — "
            "perlu verifikasi manual apakah kegiatan benar-benar dilarang. Tingkat kepercayaan data: sedang."
        )
        hasil = _bersihkan_disclaimer_fallback_palsu(disclaimer)
        assert "matriks rdtr" not in hasil.lower()
        assert "Tingkat kepercayaan data: sedang." in hasil

    def test_disclaimer_bersih_tak_berubah(self):
        assert _bersihkan_disclaimer_fallback_palsu("Tingkat kepercayaan data: sedang.") == "Tingkat kepercayaan data: sedang."

    def test_disclaimer_none_tetap_none(self):
        assert _bersihkan_disclaimer_fallback_palsu(None) is None

    def test_seluruh_disclaimer_cocok_pola_return_none(self):
        assert _bersihkan_disclaimer_fallback_palsu(
            "data matriks rdtr kosong dan perlu verifikasi manual."
        ) is None


class TestPaksaFieldWajibScrubFallbackPalsu:
    def _assessment_tanpa_meta(self) -> L2Assessment:
        assessment = _muat_assessment("l2_sample_lolos.json")
        return assessment.model_copy(update={"meta": None})

    def test_disclaimer_caveat_palsu_di_scrub_bukan_dibiarkan(self):
        poin = _poin(status="T", fakta={"lolos": True, "reason": "x", "fallback_data_kosong": False})
        output = _poin_output(
            status="T",
            rekomendasi=RekomendasiOutput(
                tipe="kategorikal", saran="x",
                disclaimer="Data matriks RDTR yang mungkin belum lengkap, perlu verifikasi manual.",
            ),
        )
        hasil = _paksa_field_wajib(output, poin, self._assessment_tanpa_meta())
        assert hasil.rekomendasi.disclaimer is None  # cuma itu isi disclaimer-nya, jadi habis di-scrub
        assert hasil.low_confidence is False  # scrub TIDAK menjatuhkan poin ke low_confidence

    def test_fallback_data_kosong_true_caveat_tidak_di_scrub(self):
        # Kalau fallback_data_kosong MEMANG True, caveat itu WAJIB ada (Cek #2) — bukan "palsu".
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x", "fallback_data_kosong": True})
        output = _poin_output(status="I")
        hasil = _paksa_field_wajib(output, poin, self._assessment_tanpa_meta())
        assert "diloloskan otomatis" in hasil.rekomendasi.disclaimer.lower()


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

    def test_fallback_itbx_status_x_caveat_tidak_memuat_diloloskan(self):
        # APP-2026-3335: status X (Tidak Lolos) + fallback data-kosong TIDAK BOLEH dapat caveat
        # "diloloskan otomatis" — kata itu kontradiktif dgn verdict Tidak Lolos yang sebenarnya.
        poin = _poin(
            poin_id="itbx",
            status="X",
            fakta={"lolos": False, "reason": "x", "fallback_data_kosong": True},
        )
        output = _poin_output(status="X", low_confidence=False)
        hasil = _paksa_field_wajib(output, poin, self._assessment_tanpa_meta())

        assert hasil.low_confidence is True
        assert "diloloskan" not in hasil.rekomendasi.disclaimer.lower()
        assert "perlu verifikasi manual" in hasil.rekomendasi.disclaimer.lower()

    def test_fallback_itbx_status_i_caveat_lama_tetap(self):
        # Regresi: status I (satu-satunya status yang benar-benar "lolos") tetap pakai caveat lama.
        assert caveat_fallback_itbx("I") == (
            "diloloskan otomatis karena data matriks RDTR kosong, bukan kepatuhan terverifikasi"
        )

    def test_fallback_itbx_status_non_i_caveat_netral(self):
        for status in ("T", "B", "TB", "X"):
            caveat = caveat_fallback_itbx(status)
            assert "diloloskan" not in caveat.lower()
            assert "perlu verifikasi manual" in caveat.lower()

    def test_meta_caveat_disuntik_tanpa_label_kepercayaan(self):
        # Item permintaan user (2026-10-02): kalimat "Tingkat kepercayaan data: ..." SENGAJA
        # DIHAPUS dari output — label generik begini bikin sistem terkesan ragu pada datanya
        # sendiri di mata reviewer, padahal tak pernah mengubah apa pun secara struktural.
        # meta.caveats (free-text spesifik dari BE) TETAP tampil — itu catatan substantif.
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(
            update={"meta": MetaL2(data_confidence_keseluruhan="Medium", caveats=["Data ITBX sebagian estimasi"])}
        )
        poin = _poin()
        output = _poin_output()
        hasil = _paksa_field_wajib(output, poin, assessment)

        assert "Data ITBX sebagian estimasi" in hasil.rekomendasi.disclaimer
        assert "DATA_CONFIDENCE" not in hasil.rekomendasi.disclaimer
        assert "Tingkat kepercayaan" not in hasil.rekomendasi.disclaimer
        assert "Medium" not in hasil.rekomendasi.disclaimer

    def test_luas_usulan_melebihi_persil_true_tambah_peringatan_disclaimer(self):
        # APP-2026-2428, Cek #3b: peringatan WAJIB dirakit deterministik di guardrail, jaring
        # pengaman kedua terlepas dari apakah LLM menyebutnya sendiri di reasoning/saran.
        poin = _poin(
            poin_id="dampak", kategori="Dampak Tata Guna Lahan", tipe_rekomendasi="numerik-mitigasi",
            status="Sedang",
            fakta={"mitigasi": {"perlu_mitigasi": False, "arah": []}, "luas_usulan_melebihi_persil": True},
        )
        output = _poin_output(poin_id="dampak", kategori="Dampak Tata Guna Lahan", status="Sedang")
        hasil = _paksa_field_wajib(output, poin, self._assessment_tanpa_meta())
        assert "melebihi luas bidang persil" in hasil.rekomendasi.disclaimer.lower()

    def test_luas_usulan_melebihi_persil_false_tidak_tambah_peringatan(self):
        poin = _poin(
            poin_id="dampak", kategori="Dampak Tata Guna Lahan", tipe_rekomendasi="numerik-mitigasi",
            status="Sedang",
            fakta={"mitigasi": {"perlu_mitigasi": False, "arah": []}, "luas_usulan_melebihi_persil": False},
        )
        output = _poin_output(poin_id="dampak", kategori="Dampak Tata Guna Lahan", status="Sedang")
        hasil = _paksa_field_wajib(output, poin, self._assessment_tanpa_meta())
        assert hasil.rekomendasi.disclaimer is None

    def test_luas_usulan_melebihi_persil_hanya_utk_poin_dampak(self):
        # Field ini cuma relevan utk poin dampak — poin itbx/intensitas TIDAK boleh dapat peringatan
        # ini walau (secara hipotetis/salah data) fakta-nya kebetulan punya key ini.
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x", "luas_usulan_melebihi_persil": True})
        output = _poin_output(status="I")
        hasil = _paksa_field_wajib(output, poin, self._assessment_tanpa_meta())
        assert hasil.rekomendasi.disclaimer is None

    def test_luas_usulan_melebihi_persil_tidak_dobel_kalau_sudah_disebut_llm(self):
        poin = _poin(
            poin_id="dampak", kategori="Dampak Tata Guna Lahan", tipe_rekomendasi="numerik-mitigasi",
            status="Sedang",
            fakta={"mitigasi": {"perlu_mitigasi": False, "arah": []}, "luas_usulan_melebihi_persil": True},
        )
        output = _poin_output(
            poin_id="dampak", kategori="Dampak Tata Guna Lahan", status="Sedang",
            reasoning_panjang=(
                "Luas usulan tapak bangunan + RTH melebihi luas bidang persil yang tercatat — hasil "
                "perhitungan dampak berikut berpotensi kurang akurat, perlu peninjauan manual."
            ),
        )
        hasil = _paksa_field_wajib(output, poin, self._assessment_tanpa_meta())
        assert hasil.rekomendasi.disclaimer is None

    def test_poin_low_confidence_tetap_sertakan_catatan_peninjauan_manual(self):
        # Catatan peninjauan manual dari template_low_confidence TETAP utuh, tidak ditimpa apa pun
        # — kalimat kepercayaan data (lama) sudah dihapus total, tes ini disempitkan maknanya.
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(update={"meta": MetaL2(data_confidence_keseluruhan="Low")})
        poin = _poin()
        output = _poin_output(
            low_confidence=True,
            rekomendasi=RekomendasiOutput(
                tipe="kategorikal",
                saran="x",
                disclaimer="Penjelasan otomatis tidak tersedia untuk poin ini. Perlu verifikasi manual.",
            ),
        )
        hasil = _paksa_field_wajib(output, poin, assessment)

        assert "Perlu verifikasi manual" in hasil.rekomendasi.disclaimer
        assert "Tingkat kepercayaan" not in hasil.rekomendasi.disclaimer


class TestPaksaFieldWajibKonsistensiIntensitas:
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

    def test_poin_aman_target_selalu_none_walau_kalkulator_kasih_angka(self):
        # Bug ditemukan live (APP-2026-8376, dampak "Sedang"): perbaiki_poin SEBELUMNYA menghitung
        # ulang target dari kalkulator TANPA cek apakah_aman() — kalau kalkulator (krn bug/gate
        # beda) tetap kasih angka utk poin yang sebenarnya aman, target itu BOCOR ke output,
        # kontradiktif dgn saran "tidak perlu tindakan". perbaiki_poin HARUS force None kalau aman,
        # apa pun yang dihitung kalkulator — pertahanan lapis-2, independen dari kebenaran kalkulator.
        poin = _poin(
            poin_id="dampak", tipe_rekomendasi="numerik-mitigasi", status="Sedang",
            fakta={
                "dinilai": True,
                "mitigasi": {"perlu_mitigasi": False, "arah": []},  # aman=True
                # target_mitigasi SENGAJA diisi (simulasi kalkulator "salah"/tak selaras) — perbaiki_poin
                # tetap TIDAK BOLEH memakainya krn poin ini aman.
                "target_mitigasi": {"runoff_change_index_maks": 1.5, "kategori_target": "Rendah", "index_saat_ini": 1.56},
            },
        )
        output = _poin_output(
            poin_id="dampak", status="Sedang",
            rekomendasi=RekomendasiOutput(tipe="numerik-mitigasi", target=1.5, saran="Tidak diperlukan tindakan khusus. Poin ini telah memenuhi ketentuan."),
        )
        poin_bersih, _ = perbaiki_poin(output, poin, [], _muat_assessment("l2_sample_lolos.json"))
        assert poin_bersih.rekomendasi.target is None
        assert poin_bersih.rekomendasi.langkah_konkret == []  # sama gerbangnya dgn target di atas

    def test_poin_tidak_aman_target_tetap_dihitung(self):
        # Kontrol negatif — poin BENAR-BENAR butuh mitigasi tetap dapat target (bukan disable total).
        poin = _poin(
            poin_id="dampak", tipe_rekomendasi="numerik-mitigasi", status="Tinggi",
            fakta={
                "dinilai": True,
                "mitigasi": {"perlu_mitigasi": True, "arah": ["turunkan KDB"]},
                "target_mitigasi": {"runoff_change_index_maks": 2.5, "kategori_target": "Sedang", "index_saat_ini": 2.85},
            },
        )
        output = _poin_output(poin_id="dampak", status="Tinggi",
                              rekomendasi=RekomendasiOutput(tipe="numerik-mitigasi", saran="Turunkan KDB."))
        poin_bersih, _ = perbaiki_poin(output, poin, [], _muat_assessment("l2_sample_lolos.json"))
        assert poin_bersih.rekomendasi.target == 2.5
        assert len(poin_bersih.rekomendasi.langkah_konkret) == 1
        assert poin_bersih.rekomendasi.langkah_konkret[0].parameter == "Indeks Limpasan (Runoff)"
        assert isinstance(poin_bersih.rekomendasi.langkah_konkret[0], LangkahKonkretOutput)  # bukan dict mentah


class TestGeneratePoinDenganGuardrail:
    def test_poin_aman_tetap_lewat_llm_dan_guardrail_saran_ditemplate(self, monkeypatch):
        # APP-2026-3468: poin aman TIDAK LAGI short-circuit ke template generik — lewat LLM+guardrail
        # penuh spt poin lain. reasoning_pendek/panjang & sitasi TETAP dari LLM (menjelaskan KENAPA
        # lolos); hanya saran/target yang ditemplate deterministik.
        from app.schemas import DasarHukum

        panggilan = {"n": 0}

        def _stub(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
            panggilan["n"] += 1
            return {
                "reasoning_pendek": "Kegiatan termasuk kategori Diizinkan (I) di zona ini.",
                "reasoning_panjang": (
                    "Kegiatan yang diusulkan termasuk kategori Diizinkan (I) sesuai Matriks ITBX zona ini."
                ),
                "sitasi": [{"citation_id": "anchor-0", "kutipan": "Kutipan dari anchor."}],
                "saran": "Saran dari LLM ini HARUS diabaikan/ditimpa oleh template.",
                "disclaimer": None,
            }

        monkeypatch.setattr(llm_client_module, "generate", _stub)

        poin = _poin(
            status="I",
            fakta={"lolos": True, "reason": "x"},
            dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="Matriks ITBX", kutipan="Data KBLI referensi")],
        )
        assessment = _muat_assessment("l2_sample_lolos.json")
        hasil = generate_poin_dengan_guardrail(poin, MockRetriever(), assessment)

        assert panggilan["n"] == 1
        assert hasil.status == "I"
        assert hasil.reasoning_pendek == "Kegiatan termasuk kategori Diizinkan (I) di zona ini."
        assert len(hasil.sitasi) == 1
        assert hasil.rekomendasi.saran == "Tidak diperlukan tindakan khusus. Poin ini telah memenuhi ketentuan."
        assert hasil.low_confidence is False

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

    def test_exception_generate_poin_dilog_bukan_ditelan_diam(self, monkeypatch, caplog):
        # Bug ditemukan live (migrasi model 2026-08-15): exception di generate_poin (rate limit,
        # BadRequestError, dst) SEBELUMNYA ditelan tanpa jejak begitu retry habis & jatuh ke
        # template_low_confidence — tak bisa dibedakan dari "model memang lemah" pasca-kejadian.
        def _stub_raise(*args, **kwargs):
            raise RuntimeError("simulasi RateLimitError")

        monkeypatch.setattr(llm_client_module, "generate", _stub_raise)

        poin = _poin(status="B", fakta={"lolos": True, "reason": "x"})
        assessment = _muat_assessment("l2_sample_lolos.json")
        with caplog.at_level("WARNING", logger="app.reasoning.guardrail"):
            generate_poin_dengan_guardrail(poin, MockRetriever(), assessment, max_retry=1)

        assert "simulasi RateLimitError" in caplog.text
        assert "itbx" in caplog.text


def test_verifikasi_entailment_sitasi_stub_selalu_true():
    assert verifikasi_entailment_sitasi(_poin_output(), []) is True


class TestDiagnosaSebab:
    """`DiagnosaPoin.sebab()` — label kasar yang memisahkan tiga sebab low_confidence yang selama
    ini tak terbedakan di logs/precheck.jsonl (51% permohonan nyata punya minimal 1 poin
    low_confidence, tanpa satu pun petunjuk kenapa)."""

    def test_berhasil(self):
        assert DiagnosaPoin(poin_id="itbx", berhasil=True).sebab() == "berhasil"

    def test_panggilan_llm_gagal_menang_atas_sebab_lain(self):
        # Rate limit / timeout: sebab paling akar, walau masalah guardrail juga terisi.
        d = DiagnosaPoin(
            poin_id="itbx", berhasil=False, jumlah_chunk=3,
            masalah_terakhir=["sitasi kosong"], exception_terakhir="RateLimitError: 429",
        )
        assert d.sebab() == "panggilan_llm_gagal"

    def test_retrieval_kosong(self):
        d = DiagnosaPoin(poin_id="itbx", berhasil=False, jumlah_chunk=0, masalah_terakhir=["sitasi kosong"])
        assert d.sebab() == "retrieval_kosong"

    def test_retrieval_provider_gagal_dibedakan_dari_llm_gagal(self):
        # APP-2026-INNER-01 (2026-10-02, live): Jina API menolak panggilan embed (403 "Insufficient
        # account balance") SEBELUM LLM sempat dipanggil — label lama "panggilan_llm_gagal"
        # menyesatkan (user mengira bug reasoning, padahal provider retrieval eksternal). Dibedakan
        # via penanda pesan `_provider_http.py::post_json` ("Panggilan provider '<nama>' gagal...").
        d = DiagnosaPoin(
            poin_id="intensitas", berhasil=False, jumlah_chunk=0,
            exception_terakhir=(
                "RuntimeError: Panggilan provider 'jina-embed' gagal setelah 3 percobaan: "
                "Client error '403 Forbidden' for url 'https://api.jina.ai/v1/embeddings'"
            ),
        )
        assert d.sebab() == "retrieval_provider_gagal"

    def test_exception_llm_asli_tetap_panggilan_llm_gagal(self):
        # Kontrol negatif — exception dari SDK Groq (RateLimitError/BadRequestError/dst, tanpa
        # penanda "Panggilan provider") TETAP "panggilan_llm_gagal" spt semula.
        d = DiagnosaPoin(poin_id="itbx", berhasil=False, exception_terakhir="RateLimitError: 429")
        assert d.sebab() == "panggilan_llm_gagal"

    def test_guardrail_menolak(self):
        d = DiagnosaPoin(poin_id="itbx", berhasil=False, jumlah_chunk=3, masalah_terakhir=["verdict kontradiktif"])
        assert d.sebab() == "guardrail_menolak"

    def test_tak_diketahui_saat_tak_ada_petunjuk(self):
        assert DiagnosaPoin(poin_id="itbx", berhasil=False, jumlah_chunk=3).sebab() == "tak_diketahui"


class TestGeneratePoinTerdiagnosis:
    """Diagnosa dari jalur sungguhan — perilaku PoinOutput-nya wajib identik dgn
    `generate_poin_dengan_guardrail` (yang kini cuma pembungkus tipis)."""

    def test_llm_gagal_terus_sebab_panggilan_llm_gagal(self, monkeypatch):
        def _stub_raise(*args, **kwargs):
            raise RuntimeError("simulasi RateLimitError")

        monkeypatch.setattr(llm_client_module, "generate", _stub_raise)
        poin = _poin(status="B", fakta={"lolos": True, "reason": "x"})
        assessment = _muat_assessment("l2_sample_lolos.json")

        hasil, diagnosa = generate_poin_terdiagnosis(poin, MockRetriever(), assessment, max_retry=1)

        assert hasil.low_confidence is True
        assert diagnosa.berhasil is False
        assert diagnosa.percobaan == 2                      # max_retry=1 -> 2 percobaan
        assert diagnosa.sebab() == "panggilan_llm_gagal"
        assert "simulasi RateLimitError" in diagnosa.exception_terakhir

    def test_guardrail_menolak_terus_sebab_guardrail_menolak(self, monkeypatch):
        def _stub_reasoning_kosong(prompt, json_schema, *, schema_name="response", system=None,
                                   temperature=0.0, max_tokens=1024):
            return {"reasoning_pendek": "", "reasoning_panjang": "", "sitasi": [],
                    "saran": "", "disclaimer": None}

        monkeypatch.setattr(llm_client_module, "generate", _stub_reasoning_kosong)
        poin = _poin(status="B", fakta={"lolos": True, "reason": "x"})
        assessment = _muat_assessment("l2_sample_lolos.json")

        hasil, diagnosa = generate_poin_terdiagnosis(poin, MockRetriever(), assessment, max_retry=1)

        assert hasil.low_confidence is True
        assert diagnosa.exception_terakhir is None          # LLM-nya menjawab, isinya yang ditolak
        assert diagnosa.masalah_terakhir, "temuan guardrail terakhir harus terekam, bukan dibuang"
        assert diagnosa.sebab() == "guardrail_menolak"

    def test_exception_lalu_berhasil_tidak_salah_label(self, monkeypatch):
        """Percobaan 1 kena exception, percobaan 2 lolos -> `berhasil`, BUKAN
        `panggilan_llm_gagal`. Tanpa reset eksplisit, exception basi akan salah melabeli."""
        n = {"i": 0}

        def _stub(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
            n["i"] += 1
            if n["i"] == 1:
                raise RuntimeError("gagal sekali")
            return {
                "reasoning_pendek": "Kegiatan termasuk kategori Bersyarat (B) di zona ini.",
                "reasoning_panjang": "Kegiatan yang diusulkan termasuk kategori Bersyarat (B) sesuai ketentuan zona.",
                "sitasi": [{"citation_id": "anchor-0", "kutipan": "Kutipan dari anchor."}],
                "saran": "Penuhi persyaratan yang ditetapkan sebelum kegiatan dijalankan.",
                "disclaimer": None,
            }

        monkeypatch.setattr(llm_client_module, "generate", _stub)
        poin = _poin(
            status="B",
            fakta={"lolos": True, "reason": "x"},
            dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="Matriks ITBX", kutipan="Data KBLI referensi")],
        )
        assessment = _muat_assessment("l2_sample_lolos.json")

        hasil, diagnosa = generate_poin_terdiagnosis(poin, MockRetriever(), assessment, max_retry=2)

        assert hasil.low_confidence is False
        assert diagnosa.berhasil is True
        assert diagnosa.percobaan == 2
        assert diagnosa.exception_terakhir is None
        assert diagnosa.sebab() == "berhasil"

    def test_teks_ditolak_direkam_tapi_tidak_masuk_prompt_retry(self, monkeypatch):
        """Kalimat yang ditolak wajib terekam di diagnosa, TAPI tidak boleh bocor ke
        `catatan_perbaikan` — daftar `masalah` dirangkai jadi prompt retry, jadi menambahinya
        berarti mengubah perilaku yang justru sedang diukur."""
        prompt_retry = []

        def _stub(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
            prompt_retry.append(prompt)
            return {
                "reasoning_pendek": "Mengacu pada Pasal 10 ketentuan zona ini.",
                "reasoning_panjang": "Berdasarkan Pasal 10, kegiatan wajib memenuhi ketentuan yang berlaku.",
                "sitasi": [],
                "saran": "Penuhi ketentuan tersebut.",
                "disclaimer": None,
            }

        monkeypatch.setattr(llm_client_module, "generate", _stub)
        poin = _poin(status="B", fakta={"lolos": True, "reason": "x"})
        assessment = _muat_assessment("l2_sample_lolos.json")

        _, diagnosa = generate_poin_terdiagnosis(poin, MockRetriever(), assessment, max_retry=1)

        assert diagnosa.teks_ditolak_terakhir, "kalimat yang ditolak harus terekam"
        assert "Pasal 10" in diagnosa.teks_ditolak_terakhir
        # Prompt retry hanya boleh memuat label temuan, bukan kalimat yang kita rekam utk diagnosa.
        assert len(prompt_retry) == 2, "harus ada 1 retry"
        assert diagnosa.teks_ditolak_terakhir not in prompt_retry[1]

    def test_teks_ditolak_kosong_saat_poin_berhasil(self, monkeypatch):
        def _stub(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
            return {
                "reasoning_pendek": "Kegiatan termasuk kategori Bersyarat (B) di zona ini.",
                "reasoning_panjang": "Kegiatan yang diusulkan termasuk kategori Bersyarat (B) sesuai ketentuan zona.",
                "sitasi": [{"citation_id": "anchor-0", "kutipan": "Kutipan dari anchor."}],
                "saran": "Penuhi persyaratan yang ditetapkan sebelum kegiatan dijalankan.",
                "disclaimer": None,
            }

        monkeypatch.setattr(llm_client_module, "generate", _stub)
        poin = _poin(
            status="B",
            fakta={"lolos": True, "reason": "x"},
            dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="Matriks ITBX", kutipan="Data KBLI referensi")],
        )
        assessment = _muat_assessment("l2_sample_lolos.json")

        _, diagnosa = generate_poin_terdiagnosis(poin, MockRetriever(), assessment, max_retry=1)

        assert diagnosa.berhasil is True
        assert diagnosa.teks_ditolak_terakhir is None

    def test_pembungkus_lama_mengembalikan_poinoutput_yang_sama(self, monkeypatch):
        def _stub_raise(*args, **kwargs):
            raise RuntimeError("gagal")

        monkeypatch.setattr(llm_client_module, "generate", _stub_raise)
        poin = _poin(status="B", fakta={"lolos": True, "reason": "x"})
        assessment = _muat_assessment("l2_sample_lolos.json")

        lama = generate_poin_dengan_guardrail(poin, MockRetriever(), assessment, max_retry=1)
        baru, _ = generate_poin_terdiagnosis(poin, MockRetriever(), assessment, max_retry=1)

        assert lama == baru

    def test_fallback_dilog_dgn_sebabnya(self, monkeypatch, caplog):
        def _stub_raise(*args, **kwargs):
            raise RuntimeError("simulasi RateLimitError")

        monkeypatch.setattr(llm_client_module, "generate", _stub_raise)
        poin = _poin(status="B", fakta={"lolos": True, "reason": "x"})
        assessment = _muat_assessment("l2_sample_lolos.json")

        with caplog.at_level("WARNING", logger="app.reasoning.guardrail"):
            generate_poin_terdiagnosis(poin, MockRetriever(), assessment, max_retry=1)

        assert "low_confidence" in caplog.text
        assert "sebab=panggilan_llm_gagal" in caplog.text


class TestRiwayatPercobaan:
    """`DiagnosaPoin.riwayat_percobaan` — alasan penolakan percobaan PERTAMA tidak boleh hilang.

    Kenapa kelas ini ada (2026-10-09). `masalah_terakhir` menyimpan percobaan TERAKHIR, jadi poin
    yang lolos di percobaan ke-2/ke-3 menimpanya jadi `[]` — terverifikasi di
    logs/precheck.jsonl 2026-10-02T06:44:22 (intensitas & dampak, percobaan=3, berhasil=true,
    masalah_terakhir=[]). Yang hilang di situ justru satu-satunya rekaman tentang apa yang
    guardrail tolak dari keluaran LLM MENTAH, yaitu penyebut bagi
    `eval/metrik_generasi.py::laju_tolak_guardrail`.
    """

    def test_tolak_lalu_lolos_tetap_merekam_penolakan_pertama(self, monkeypatch):
        """Inti perbaikannya: percobaan 1 ditolak guardrail, percobaan 2 lolos. `masalah_terakhir`
        WAJIB kosong (poin ini memang akhirnya bersih) — dan riwayatnya WAJIB tidak kosong."""
        n = {"i": 0}

        def _stub(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
            n["i"] += 1
            if n["i"] == 1:
                # Titik koma -> ditolak _cek_tanda_baca_dilarang (aturan #17), bukan exception.
                return {"reasoning_pendek": "Kegiatan termasuk kategori Bersyarat (B) di zona ini.",
                        "reasoning_panjang": "Kegiatan termasuk kategori Bersyarat (B); wajib memenuhi ketentuan.",
                        "sitasi": [{"citation_id": "anchor-0", "kutipan": "Kutipan dari anchor."}],
                        "saran": "Penuhi persyaratan yang ditetapkan sebelum kegiatan dijalankan.",
                        "disclaimer": None}
            return {"reasoning_pendek": "Kegiatan termasuk kategori Bersyarat (B) di zona ini.",
                    "reasoning_panjang": "Kegiatan yang diusulkan termasuk kategori Bersyarat (B) sesuai ketentuan zona.",
                    "sitasi": [{"citation_id": "anchor-0", "kutipan": "Kutipan dari anchor."}],
                    "saran": "Penuhi persyaratan yang ditetapkan sebelum kegiatan dijalankan.",
                    "disclaimer": None}

        monkeypatch.setattr(llm_client_module, "generate", _stub)
        poin = _poin(
            status="B", fakta={"lolos": True, "reason": "x"},
            dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="Matriks ITBX", kutipan="Data KBLI referensi")],
        )
        assessment = _muat_assessment("l2_sample_lolos.json")

        _, diagnosa = generate_poin_terdiagnosis(poin, MockRetriever(), assessment, max_retry=2)

        assert diagnosa.berhasil is True
        assert diagnosa.masalah_terakhir == [], "percobaan terakhir memang bersih"
        assert [e["percobaan"] for e in diagnosa.riwayat_percobaan] == [1, 2]
        assert [e["sebab"] for e in diagnosa.riwayat_percobaan] == ["guardrail_menolak", "berhasil"]
        assert diagnosa.riwayat_percobaan[0]["masalah"], \
            "alasan penolakan percobaan pertama TIDAK boleh hilang — justru ini datanya"
        assert any("titik koma" in m for m in diagnosa.riwayat_percobaan[0]["masalah"])
        assert diagnosa.riwayat_percobaan[1]["masalah"] == []

    def test_lolos_di_percobaan_pertama_punya_satu_entri(self, monkeypatch):
        """Penyebut `laju_tolak_guardrail` dihitung dari riwayat, jadi percobaan yang LOLOS pun
        harus punya entri — tanpa itu, "sampai ke guardrail dan diterima" tak terhitung."""
        def _stub(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
            return {"reasoning_pendek": "Kegiatan termasuk kategori Bersyarat (B) di zona ini.",
                    "reasoning_panjang": "Kegiatan yang diusulkan termasuk kategori Bersyarat (B) sesuai ketentuan zona.",
                    "sitasi": [{"citation_id": "anchor-0", "kutipan": "Kutipan dari anchor."}],
                    "saran": "Penuhi persyaratan yang ditetapkan sebelum kegiatan dijalankan.",
                    "disclaimer": None}

        monkeypatch.setattr(llm_client_module, "generate", _stub)
        poin = _poin(
            status="B", fakta={"lolos": True, "reason": "x"},
            dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="Matriks ITBX", kutipan="Data KBLI referensi")],
        )
        assessment = _muat_assessment("l2_sample_lolos.json")

        _, diagnosa = generate_poin_terdiagnosis(poin, MockRetriever(), assessment, max_retry=2)

        assert diagnosa.riwayat_percobaan == [
            {"percobaan": 1, "sebab": "berhasil", "masalah": [], "exception": None}
        ]

    def test_ditolak_terus_merekam_tiap_percobaan(self, monkeypatch):
        def _stub_kosong(prompt, json_schema, *, schema_name="response", system=None,
                         temperature=0.0, max_tokens=1024):
            return {"reasoning_pendek": "", "reasoning_panjang": "", "sitasi": [],
                    "saran": "", "disclaimer": None}

        monkeypatch.setattr(llm_client_module, "generate", _stub_kosong)
        poin = _poin(status="B", fakta={"lolos": True, "reason": "x"})
        assessment = _muat_assessment("l2_sample_lolos.json")

        _, diagnosa = generate_poin_terdiagnosis(poin, MockRetriever(), assessment, max_retry=2)

        assert len(diagnosa.riwayat_percobaan) == 3
        assert {e["sebab"] for e in diagnosa.riwayat_percobaan} == {"guardrail_menolak"}
        assert all(e["masalah"] for e in diagnosa.riwayat_percobaan)

    def test_percobaan_yang_gagal_di_llm_tidak_dilabeli_temuan_guardrail(self, monkeypatch):
        """Exception LLM dicatat dgn `masalah: []` + `exception` terisi. Kalau pesan exception ikut
        masuk `masalah`, sebaran label di `laju_tolak_guardrail` akan mengarang temuan guardrail
        bernama "RateLimitError: 429" — dan laju tolaknya ikut salah karena penyebutnya berbeda.
        Terbukti nyata di log: 8 baris 2026-09-08 memuat `TypeError: build_user_prompt() ...
        'konteks_induk'` di dalam `masalah_terakhir`, padahal itu bug pengembangan, bukan temuan."""
        n = {"i": 0}

        def _stub(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
            n["i"] += 1
            if n["i"] == 1:
                raise RuntimeError("simulasi RateLimitError: 429")
            return {"reasoning_pendek": "Kegiatan termasuk kategori Bersyarat (B) di zona ini.",
                    "reasoning_panjang": "Kegiatan yang diusulkan termasuk kategori Bersyarat (B) sesuai ketentuan zona.",
                    "sitasi": [{"citation_id": "anchor-0", "kutipan": "Kutipan dari anchor."}],
                    "saran": "Penuhi persyaratan yang ditetapkan sebelum kegiatan dijalankan.",
                    "disclaimer": None}

        monkeypatch.setattr(llm_client_module, "generate", _stub)
        poin = _poin(
            status="B", fakta={"lolos": True, "reason": "x"},
            dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="Matriks ITBX", kutipan="Data KBLI referensi")],
        )
        assessment = _muat_assessment("l2_sample_lolos.json")

        _, diagnosa = generate_poin_terdiagnosis(poin, MockRetriever(), assessment, max_retry=2)

        assert [e["sebab"] for e in diagnosa.riwayat_percobaan] == ["panggilan_llm_gagal", "berhasil"]
        assert diagnosa.riwayat_percobaan[0]["masalah"] == []
        assert "RateLimitError: 429" in diagnosa.riwayat_percobaan[0]["exception"]

    def test_retrieval_gagal_sebelum_llm_meninggalkan_riwayat_kosong(self, monkeypatch):
        """Retrieval raise -> LLM tak pernah dipanggil, jadi guardrail tak pernah menilai apa pun.
        Riwayat WAJIB kosong supaya poin ini keluar dari penyebut laju tolak, bukan terhitung
        sebagai percobaan yang diterima (APP-2026-INNER-01: kuota Jina habis, bukan model lemah)."""
        def _gagal(*args, **kwargs):
            raise RuntimeError("Panggilan provider 'jina-embed' gagal setelah 3 percobaan: 403")

        monkeypatch.setattr(guardrail_module, "ambil_chunks_pendukung", _gagal)
        poin = _poin(status="B", fakta={"lolos": True, "reason": "x"})
        assessment = _muat_assessment("l2_sample_lolos.json")

        with pytest.raises(RuntimeError):
            generate_poin_terdiagnosis(poin, MockRetriever(), assessment, max_retry=1)

    def test_default_kosong_bukan_dibagi_antar_instans(self):
        """`field(default_factory=list)` — dua diagnosa tak boleh berbagi daftar yang sama."""
        a, b = DiagnosaPoin(poin_id="itbx", berhasil=False), DiagnosaPoin(poin_id="dampak", berhasil=False)
        a.riwayat_percobaan.append({"percobaan": 1, "sebab": "guardrail_menolak",
                                    "masalah": ["x"], "exception": None})
        assert b.riwayat_percobaan == []


class TestAngkaDariChunkYangDisitasi:
    """Pelonggaran provenance angka ke teks chunk RAG — TAPI hanya chunk yang benar-benar DISITASI.

    Bukti (replay 8 fixture, 2026-09-08): angka yang ditolak `_cek_konsistensi_numerik` ternyata
    nilai asli Lampiran VI untuk sub-zona pemohon ("KDB maksimum 10%, KLB 0.1, KDH 85-88% untuk zona
    P-1"), dikutip verbatim dari chunk yang sistem sodorkan sendiri lengkap dgn atribusi pasalnya —
    bukan halusinasi. Whitelist lama tak pernah mencakup isi chunk, jadi retry selalu terbuang dan
    poin jatuh ke low_confidence."""

    CHUNK_VI = Chunk(
        id="rdtr-sleman-tengah-vi-p-1",
        level="tabel",
        teks="Lampiran VI — Zona P-1: KDB maksimum 10%, KLB maksimum 0.1, KDH minimum 85%.",
        dokumen="Perbup Sleman 80/2023",
    )

    @staticmethod
    def _output(teks: str, citation_ids: tuple[str, ...]) -> PoinOutput:
        return PoinOutput(
            poin_id="itbx",
            kategori="Klasifikasi Kegiatan (ITBX)",
            status="I",
            reasoning_pendek=teks,
            reasoning_panjang=teks,
            sitasi=[
                # pasal/halaman sengaja TANPA digit yang dipakai kasus uji (10/77) — `pasal` ikut
                # masuk `sumber` di _angka_terlacak_ke_sumber, jadi angka di situ akan mencemari hasil.
                SitasiOutput(
                    citation_id=cid, dokumen="Perbup Sleman", pasal="Lampiran VI", halaman=1,
                    kutipan="x", terverifikasi=True,
                )
                for cid in citation_ids
            ],
            rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Ikuti ketentuan."),
        )

    def test_angka_dari_chunk_yang_disitasi_diterima(self):
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x"})
        out = self._output(
            "Lampiran VI menetapkan KDB maksimum 10% untuk zona P-1.",
            ("rdtr-sleman-tengah-vi-p-1",),
        )

        assert _cek_konsistensi_numerik(out, poin, [self.CHUNK_VI]) == []

    def test_angka_dari_chunk_yang_TIDAK_disitasi_tetap_ditolak(self):
        """Inti pembedaannya: chunk disodorkan ke LLM, tapi tidak disitasi -> angka yang diseret
        darinya tetap ditolak. Tanpa syarat ini, model bebas menarik ambang intensitas ke narasi
        ITBX/dampak tanpa menyebut sumbernya — pola yang justru terlihat di replay."""
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x"})
        out = self._output("Ketentuan menetapkan KDB maksimum 10% untuk zona ini.", ())

        masalah = _cek_konsistensi_numerik(out, poin, [self.CHUNK_VI])

        assert masalah and "10" in masalah[0]

    def test_tanpa_chunk_perilaku_lama_dipertahankan(self):
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x"})
        out = self._output("Ketentuan menetapkan KDB maksimum 10%.", ("rdtr-sleman-tengah-vi-p-1",))

        assert _cek_konsistensi_numerik(out, poin, None) != []

    def test_angka_karangan_tetap_ditolak_walau_chunk_disitasi(self):
        # 77 tidak ada di chunk manapun — pelonggaran ini TIDAK boleh jadi pintu masuk halusinasi.
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x"})
        out = self._output("Ketentuan menetapkan KDB maksimum 77%.", ("rdtr-sleman-tengah-vi-p-1",))

        masalah = _cek_konsistensi_numerik(out, poin, [self.CHUNK_VI])

        assert masalah and "77" in masalah[0]


class TestTeksChunkDisitasi:
    def test_hanya_chunk_yang_id_nya_disitasi(self):
        a = Chunk(id="chunk-a", level="pasal", teks="isi A", dokumen="d")
        b = Chunk(id="chunk-b", level="pasal", teks="isi B", dokumen="d")
        out = PoinOutput(
            poin_id="itbx", kategori="k", status="I", reasoning_pendek="x", reasoning_panjang="x",
            sitasi=[SitasiOutput(citation_id="chunk-b", dokumen="d", pasal="p", halaman=1, kutipan="x", terverifikasi=True)],
            rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="x"),
        )

        hasil = _teks_chunk_disitasi(out, [a, b])

        assert "isi B" in hasil and "isi A" not in hasil

    def test_aman_saat_output_atau_chunk_kosong(self):
        assert _teks_chunk_disitasi(None, None) == ""
        assert _teks_chunk_disitasi(None, [Chunk(id="c", level="pasal", teks="t", dokumen="d")]) == ""


class TestBersihkanLabelTeknis:
    """Label teknis prompt yang tersalin mentah ke narasi petugas. Scrub DETERMINISTIK (tidak memicu
    retry) — alasan sama dgn _bersihkan_disclaimer_fallback_palsu: satu kata salah tak sebanding dgn
    membuang seluruh reasoning + sitasi yang sudah benar."""

    def test_label_dikenal_diganti_bahasa_biasa(self):
        assert _bersihkan_label_teknis("KATEGORI_DAMPAK: Rendah menunjukkan...").startswith("kategori dampak:")
        assert "kelengkapan data matriks RDTR" in _bersihkan_label_teknis("FALLBACK_DATA_KOSONG bernilai False")

    def test_akronim_sah_tidak_ikut_tergilas(self):
        # Tanpa syarat garis bawah, akronim domain akan rusak — ini yang menjaga RDTR/KDB/ITBX utuh.
        teks = "Zona RDTR dengan KDB, KLB, KDH, ITBX, LP2B, PBG, RTH tetap utuh"
        assert _bersihkan_label_teknis(teks) == teks

    def test_label_tak_dikenal_tetap_dinormalkan(self):
        # Label BARU yang ditambahkan ke prompt nanti tidak boleh diam-diam bocor lagi.
        assert _bersihkan_label_teknis("nilai TOKEN_BARU_X di sini") == "nilai token baru x di sini"

    def test_none_dan_kosong_aman(self):
        assert _bersihkan_label_teknis(None) is None
        assert _bersihkan_label_teknis("") == ""

    def test_diterapkan_ke_semua_field_narasi(self):
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x"})
        assessment = _muat_assessment("l2_sample_lolos.json")
        keluaran = PoinOutput(
            poin_id="itbx", kategori="Klasifikasi Kegiatan (ITBX)", status="I",
            reasoning_pendek="STATUS_ITBX: I",
            reasoning_panjang="KATEGORI_DAMPAK tidak relevan di sini",
            sitasi=[],
            rekomendasi=RekomendasiOutput(
                tipe="kategorikal", saran="Perhatikan KEGIATAN_DIUSULKAN.",
                disclaimer="FALLBACK_DATA_KOSONG bernilai False.",
            ),
        )

        hasil = _paksa_field_wajib(keluaran, poin, assessment)

        gabungan = f"{hasil.reasoning_pendek} {hasil.reasoning_panjang} {hasil.rekomendasi.saran} {hasil.rekomendasi.disclaimer}"
        for token in ("STATUS_ITBX", "KATEGORI_DAMPAK", "KEGIATAN_DIUSULKAN", "FALLBACK_DATA_KOSONG"):
            assert token not in gabungan, f"{token} masih bocor ke narasi"


class TestCekCitationIdBocor:
    """Item permintaan user 2026-09-21 (SYSTEM_PROMPT aturan #17): citation_id (ID baris basis
    data) TIDAK boleh muncul mentah di reasoning/saran/disclaimer — reviewer tak paham/tak perlu
    tahu ID internal, kalau ingin merujuk sumber harus pakai nama dokumen/nomor pasal."""

    def test_anchor_id_bocor_di_reasoning_terdeteksi(self):
        from app.schemas import DasarHukum

        poin = _poin(dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="Matriks ITBX", kutipan="x")])
        output = _poin_output(reasoning_panjang="Sesuai anchor-0, kegiatan ini diizinkan.")
        masalah = _cek_citation_id_bocor(output, poin, [])
        assert any("anchor-0" in m for m in masalah)

    def test_chunk_id_bocor_di_saran_terdeteksi(self):
        chunk = Chunk(id="rdtr-sleman-tengah-p53-a3", level="ayat", teks="x", dokumen="RDTR Sleman", pasal="53")
        output = _poin_output(
            rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Lihat rdtr-sleman-tengah-p53-a3 untuk detail."),
        )
        masalah = _cek_citation_id_bocor(output, _poin(), [chunk])
        assert any("rdtr-sleman-tengah-p53-a3" in m for m in masalah)

    def test_chunk_id_bocor_di_disclaimer_terdeteksi(self):
        chunk = Chunk(id="rdtr-sleman-tengah-p53-a3", level="ayat", teks="x", dokumen="RDTR Sleman", pasal="53")
        output = _poin_output(
            rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="x", disclaimer="Rujukan: rdtr-sleman-tengah-p53-a3."),
        )
        masalah = _cek_citation_id_bocor(output, _poin(), [chunk])
        assert any("rdtr-sleman-tengah-p53-a3" in m for m in masalah)

    def test_id_yang_tak_pernah_disodorkan_tidak_dicek(self):
        # ID yang bahkan tak ada di daftar kandidat (anchor/chunk) tidak relevan dicek di sini —
        # kalau muncul di narasi berarti dikarang sepenuhnya, ditangkap cek lain (konsistensi angka).
        output = _poin_output(reasoning_panjang="Sesuai rdtr-entah-mana-p99, kegiatan diizinkan.")
        assert _cek_citation_id_bocor(output, _poin(), []) == []

    def test_menyebut_nama_dokumen_pasal_bukan_id_tidak_terdeteksi(self):
        chunk = Chunk(id="rdtr-sleman-tengah-p53-a3", level="ayat", teks="x", dokumen="RDTR Sleman", pasal="53")
        output = _poin_output(reasoning_panjang="Sesuai Pasal 53 RDTR Sleman Tengah, kegiatan ini diizinkan.")
        assert _cek_citation_id_bocor(output, _poin(), [chunk]) == []


class TestCekTandaBacaDilarang:
    """Item permintaan user 2026-09-21 (SYSTEM_PROMPT aturan #17): titik koma bukan gaya bahasa
    umum bagi pembaca non-teknis — DILARANG di reasoning/saran/disclaimer manapun."""

    def test_titik_koma_di_reasoning_panjang_terdeteksi(self):
        output = _poin_output(reasoning_panjang="Kegiatan ini diizinkan; tidak ada catatan tambahan.")
        assert _cek_tanda_baca_dilarang(output) != []

    def test_titik_koma_di_saran_terdeteksi(self):
        output = _poin_output(rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Lengkapi dokumen X; ajukan izin Y."))
        assert _cek_tanda_baca_dilarang(output) != []

    def test_titik_koma_di_disclaimer_terdeteksi(self):
        output = _poin_output(
            rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="x", disclaimer="Data belum lengkap; perlu verifikasi."),
        )
        assert _cek_tanda_baca_dilarang(output) != []

    def test_tanpa_titik_koma_tidak_terdeteksi(self):
        output = _poin_output(
            reasoning_panjang="Kegiatan ini diizinkan. Tidak ada catatan tambahan.",
            rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Lengkapi dokumen X, lalu ajukan izin Y."),
        )
        assert _cek_tanda_baca_dilarang(output) == []
