import json
from pathlib import Path

from app.reasoning import llm_client as llm_client_module
from app.reasoning.guardrail import (
    _bersihkan_disclaimer_fallback_palsu,
    _cari_band_untuk_index,
    _cek_invers_skor,
    _cek_konsistensi_numerik,
    _cek_konsistensi_verdict,
    _gabung_kalimat,
    _kalimat_tingkat_kepercayaan,
    _paksa_field_wajib,
    caveat_fallback_itbx,
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

    def test_meta_caveat_dan_data_confidence_disuntik(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(
            update={"meta": MetaL2(data_confidence_keseluruhan="Medium", caveats=["Data ITBX sebagian estimasi"])}
        )
        poin = _poin()
        output = _poin_output()
        hasil = _paksa_field_wajib(output, poin, assessment)

        assert "Tingkat kepercayaan data: sedang." in hasil.rekomendasi.disclaimer
        assert "Data ITBX sebagian estimasi" in hasil.rekomendasi.disclaimer
        assert "DATA_CONFIDENCE" not in hasil.rekomendasi.disclaimer
        assert "Medium" not in hasil.rekomendasi.disclaimer

    def test_data_confidence_konsisten_utk_ketiga_poin(self):
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(update={"meta": MetaL2(data_confidence_keseluruhan="High")})

        for poin_id in ("itbx", "intensitas", "dampak"):
            poin = _poin(poin_id=poin_id, kategori="x")
            output = _poin_output(poin_id=poin_id, kategori="x")
            hasil = _paksa_field_wajib(output, poin, assessment)
            assert "Tingkat kepercayaan data: tinggi." in hasil.rekomendasi.disclaimer
            assert "DATA_CONFIDENCE" not in hasil.rekomendasi.disclaimer

    def test_poin_low_confidence_tetap_sertakan_catatan_peninjauan_manual(self):
        # Fix #4: kalimat kepercayaan (baru) HARUS berdampingan dengan catatan peninjauan manual
        # yang sudah ada dari template_low_confidence — bukan menimpanya.
        assessment = _muat_assessment("l2_sample_lolos.json")
        assessment = assessment.model_copy(update={"meta": MetaL2(data_confidence_keseluruhan="Low")})
        poin = _poin()
        output = _poin_output(
            low_confidence=True,
            rekomendasi=RekomendasiOutput(
                tipe="kategorikal",
                saran="x",
                disclaimer="Penjelasan otomatis tidak tersedia untuk poin ini; perlu verifikasi manual.",
            ),
        )
        hasil = _paksa_field_wajib(output, poin, assessment)

        assert "perlu verifikasi manual" in hasil.rekomendasi.disclaimer
        assert "Tingkat kepercayaan data: rendah." in hasil.rekomendasi.disclaimer


class TestKalimatTingkatKepercayaan:
    def test_high_jadi_tinggi(self):
        assert _kalimat_tingkat_kepercayaan("High") == "Tingkat kepercayaan data: tinggi."

    def test_medium_jadi_sedang(self):
        assert _kalimat_tingkat_kepercayaan("Medium") == "Tingkat kepercayaan data: sedang."

    def test_low_jadi_rendah(self):
        assert _kalimat_tingkat_kepercayaan("Low") == "Tingkat kepercayaan data: rendah."

    def test_case_insensitive(self):
        assert _kalimat_tingkat_kepercayaan("high") == "Tingkat kepercayaan data: tinggi."
        assert _kalimat_tingkat_kepercayaan("MEDIUM") == "Tingkat kepercayaan data: sedang."

    def test_none_tidak_tampilkan_label(self):
        assert _kalimat_tingkat_kepercayaan(None) is None

    def test_nilai_tak_dikenal_tidak_tampilkan_label(self):
        assert _kalimat_tingkat_kepercayaan("Sangat Tinggi Sekali") is None

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
        assert hasil.rekomendasi.saran == "Tidak diperlukan tindakan khusus; poin ini telah memenuhi ketentuan."
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


def test_verifikasi_entailment_sitasi_stub_selalu_true():
    assert verifikasi_entailment_sitasi(_poin_output(), []) is True
