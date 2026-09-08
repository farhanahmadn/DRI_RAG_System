from app.reasoning.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_KESIMPULAN,
    _angka_prompt,
    build_kesimpulan_prompt,
    build_user_prompt,
)
from app.retrieval.base import Chunk
from app.schemas import DasarHukum, MetaL2, PoinKonteks, PoinOutput, RekomendasiOutput


def _poin(**overrides) -> PoinKonteks:
    defaults = dict(
        poin_id="itbx",
        kategori="Klasifikasi Kegiatan (ITBX)",
        tipe_rekomendasi="kategorikal",
        status="I",
        fakta={"lolos": True, "reason": "Lolos karena kegiatan Diizinkan (I) di zona Zona Perumahan"},
        dasar_hukum=[],
    )
    defaults.update(overrides)
    return PoinKonteks(**defaults)


def _chunk(**overrides) -> Chunk:
    defaults = dict(
        id="rdtr-matriks-itbx",
        level="tabel",
        teks="Matriks ITBX zona perumahan.",
        dokumen="RDTR Sleman",
        pasal="Matriks ITBX",
        halaman=12,
    )
    defaults.update(overrides)
    return Chunk(**defaults)


def test_system_prompt_memuat_semua_aturan_wajib():
    assert "FINAL" in SYSTEM_PROMPT
    assert "SKOR DAMPAK INVERS" in SYSTEM_PROMPT
    assert "skor TINGGI berarti dampak RENDAH" in SYSTEM_PROMPT
    assert "FALLBACK_DATA_KOSONG" in SYSTEM_PROMPT
    assert "diloloskan otomatis karena data matriks RDTR kosong" in SYSTEM_PROMPT
    assert "MAKNA X GANDA" in SYSTEM_PROMPT
    assert "JANGAN default ke" in SYSTEM_PROMPT
    assert "CAVEAT" in SYSTEM_PROMPT
    assert "citation_id" in SYSTEM_PROMPT
    assert "JANGAN mengarang" in SYSTEM_PROMPT
    assert "REVIEWER" in SYSTEM_PROMPT
    assert '"Anda"' in SYSTEM_PROMPT  # dilarang, bukan diwajibkan — lihat test_voice di bawah
    assert "SARAN HARUS KONKRET" in SYSTEM_PROMPT
    assert "WAJIB dikutip persis di saran" in SYSTEM_PROMPT  # aturan #11 diperkuat: wajib angka m²


def test_system_prompt_suara_reviewer_bukan_pemohon():
    assert "decision-support" in SYSTEM_PROMPT
    assert "orang ketiga" in SYSTEM_PROMPT
    assert "JANGAN memakai \"Anda\"" in SYSTEM_PROMPT
    assert "warga awam" not in SYSTEM_PROMPT


def test_system_prompt_caveat_fallback_itbx_sadar_status():
    # APP-2026-3335: caveat fallback ITBX beda tergantung status — "diloloskan" hanya utk status I,
    # status lain (mis. X/Tidak Lolos) pakai framing netral "perlu verifikasi manual".
    assert "diloloskan otomatis karena data matriks RDTR kosong, bukan kepatuhan terverifikasi" in SYSTEM_PROMPT
    assert (
        "penentuan status ini didasarkan pada data matriks RDTR yang mungkin belum lengkap — perlu "
        "verifikasi manual apakah kegiatan benar-benar dilarang atau datanya belum tersedia"
    ) in SYSTEM_PROMPT
    assert 'JANGAN PERNAH memakai kata "diloloskan"' in SYSTEM_PROMPT


def test_system_prompt_tidak_lagi_suruh_llm_echo_data_confidence():
    # Fix #4: label kepercayaan data = FAKTA, dirakit deterministik di guardrail.py, BUKAN
    # diserahkan ke LLM utk echo/parafrase (sumber kebocoran token mentah "DATA_CONFIDENCE: X").
    assert "DATA_CONFIDENCE" not in SYSTEM_PROMPT


class TestFaktaItbx:
    def test_status_x_tidak_diberi_label_dilarang(self):
        poin = _poin(status="X", fakta={"lolos": False, "reason": "Tidak ditemukan di matriks RDTR"})
        prompt = build_user_prompt(poin, [])
        assert "STATUS_ITBX: X" in prompt
        assert "Dilarang" not in prompt
        assert "lihat REASON" in prompt

    def test_reason_muncul(self):
        poin = _poin(fakta={"lolos": True, "reason": "Lolos karena kegiatan Diizinkan (I)"})
        prompt = build_user_prompt(poin, [])
        assert "REASON: Lolos karena kegiatan Diizinkan (I)" in prompt

    def test_fallback_flag_muncul_true(self):
        poin = _poin(fakta={"lolos": True, "reason": "x", "fallback_data_kosong": True})
        prompt = build_user_prompt(poin, [])
        assert "FALLBACK_DATA_KOSONG: True" in prompt

    def test_fallback_flag_muncul_false_default(self):
        poin = _poin(fakta={"lolos": True, "reason": "x"})
        prompt = build_user_prompt(poin, [])
        assert "FALLBACK_DATA_KOSONG: False" in prompt

    def test_daftar_kegiatan_diizinkan_muncul(self):
        poin = _poin(fakta={"lolos": True, "reason": "x", "kegiatan_diizinkan": ["Rumah Tunggal", "Warung"]})
        prompt = build_user_prompt(poin, [])
        assert "Rumah Tunggal" in prompt
        assert "Warung" in prompt

    def test_keterangan_ketentuan_muncul_utk_status_bersyarat(self):
        poin = _poin(
            status="B",
            fakta={"lolos": True, "reason": "x", "keterangan_ketentuan": ["Syarat 1", "Syarat 2"]},
        )
        prompt = build_user_prompt(poin, [])
        assert "Syarat 1" in prompt
        assert "Syarat 2" in prompt

    def test_keterangan_ketentuan_tak_muncul_utk_status_i(self):
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x", "keterangan_ketentuan": ["Syarat 1"]})
        prompt = build_user_prompt(poin, [])
        assert "Syarat 1" not in prompt

    def test_keterangan_ketentuan_dipotong_20_item(self):
        # Tanpa kegiatan_diusulkan -> _urutkan_relevansi_ketentuan no-op (return apa adanya),
        # jadi potongan tetap urutan asli — cuma batas jumlahnya yg diuji di sini (15 -> 20,
        # APP-2026-8090, headroom konteks gpt-oss-20b). Urutan relevansi diuji terpisah di bawah.
        ketentuan = [f"Syarat {i}" for i in range(30)]
        poin = _poin(status="T", fakta={"lolos": True, "reason": "x", "keterangan_ketentuan": ketentuan})
        prompt = build_user_prompt(poin, [])
        assert "Syarat 0" in prompt
        assert "Syarat 19" in prompt
        assert "Syarat 20" not in prompt
        assert "dipotong dari 30 item" in prompt

    def test_keterangan_ketentuan_diurutkan_relevansi_bukan_urutan_dokumen(self):
        # APP-2026-8090 (live): potongan 15/20-teratas versi lama SELALU kelompok pertama di
        # dokumen (mis. "pertanian") apapun kegiatan pemohon. Item relevan (kata kunci overlap
        # dgn kegiatan_diusulkan) HARUS naik ke depan meski posisi aslinya jauh di belakang.
        ketentuan = (
            [f"Kelompok pertanian dan kehutanan syarat {i}" for i in range(25)]
            + ["Kelompok konstruksi wajib pembatasan pengambilan air tanah dan pemeliharaan irigasi"]
        )
        poin = _poin(
            status="B",
            fakta={
                "lolos": True,
                "reason": "x",
                "kegiatan_diusulkan": "Konstruksi Bangunan Gedung",
                "keterangan_ketentuan": ketentuan,
            },
        )
        prompt = build_user_prompt(poin, [])
        assert "Kelompok konstruksi wajib pembatasan" in prompt  # item ke-26 (posisi asli) tetap masuk
        assert "diurutkan berdasar relevansi" in prompt

    def test_keterangan_ketentuan_tanpa_kegiatan_diusulkan_urutan_apa_adanya(self):
        ketentuan = ["Syarat A", "Syarat B", "Syarat C"]
        poin = _poin(status="B", fakta={"lolos": True, "reason": "x", "keterangan_ketentuan": ketentuan})
        prompt = build_user_prompt(poin, [])
        # Tak ada kegiatan_diusulkan -> _urutkan_relevansi_ketentuan no-op, urutan asli dipertahankan.
        pos_a = prompt.index("Syarat A")
        pos_b = prompt.index("Syarat B")
        pos_c = prompt.index("Syarat C")
        assert pos_a < pos_b < pos_c


class TestUrutkanRelevansiKetentuan:
    def test_item_relevan_naik_ke_depan(self):
        from app.reasoning.prompts import _urutkan_relevansi_ketentuan

        ketentuan = ["Syarat pertanian dan kehutanan", "Syarat reparasi mobil dan sepeda motor"]
        hasil = _urutkan_relevansi_ketentuan(ketentuan, "Reparasi dan Perawatan Mobil")
        assert hasil[0] == "Syarat reparasi mobil dan sepeda motor"

    def test_tanpa_kegiatan_diusulkan_return_apa_adanya(self):
        from app.reasoning.prompts import _urutkan_relevansi_ketentuan

        ketentuan = ["Syarat A", "Syarat B"]
        assert _urutkan_relevansi_ketentuan(ketentuan, None) == ketentuan

    def test_tak_ada_overlap_tetap_return_semua_urutan_asli(self):
        from app.reasoning.prompts import _urutkan_relevansi_ketentuan

        ketentuan = ["Syarat X sama sekali tak nyambung", "Syarat Y juga tak nyambung"]
        hasil = _urutkan_relevansi_ketentuan(ketentuan, "Reparasi dan Perawatan Mobil")
        assert hasil == ketentuan  # skor 0 semua -> stable sort -> urutan asli dipertahankan

    def test_stopword_tidak_mendominasi_skor(self):
        from app.reasoning.prompts import _urutkan_relevansi_ketentuan

        # "yang", "dengan", "untuk" dll SENGAJA muncul di kedua kalimat (stopword) — tanpa filter
        # stopword, kalimat ke-2 (tak nyambung ke "mobil") bisa menang krn overlap kata umum.
        ketentuan = [
            "Syarat yang diperbolehkan untuk dengan pertanian pada umumnya",
            "Syarat reparasi mobil yang diperbolehkan untuk dengan bengkel pada umumnya",
        ]
        hasil = _urutkan_relevansi_ketentuan(ketentuan, "Reparasi dan Perawatan Mobil")
        assert hasil[0] == "Syarat reparasi mobil yang diperbolehkan untuk dengan bengkel pada umumnya"


class TestFaktaIntensitas:
    def _poin_intensitas(self, target=None) -> PoinKonteks:
        return _poin(
            poin_id="intensitas",
            kategori="Intensitas Bangunan (KDB/KLB/KDH)",
            tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS",
            fakta={
                "parameter": {
                    "kdb": {"usulan": 70, "ambang_maks": 60, "ambang_min": None, "memenuhi": False, "satuan": "persen"},
                },
                "luas_tapak_m2": 400,
                "jumlah_lantai": 2,
                "luas_rth_usulan_m2": 250,
                "target": target or {},
            },
        )

    def test_status_dan_parameter_muncul(self):
        prompt = build_user_prompt(self._poin_intensitas(), [])
        assert "STATUS_INTENSITAS: MELAMPAUI_BATAS" in prompt
        assert "usulan=70" in prompt
        assert "memenuhi=False" in prompt

    def test_target_muncul_saat_ada(self):
        target = {"kdb": {"target_kdb": 60.0, "selisih": 10.0, "footprint_maks_m2": 510.0}}
        prompt = build_user_prompt(self._poin_intensitas(target=target), [])
        assert "TARGET PATUH" in prompt
        assert "60.0" in prompt

    def test_target_tak_muncul_saat_kosong(self):
        prompt = build_user_prompt(self._poin_intensitas(target={}), [])
        assert "TARGET PATUH" not in prompt

    def test_target_kdb_kalimat_berlabel_bukan_dict_mentah(self):
        # Sebelumnya: "[TARGET PATUH: {'target_kdb': 60.0, 'selisih': 10.0, ...}]" (repr dict
        # Python mentah) — sekarang kalimat siap-kutip termasuk angka fisik m².
        target = {"kdb": {"target_kdb": 60.0, "selisih": 10.0, "footprint_maks_m2": 510.0}}
        prompt = build_user_prompt(self._poin_intensitas(target=target), [])
        assert "{'target_kdb'" not in prompt  # bukan lagi repr dict mentah
        assert "KDB harus turun ke maksimal 60.0%" in prompt
        assert "luas lantai dasar bangunan maksimal 510.0 m²" in prompt

    def test_target_kdh_sebut_rth_kurang_m2(self):
        poin = _poin(
            poin_id="intensitas",
            kategori="Intensitas Bangunan (KDB/KLB/KDH)",
            tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS",
            fakta={
                "parameter": {
                    "kdh": {"usulan": 25.0, "ambang_maks": None, "ambang_min": 30.0, "memenuhi": False, "satuan": "persen"},
                },
                "target": {"kdh": {"target_kdh": 30.0, "selisih": 5.0, "rth_dibutuhkan_m2": 255.0, "rth_kurang_m2": 45.0}},
            },
        )
        prompt = build_user_prompt(poin, [])
        assert "KDH harus naik ke minimal 30.0%" in prompt
        assert "RTH dibutuhkan minimal 255.0 m²" in prompt
        assert "masih kurang 45.0 m²" in prompt


# TestFormatTargetParameter dipindah ke tests/test_calculator.py — format_target_parameter() kini
# tinggal di app/reasoning/calculator.py (satu sumber kebenaran teks, dipakai prompts.py & juga
# bangun_langkah_konkret_intensitas/dampak, bukan digandakan di sini).


class TestFaktaDampak:
    def _poin_dampak(self, perlu_mitigasi=False) -> PoinKonteks:
        return _poin(
            poin_id="dampak",
            kategori="Dampak Tata Guna Lahan",
            tipe_rekomendasi="numerik-mitigasi",
            status="Tinggi" if perlu_mitigasi else "Sedang",
            fakta={
                "impact_score": 65,
                "runoff_change_index": 1.78,
                "mitigasi": {
                    "perlu_mitigasi": perlu_mitigasi,
                    "arah": ["turunkan KDB (proporsi tapak/bangunan)", "sumur resapan"] if perlu_mitigasi else [],
                },
            },
        )

    def test_kategori_muncul(self):
        prompt = build_user_prompt(self._poin_dampak(), [])
        assert "KATEGORI_DAMPAK: Sedang" in prompt

    def test_skor_mentah_tidak_lagi_disuntik_ke_prompt(self):
        # Fix #2 (dampak intermiten low_confidence via Cek #1 invers): skor mentah membingungkan
        # LLM ("40 terlihat rendah -> dampak rendah" padahal kategori aktual Tinggi) — dihapus dari
        # prompt, LLM cukup diberi KATEGORI_DAMPAK yang sudah final.
        prompt = build_user_prompt(self._poin_dampak(), [])
        assert "IMPACT_SCORE" not in prompt
        assert "RUNOFF_CHANGE_INDEX" not in prompt
        assert "65" not in prompt
        assert "1.78" not in prompt

    def test_kategori_final_arahan_tegas_muncul(self):
        prompt = build_user_prompt(self._poin_dampak(perlu_mitigasi=True), [])
        assert "Kategori dampak ini FINAL" in prompt
        assert "JANGAN menafsirkan atau menyebut skor angka" in prompt

    def test_arah_mitigasi_muncul_saat_perlu(self):
        prompt = build_user_prompt(self._poin_dampak(perlu_mitigasi=True), [])
        assert "turunkan KDB" in prompt
        assert "sumur resapan" in prompt

    def test_arah_mitigasi_tak_muncul_saat_tak_perlu(self):
        prompt = build_user_prompt(self._poin_dampak(perlu_mitigasi=False), [])
        assert "Arah Mitigasi" not in prompt

    def test_alasan_tidak_dinilai_muncul_dgn_limitations(self):
        # APP-2026-2428: status "Tidak Dinilai" + limitations terisi -> WAJIB disuntikkan sbg
        # ALASAN_TIDAK_DINILAI, instruksikan LLM jangan mengarang alasan/solusi teknis lain.
        poin = _poin(
            poin_id="dampak",
            kategori="Dampak Tata Guna Lahan",
            tipe_rekomendasi="numerik-mitigasi",
            status="Tidak Dinilai",
            fakta={
                "dinilai": False,
                "limitations": "Permohonan bersinggungan dengan lebih dari 1 persil (memerlukan pengecekan manual)",
            },
        )
        prompt = build_user_prompt(poin, [])
        assert "ALASAN_TIDAK_DINILAI" in prompt
        assert "lebih dari 1 persil" in prompt
        assert "JANGAN mengarang alasan lain" in prompt
        assert "JANGAN merekomendasikan solusi teknis drainase" in prompt

    def test_alasan_tidak_dinilai_tak_muncul_tanpa_limitations(self):
        poin = _poin(
            poin_id="dampak",
            kategori="Dampak Tata Guna Lahan",
            tipe_rekomendasi="numerik-mitigasi",
            status="Tidak Dinilai",
            fakta={"dinilai": False},
        )
        prompt = build_user_prompt(poin, [])
        assert "ALASAN_TIDAK_DINILAI" not in prompt

    def test_alasan_tidak_dinilai_tak_muncul_saat_status_bukan_tidak_dinilai(self):
        poin = self._poin_dampak()
        prompt = build_user_prompt(poin, [])
        assert "ALASAN_TIDAK_DINILAI" not in prompt

    def test_peringatan_luas_persil_muncul_saat_true(self):
        poin = _poin(
            poin_id="dampak",
            kategori="Dampak Tata Guna Lahan",
            tipe_rekomendasi="numerik-mitigasi",
            status="Sedang",
            fakta={
                "impact_score": 65,
                "mitigasi": {"perlu_mitigasi": False, "arah": []},
                "luas_usulan_melebihi_persil": True,
            },
        )
        prompt = build_user_prompt(poin, [])
        assert "PERINGATAN_LUAS_PERSIL" in prompt
        assert "MELEBIHI" in prompt
        assert "KDB/KDH" in prompt

    def test_peringatan_luas_persil_tak_muncul_saat_false_atau_none(self):
        poin = self._poin_dampak()  # tak set luas_usulan_melebihi_persil -> None
        prompt = build_user_prompt(poin, [])
        assert "PERINGATAN_LUAS_PERSIL" not in prompt


class TestSitasi:
    def test_chunk_rag_muncul(self):
        chunk = _chunk()
        prompt = build_user_prompt(_poin(), [chunk])
        assert "citation_id=rdtr-matriks-itbx" in prompt
        assert chunk.teks in prompt

    def test_anchor_dasar_hukum_muncul_mendahului_chunk(self):
        anchor = DasarHukum(dokumen="RDTR Sleman", pasal="Matriks ITBX", kutipan="Data KBLI referensi")
        poin = _poin(dasar_hukum=[anchor])
        prompt = build_user_prompt(poin, [_chunk()])
        idx_anchor = prompt.index("citation_id=anchor-0")
        idx_chunk = prompt.index("citation_id=rdtr-matriks-itbx")
        assert idx_anchor < idx_chunk
        assert "Data KBLI referensi" in prompt

    def test_tanpa_sitasi_ada_peringatan(self):
        prompt = build_user_prompt(_poin(), [])
        assert "jangan mengarang sitasi" in prompt


class TestMeta:
    def test_caveat_muncul_data_confidence_tidak_disuntik_ke_prompt(self):
        # Fix #4: data_confidence_keseluruhan TIDAK disuntikkan ke prompt sama sekali — label
        # kepercayaan dirakit deterministik di guardrail.py, bukan diserahkan ke LLM.
        meta = MetaL2(data_confidence_keseluruhan="Medium", caveats=["Data ITBX sebagian estimasi"])
        prompt = build_user_prompt(_poin(), [], meta)
        assert "DATA_CONFIDENCE" not in prompt
        assert "CAVEAT: Data ITBX sebagian estimasi" in prompt

    def test_meta_none_tidak_muncul(self):
        prompt = build_user_prompt(_poin(), [], None)
        assert "DATA_CONFIDENCE" not in prompt
        assert "CAVEAT" not in prompt

    def test_meta_kosong_tidak_muncul(self):
        meta = MetaL2(data_confidence_keseluruhan=None, caveats=None)
        prompt = build_user_prompt(_poin(), [], meta)
        assert "DATA_CONFIDENCE" not in prompt
        assert "CAVEAT" not in prompt


def test_build_user_prompt_catatan_perbaikan_muncul():
    prompt = build_user_prompt(_poin(), [], catatan_perbaikan="Sitasi sebelumnya tidak grounded.")
    assert "Catatan Perbaikan" in prompt
    assert "Sitasi sebelumnya tidak grounded." in prompt


def _poin_output(**overrides) -> PoinOutput:
    defaults = dict(
        poin_id="dampak",
        kategori="Dampak Tata Guna Lahan",
        status="Sedang",
        reasoning_pendek="x",
        reasoning_panjang="x",
        sitasi=[],
        rekomendasi=RekomendasiOutput(tipe="numerik-mitigasi", saran="x"),
    )
    defaults.update(overrides)
    return PoinOutput(**defaults)


class TestBuildKesimpulanPrompt:
    def test_reasoning_pendek_tidak_disertakan(self):
        # Bug ditemukan live (APP-2026-8376): reasoning_pendek poin "aman" (narasi bebas LLM,
        # bisa terdengar kayak masih ada kewajiban) ikut jadi konteks kesimpulan berdampingan dgn
        # saran (templated "tidak perlu tindakan") -> LLM sintesis ikut nada reasoning, bukan saran,
        # hasilnya langkah_berdampak kontradiktif dgn saran poin itu sendiri. reasoning_pendek
        # SENGAJA dihapus dari prompt ini supaya LLM tak punya sumber utk kontradiksi itu.
        poin = _poin_output(
            reasoning_pendek="Pemohon harus memastikan bahwa kegiatan tidak merusak fungsi kawasan resapan air.",
            rekomendasi=RekomendasiOutput(tipe="numerik-mitigasi", saran="Tidak diperlukan tindakan khusus; poin ini telah memenuhi ketentuan."),
        )
        prompt = build_kesimpulan_prompt([poin], "Setuju Bersyarat")
        assert "harus memastikan" not in prompt
        assert "kawasan resapan air" not in prompt
        assert "Tidak diperlukan tindakan khusus" in prompt

    def test_saran_dan_status_tetap_muncul(self):
        poin = _poin_output(poin_id="intensitas", status="MELAMPAUI_BATAS",
                            rekomendasi=RekomendasiOutput(tipe="numerik", saran="Kurangi KDB hingga memenuhi ambang."))
        prompt = build_kesimpulan_prompt([poin], "Setuju Bersyarat")
        assert "intensitas" in prompt
        assert "MELAMPAUI_BATAS" in prompt
        assert "Kurangi KDB hingga memenuhi ambang." in prompt

    def test_system_prompt_kesimpulan_larang_langkah_utk_saran_aman(self):
        assert "tidak diperlukan tindakan khusus" in SYSTEM_PROMPT_KESIMPULAN.lower()
        assert "hanya dari" in SYSTEM_PROMPT_KESIMPULAN.lower()


class TestAngkaPromptDibulatkan:
    """Replay 2026-09-08 (APP-2026-3468): `usulan` KDB dikirim BE sebagai 47.05882352941176 dan
    disuntikkan apa adanya ke prompt, lalu disalin utuh oleh LLM ke narasi petugas — presisi palsu."""

    def test_float_ekor_panjang_dibulatkan_2_desimal(self):
        assert _angka_prompt(47.05882352941176) == "47.06"

    def test_float_pendek_tidak_diubah(self):
        assert _angka_prompt(1.8) == "1.8"
        assert _angka_prompt(60.0) == "60.0"

    def test_nilai_kecil_tidak_dibulatkan_jadi_nol(self):
        # 0.004 -> 0.0 akan MENGHILANGKAN angkanya; lebih baik apa adanya.
        assert _angka_prompt(0.004) == "0.004"

    def test_bukan_float_apa_adanya(self):
        assert _angka_prompt(None) == "None"
        assert _angka_prompt(True) == "True"
        assert _angka_prompt(12) == "12"

    def test_prompt_intensitas_tidak_memuat_float_ekor_panjang(self):
        poin = _poin(
            poin_id="intensitas",
            kategori="Intensitas Bangunan (KDB/KLB/KDH)",
            tipe_rekomendasi="numerik",
            status="MEMENUHI_SYARAT",
            fakta={
                "parameter": {
                    "kdb": {
                        "usulan": 47.05882352941176, "ambang_maks": 60.0, "ambang_min": None,
                        "memenuhi": True, "satuan": "persen",
                    }
                }
            },
        )

        prompt = build_user_prompt(poin, [])

        assert "47.05882352941176" not in prompt
        assert "47.06" in prompt


def test_system_prompt_melarang_bahasa_sistem_di_narasi():
    """Replay 2026-09-08: LLM menulis "KATEGORI_DAMPAK: Rendah menunjukkan..." dan "karena status
    tidak I (tidak terverifikasi) dan fallback data tidak kosong" — bahasa internal, bukan bahasa
    yang dipahami petugas tata ruang."""
    assert "BAHASA UNTUK PETUGAS" in SYSTEM_PROMPT
    assert "KATEGORI_DAMPAK" in SYSTEM_PROMPT


class TestKonteksIndukDiPrompt:
    def test_konteks_induk_muncul_dgn_larangan_menyitasinya(self):
        chunk = _chunk(id="p41-a3", level="ayat", teks="(3) Lokasi sebagaimana dimaksud pada ayat (1).")

        prompt = build_user_prompt(_poin(), [chunk], konteks_induk={"p41-a3": "Pasal 41 teks lengkap"})

        assert "Pasal 41 teks lengkap" in prompt
        assert "JANGAN" in prompt and "disitasi" in prompt
        # citation_id ayat tetap yang ditawarkan sbg sitasi
        assert "citation_id=p41-a3" in prompt

    def test_tanpa_konteks_induk_prompt_tak_berubah(self):
        chunk = _chunk(id="p41-a3", level="ayat", teks="(3) Lokasi.")

        assert build_user_prompt(_poin(), [chunk]) == build_user_prompt(_poin(), [chunk], konteks_induk={})

    def test_konteks_induk_chunk_lain_tidak_nyasar(self):
        chunk = _chunk(id="p41-a3", level="ayat", teks="(3) Lokasi.")

        prompt = build_user_prompt(_poin(), [chunk], konteks_induk={"chunk-lain": "teks nyasar"})

        assert "teks nyasar" not in prompt
