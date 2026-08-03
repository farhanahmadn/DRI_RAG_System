from app.reasoning.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.base import Chunk
from app.schemas import DasarHukum, MetaL2, PoinKonteks


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


def test_system_prompt_suara_reviewer_bukan_pemohon():
    assert "decision-support" in SYSTEM_PROMPT
    assert "orang ketiga" in SYSTEM_PROMPT
    assert "JANGAN memakai \"Anda\"" in SYSTEM_PROMPT
    assert "warga awam" not in SYSTEM_PROMPT


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

    def test_keterangan_ketentuan_dipotong_15_item(self):
        ketentuan = [f"Syarat {i}" for i in range(30)]
        poin = _poin(status="T", fakta={"lolos": True, "reason": "x", "keterangan_ketentuan": ketentuan})
        prompt = build_user_prompt(poin, [])
        assert "Syarat 0" in prompt
        assert "Syarat 14" in prompt
        assert "Syarat 15" not in prompt
        assert "dipotong dari 30 item" in prompt


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
