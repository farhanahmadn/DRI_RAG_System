"""tests/test_metrik_atribusi.py — verifikasi eval/metrik_atribusi.py atas log SINTETIS berjawaban
diketahui.

Modul yang diuji menghasilkan angka yang akan dikutip sebagai bukti mutu sitasi. Alat ukur yang
salah di situ lebih berbahaya daripada tidak mengukur: ia memberi keyakinan palsu. Jadi tiap metrik
diperiksa pada kasus yang jawabannya sudah pasti — termasuk kasus yang penyebutnya nol, yang WAJIB
dilaporkan "tak terukur" dan bukan 0% (0% terbaca sebagai gagal total, artinya kebalikannya).
"""

import json

from eval.metrik_atribusi import _jenis_id, _normalisasi, hitung, muat_permohonan

_DOK_NYATA = "Peraturan Bupati Sleman Nomor 80 Tahun 2023 tentang RDTR Kawasan Sleman Tengah"
_DOK_MOCK = "Peraturan Daerah Kabupaten Sleman tentang Rencana Detail Tata Ruang (RDTR)"


def _sitasi(cid, kutipan="Ketentuan intensitas pemanfaatan ruang zona perumahan kepadatan tinggi",
            dokumen=_DOK_NYATA, terverifikasi=True):
    return {"citation_id": cid, "dokumen": dokumen, "pasal": "x", "halaman": 1,
            "kutipan": kutipan, "terverifikasi": terverifikasi}


def _baris(sitasi_per_poin, zona="Zona Perumahan", subzona=None, timestamp="2026-09-20T10:00:00",
           dasar_hukum_per_poin=None, status_per_poin=None):
    tahapan = {}
    for pid in ("itbx", "intensitas", "dampak"):
        tahapan[pid] = {"dasar_hukum": (dasar_hukum_per_poin or {}).get(pid, [])}
    return json.dumps({
        "timestamp": timestamp,
        "request": {
            "application_number": "APP-TEST",
            "lokasi": {"rdtr_zone": zona, "rdtr_subzone": subzona},
            "gate_hukum": {"tahapan": tahapan},
        },
        "response": {"poin": [
            {"poin_id": pid, "kategori": pid, "status": (status_per_poin or {}).get(pid, "MEMENUHI_SYARAT"),
             "reasoning_pendek": "x", "reasoning_panjang": "y", "sitasi": sit,
             "rekomendasi": {}, "low_confidence": False}
            for pid, sit in sitasi_per_poin.items()
        ]},
    }, ensure_ascii=False)


def _tulis(tmp_path, baris):
    p = tmp_path / "precheck.jsonl"
    p.write_text("\n".join(baris) + "\n", encoding="utf-8")
    return p


class TestBantu:
    def test_normalisasi_membuang_tanda_baca_dan_kapital(self):
        assert _normalisasi("A. Pengembangan BARU;  dengan   syarat!") == \
            "a pengembangan baru dengan syarat"

    def test_jenis_id_membedakan_korpus_anchor_dan_lain(self):
        assert _jenis_id("rdtr-sleman-tengah-vi-r-2") == "korpus"
        assert _jenis_id("anchor-0") == "anchor"
        assert _jenis_id("rdtr-p1-a107") == "lain", "pola MockRetriever bukan korpus"
        assert _jenis_id(None) == "kosong"


class TestPresisiSitasi:
    def test_id_korpus_tak_ada_di_db_menurunkan_presisi(self, tmp_path):
        log = _tulis(tmp_path, [_baris({"intensitas": [
            _sitasi("rdtr-sleman-tengah-vi-r-2"),
            _sitasi("rdtr-sleman-tengah-vi-karangan"),   # tidak ada di korpus
        ]})])
        pm, _ = muat_permohonan(log)
        chunks = {"rdtr-sleman-tengah-vi-r-2": {"zona": "R-2", "level": "tabel", "teks": "x"}}

        h = hitung(pm, chunks)

        assert h["presisi_korpus"] == {"n": 2, "kena": 1, "nilai": 0.5}

    def test_anchor_di_luar_jangkauan_dasar_hukum_terdeteksi(self, tmp_path):
        log = _tulis(tmp_path, [_baris(
            {"itbx": [_sitasi("anchor-0"), _sitasi("anchor-5")]},
            dasar_hukum_per_poin={"itbx": [{"dokumen": "RDTR Sleman", "pasal": "Matriks ITBX"}]},
        )])
        pm, _ = muat_permohonan(log)

        h = hitung(pm, {})

        assert h["presisi_anchor"] == {"n": 2, "kena": 1, "nilai": 0.5}, \
            "anchor-5 tak punya padanan di dasar_hukum yang cuma berisi 1 item"


class TestGroundednessKutipan:
    def test_kutipan_yang_memang_ada_di_chunk_dihitung_grounded(self, tmp_path):
        log = _tulis(tmp_path, [_baris({"intensitas": [
            _sitasi("rdtr-sleman-tengah-vi-r-2", kutipan="KDB maksimum 60-70%, KLB maksimum 3-9.8")
        ]})])
        pm, _ = muat_permohonan(log)
        chunks = {"rdtr-sleman-tengah-vi-r-2": {
            "zona": "R-2", "level": "tabel",
            "teks": "Lampiran VI Zona R-2: KDB maksimum 60–70%; KLB maksimum 3–9.8; KDH minimum 20%."}}

        h = hitung(pm, chunks)

        assert h["groundedness_kutipan"]["nilai"] == 1.0, \
            "beda tanda baca/dash tidak boleh dihitung sbg kutipan karangan"

    def test_kutipan_karangan_terdeteksi(self, tmp_path):
        log = _tulis(tmp_path, [_baris({"intensitas": [
            _sitasi("rdtr-sleman-tengah-vi-r-2", kutipan="KDB maksimum sembilan puluh persen penuh")
        ]})])
        pm, _ = muat_permohonan(log)
        chunks = {"rdtr-sleman-tengah-vi-r-2": {
            "zona": "R-2", "level": "tabel", "teks": "KDB maksimum 60-70%."}}

        h = hitung(pm, chunks)

        assert h["groundedness_kutipan"] == {
            "n": 1, "kena": 0, "nilai": 0.0,
            "kutipan_judul_dokumen": 0, "kutipan_terlalu_pendek_dilewati": 0}

    def test_kutipan_berisi_judul_dokumen_dihitung_terpisah(self, tmp_path):
        """Mode gagal nyata di log: `terverifikasi` true karena id-nya sah, tapi yang dikutip adalah
        judul peraturannya sendiri — tidak menjelaskan apa pun."""
        log = _tulis(tmp_path, [_baris({"intensitas": [_sitasi(
            "rdtr-sleman-tengah-vi-r-2",
            kutipan="Peraturan Bupati Sleman Nomor 80 Tahun 2023 tentang RDTR Kawasan Sleman Tengah",
        )]})])
        pm, _ = muat_permohonan(log)
        chunks = {"rdtr-sleman-tengah-vi-r-2": {
            "zona": "R-2", "level": "tabel", "teks": "KDB maksimum 60-70%."}}

        h = hitung(pm, chunks)

        assert h["groundedness_kutipan"]["kutipan_judul_dokumen"] == 1
        assert h["groundedness_kutipan"]["nilai"] == 0.0

    def test_kutipan_terlalu_pendek_dilewati_bukan_dihitung_gagal(self, tmp_path):
        log = _tulis(tmp_path, [_baris({"intensitas": [
            _sitasi("rdtr-sleman-tengah-vi-r-2", kutipan="KDB 60%")
        ]})])
        pm, _ = muat_permohonan(log)
        chunks = {"rdtr-sleman-tengah-vi-r-2": {
            "zona": "R-2", "level": "tabel", "teks": "KDB maksimum 60-70%."}}

        h = hitung(pm, chunks)

        g = h["groundedness_kutipan"]
        assert g["kutipan_terlalu_pendek_dilewati"] == 1
        assert g["n"] == 0 and g["nilai"] is None, "penyebut nol -> tak terukur, BUKAN 0%"


class TestRecallZona:
    CHUNKS = {
        "rdtr-sleman-tengah-vi-r-2": {"zona": "R-2", "level": "tabel", "teks": "t"},
        "rdtr-sleman-tengah-vi-kt": {"zona": "KT", "level": "tabel", "teks": "t"},
        "rdtr-sleman-tengah-p44-a1": {"zona": None, "level": "ayat", "teks": "t"},
    }

    def test_sitasi_keluarga_zona_pemohon_dihitung_kena(self, tmp_path):
        log = _tulis(tmp_path, [_baris({"intensitas": [_sitasi("rdtr-sleman-tengah-vi-r-2")]},
                                       zona="Zona Perumahan")])
        pm, _ = muat_permohonan(log)

        h = hitung(pm, self.CHUNKS)

        assert h["recall_zona"]["per_poin"]["intensitas"] == {"n": 1, "kena": 1, "nilai": 1.0}

    def test_sitasi_keluarga_zona_lain_dihitung_tidak_kena(self, tmp_path):
        """Persis kelas bug APP-2026-6191: pemohon Zona Perumahan disodori Lampiran VI zona KT."""
        log = _tulis(tmp_path, [_baris({"intensitas": [_sitasi("rdtr-sleman-tengah-vi-kt")]},
                                       zona="Zona Perumahan")])
        pm, _ = muat_permohonan(log)

        h = hitung(pm, self.CHUNKS)

        assert h["recall_zona"]["per_poin"]["intensitas"] == {"n": 1, "kena": 0, "nilai": 0.0}

    def test_hanya_menyitasi_pasal_umum_dihitung_tidak_kena(self, tmp_path):
        log = _tulis(tmp_path, [_baris({"intensitas": [_sitasi("rdtr-sleman-tengah-p44-a1")]},
                                       zona="Zona Perumahan")])
        pm, _ = muat_permohonan(log)

        h = hitung(pm, self.CHUNKS)

        assert h["recall_zona"]["per_poin"]["intensitas"]["kena"] == 0, \
            "ambang intensitas bersumber dari tabel per-zona; pasal definisional bukan penggantinya"

    def test_dampak_tidak_dituntut_sitasi_berzona(self, tmp_path):
        """Ketentuan dampak tersaji sbg pasal prosa lintas-zona, jadi menuntutnya di sana akan
        menandai jawaban yang benar sebagai salah."""
        log = _tulis(tmp_path, [_baris({"dampak": [_sitasi("rdtr-sleman-tengah-p44-a1")]})])
        pm, _ = muat_permohonan(log)

        h = hitung(pm, self.CHUNKS)

        assert "dampak" not in h["recall_zona"]["per_poin"]

    def test_poin_tidak_dinilai_tak_ikut_terhitung(self, tmp_path):
        log = _tulis(tmp_path, [_baris({"intensitas": [_sitasi("rdtr-sleman-tengah-vi-kt")]},
                                       status_per_poin={"intensitas": "Tidak Dinilai"})])
        pm, _ = muat_permohonan(log)

        h = hitung(pm, self.CHUNKS)

        assert h["recall_zona"]["per_poin"]["intensitas"]["n"] == 0
        assert h["recall_zona"]["per_poin"]["intensitas"]["nilai"] is None


class TestPemilahanSumberDanWaktu:
    def test_permohonan_dari_mock_dibuang(self, tmp_path):
        """Tanpa ini, yang terukur adalah MockRetriever — 335 dari 444 permohonan di log nyata."""
        log = _tulis(tmp_path, [
            _baris({"intensitas": [_sitasi("rdtr-p1-a107", dokumen=_DOK_MOCK)]}),
            _baris({"intensitas": [_sitasi("rdtr-sleman-tengah-vi-r-2")]}),
        ])
        pm, _ = muat_permohonan(log)

        h = hitung(pm, {"rdtr-sleman-tengah-vi-r-2": {"zona": "R-2", "level": "tabel", "teks": "t"}})

        assert h["cakupan"]["permohonan_dari_mock"] == 1
        assert h["cakupan"]["permohonan_retriever_nyata"] == 1
        assert h["presisi_korpus"]["n"] == 1, "sitasi mock tidak boleh ikut penyebut"

    def test_sejak_membuang_permohonan_versi_kode_lama(self, tmp_path):
        log = _tulis(tmp_path, [
            _baris({"intensitas": [_sitasi("rdtr-sleman-tengah-vi-kt")]}, timestamp="2026-08-07T09:00:00"),
            _baris({"intensitas": [_sitasi("rdtr-sleman-tengah-vi-r-2")]}, timestamp="2026-09-20T09:00:00"),
        ])

        semua, _ = muat_permohonan(log)
        baru, lewat = muat_permohonan(log, sejak="2026-09-07")

        assert len(semua) == 2
        assert len(baru) == 1, "permohonan 7 Agustus dijalankan sebelum filter zona_prefix ada"
        assert lewat["sebelum 2026-09-07"] == 1

    def test_rentang_tanggal_ikut_dilaporkan(self, tmp_path):
        log = _tulis(tmp_path, [
            _baris({"intensitas": [_sitasi("rdtr-sleman-tengah-vi-r-2")]}, timestamp="2026-09-10T08:00:00"),
            _baris({"intensitas": [_sitasi("rdtr-sleman-tengah-vi-r-2")]}, timestamp="2026-09-25T08:00:00"),
        ])
        pm, _ = muat_permohonan(log)

        h = hitung(pm, {"rdtr-sleman-tengah-vi-r-2": {"zona": "R-2", "level": "tabel", "teks": "t"}})

        assert h["cakupan"]["rentang_tanggal_dinilai"] == \
            ["2026-09-10T08:00:00", "2026-09-25T08:00:00"]


class TestTanpaDB:
    def test_metrik_yang_butuh_teks_dilaporkan_tak_terukur(self, tmp_path):
        """DB mati bukan alasan menampilkan angka. Yang butuh teks chunk harus 'tak terukur'."""
        log = _tulis(tmp_path, [_baris({"intensitas": [_sitasi("rdtr-sleman-tengah-vi-r-2")]})])
        pm, _ = muat_permohonan(log)

        h = hitung(pm, {})      # korpus tak tersedia

        assert h["presisi_korpus"]["nilai"] is None
        assert h["groundedness_kutipan"]["nilai"] is None
        assert h["kesepakatan_flag_terverifikasi"]["nilai"] is None
