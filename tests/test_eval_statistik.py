"""tests/test_eval_statistik.py — verifikasi eval/statistik.py atas data SINTETIS berjawaban diketahui.

Alat statistik yang salah lebih berbahaya daripada tidak punya alat sama sekali: ia memberi
keyakinan palsu pada angka yang akan dipakai mendukung pengajuan paten. Karena itu bootstrap & uji
berpasangan diverifikasi pada kasus yang jawabannya sudah pasti sebelum disentuhkan ke data nyata.
"""

import pytest

from eval.statistik import bootstrap_ci, wilcoxon_berpasangan


class TestBootstrapCI:
    def test_semua_nilai_sama_ci_menyempit_ke_titik(self):
        hasil = bootstrap_ci([0.8] * 30)

        # approx, bukan sama persis: menjumlah 0.8 tiga puluh kali lalu membaginya menghasilkan
        # 0.8000000000000004 — galat floating-point biasa, bukan ketidakpastian statistik.
        assert hasil.rata == pytest.approx(0.8)
        assert hasil.bawah == pytest.approx(0.8) and hasil.atas == pytest.approx(0.8), \
            "tanpa variasi, tak ada ketidakpastian"

    def test_rata_selalu_di_dalam_selang(self):
        nilai = [0.0, 1.0, 0.5, 0.75, 0.25, 1.0, 0.0, 0.9]

        hasil = bootstrap_ci(nilai)

        assert hasil.bawah <= hasil.rata <= hasil.atas
        assert hasil.n == len(nilai)

    def test_sampel_lebih_besar_menghasilkan_selang_lebih_sempit(self):
        """Inti alasan modul ini ada: menambah TOPIK mengurangi ketidakpastian, bukan mengulang
        query yang sama (retrieval deterministik, pengulangan menghasilkan angka identik)."""
        pola = [0.0, 1.0] * 10

        kecil = bootstrap_ci(pola)
        besar = bootstrap_ci(pola * 10)

        assert (besar.atas - besar.bawah) < (kecil.atas - kecil.bawah)

    def test_hasil_dapat_diproduksi_ulang(self):
        nilai = [0.1, 0.9, 0.4, 0.6, 0.2]

        assert bootstrap_ci(nilai) == bootstrap_ci(nilai), "seed tetap -> laporan reproduktif"

    def test_daftar_kosong_dan_satu_elemen_tidak_meledak(self):
        assert bootstrap_ci([]).n == 0
        satu = bootstrap_ci([0.7])
        assert satu.rata == satu.bawah == satu.atas == 0.7


class TestWilcoxonBerpasangan:
    def test_dua_deret_identik_dilaporkan_identik_bukan_p_palsu(self):
        a = [0.5, 0.7, 0.9, 0.2]

        hasil = wilcoxon_berpasangan(a, list(a))

        assert hasil.n_beda == 0
        assert hasil.p == 1.0 and not hasil.signifikan
        assert "identik" in hasil.ringkas()

    def test_selisih_konsisten_besar_terdeteksi_signifikan(self):
        b = [0.10, 0.12, 0.15, 0.11, 0.09, 0.13, 0.14, 0.08, 0.12, 0.10, 0.11, 0.13]
        a = [x + 0.40 for x in b]

        hasil = wilcoxon_berpasangan(a, b)

        assert hasil.signifikan, "beda konsisten di semua topik harus signifikan"
        assert hasil.selisih_rata > 0
        assert hasil.efek > 0.9, "semua topik searah -> efek mendekati +1"

    def test_selisih_satu_topik_dari_banyak_tidak_signifikan(self):
        """Persis pola yang sempat saya salah baca sebagai keunggulan: dense+rerank vs rrf+rerank
        di k=3 hanya berbeda pada SATU topik dari 22."""
        b = [1.0] * 21 + [0.0]
        a = [1.0] * 22

        hasil = wilcoxon_berpasangan(a, b)

        assert not hasil.signifikan, "beda satu topik tidak boleh lolos sbg keunggulan"

    def test_arah_selisih_terbaca_benar(self):
        a = [0.2, 0.3, 0.1, 0.25, 0.15, 0.2, 0.3, 0.1]
        b = [x + 0.5 for x in a]

        hasil = wilcoxon_berpasangan(a, b)

        assert hasil.selisih_rata < 0 and hasil.efek < 0
        assert "kalah" in hasil.ringkas()

    def test_topik_seri_dibuang_sesuai_konvensi(self):
        # 8 topik seri, 4 topik berbeda -> hanya 4 yang masuk uji.
        a = [0.5] * 8 + [0.9, 0.8, 0.85, 0.95]
        b = [0.5] * 8 + [0.1, 0.2, 0.15, 0.05]

        hasil = wilcoxon_berpasangan(a, b)

        assert hasil.n_beda == 4

    def test_panjang_tak_sama_ditolak(self):
        try:
            wilcoxon_berpasangan([0.1, 0.2], [0.1])
        except ValueError as exc:
            assert "panjang" in str(exc)
        else:
            raise AssertionError("harus menolak pasangan yang panjangnya beda")


class TestStatistikAtasMetrikNyata:
    """Jembatan yang sebelumnya tidak diuji.

    `TestBootstrapCI`/`TestWilcoxonBerpasangan` di atas menguji eval/statistik.py atas list angka
    sintetis — dan tetap hijau meski pemanggilnya salah mengambil kunci. Akibatnya nyata: evaluasi
    192 topik selesai memanggil seluruh API lalu mati dengan `KeyError: 'mrr'`, karena per-topik yang
    ada hanyalah `rr`/`ap`. Tes ini memakai dict yang benar-benar dihasilkan `_metrik_satu_query`.
    """

    def test_statistik_jalan_atas_keluaran_metrik_satu_query(self):
        from eval.eval_rag import _METRIK_KLAIM, _metrik_satu_query, _statistik

        relevan = {"chunk-a"}
        bagus = [_metrik_satu_query(["chunk-a", "chunk-x", "chunk-y"], relevan) for _ in range(6)]
        jelek = [_metrik_satu_query(["chunk-x", "chunk-y", "chunk-a"], relevan) for _ in range(6)]

        hasil = _statistik({"bagus": bagus, "jelek": jelek}, ["bagus", "jelek"], "bagus")

        for metrik in _METRIK_KLAIM:
            assert metrik in hasil["ci"]["bagus"], f"{metrik} harus punya CI"
            assert hasil["ci"]["bagus"][metrik]["n"] == 6
        assert hasil["ci"]["bagus"]["mrr"]["rata"] == 1.0, "relevan di peringkat 1 -> RR=1"
        assert hasil["ci"]["jelek"]["mrr"]["rata"] == pytest.approx(1 / 3)
        assert hasil["uji_vs_produksi"]["jelek"]["mrr"]["selisih_rata"] < 0

    def test_skor_klaim_memakai_kunci_per_query_yang_benar(self):
        from eval.eval_rag import _METRIK_KLAIM, _metrik_satu_query, _nilai_klaim

        q = _metrik_satu_query(["chunk-x", "chunk-a"], {"chunk-a"})

        assert _nilai_klaim(q, "mrr") == q["rr"] == 0.5
        assert _nilai_klaim(q, "map") == q["ap"]
        for metrik in _METRIK_KLAIM:
            assert isinstance(_nilai_klaim(q, metrik), float)
