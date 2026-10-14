"""tests/test_replay_prompt_zona.py — verifikasi logika uji berpasangan jangkar zona.

Modul yang diuji menghasilkan angka "sebelum vs sesudah" yang akan dipakai membenarkan perubahan
prompt produksi. Dua kesalahan yang paling mudah terjadi di alat ukur seperti ini, dan keduanya
membuat perubahan tampak berhasil padahal belum terbukti:

  1. mencampur lengan — permohonan yang hanya punya satu lengan ikut masuk ke angka berpasangan,
  2. ambang yang terlalu longgar — satu jalan yang kebetulan kena dihitung sebagai "kena", padahal
     generasi tak deterministik.

Tiap kelas di bawah mengunci salah satunya. Tak ada panggilan Groq/DB di berkas ini — hanya fungsi
murni (`_ukur`, `ringkas`, `_muat_fixture`).
"""

import json

from app.schemas import PoinKonteks, PoinOutput, RekomendasiOutput, SitasiOutput
from eval.replay_prompt_zona import _muat_fixture, _RE_KODE_SUBZONA, _ukur, ringkas


def _poin(**ubah) -> PoinKonteks:
    dasar = dict(
        poin_id="intensitas",
        kategori="Intensitas Bangunan (KDB/KLB/KDH)",
        tipe_rekomendasi="numerik",
        status="MELAMPAUI_BATAS",
        fakta={"parameter": {"kdb": {"usulan": 58.0, "ambang_maks": 10.0, "ambang_min": None,
                                     "memenuhi": False, "satuan": "persen"}}, "target": {}},
        dasar_hukum=[],
        zona="Zona Pertanian",
    )
    dasar.update(ubah)
    return PoinKonteks(**dasar)


def _keluaran(panjang: str, saran: str = "x") -> PoinOutput:
    return PoinOutput(
        poin_id="intensitas", kategori="x", status="MELAMPAUI_BATAS",
        reasoning_pendek=panjang[:80] or "x", reasoning_panjang=panjang,
        sitasi=[SitasiOutput(citation_id="anchor-0", dokumen="RDTR", pasal="44", halaman=1,
                             kutipan="x", terverifikasi=True)],
        rekomendasi=RekomendasiOutput(tipe="numerik", target=None, saran=saran,
                                      disclaimer=None, langkah_konkret=[]),
        low_confidence=False,
    )


def _baris(permohonan, jalan, jalur="llm", kena=True, dikarang=None, zona="Zona Pertanian",
           subzona=None):
    return {"fixture": f"{permohonan}.json", "jalan": jalan, "permohonan": permohonan,
            "zona": zona, "zona_subzone": subzona, "status": "MELAMPAUI_BATAS",
            "jalur": jalur, "kena": kena if jalur == "llm" else None,
            "token_cocok": "pertanian" if kena else None,
            "jangkar_tersedia": ["pertanian"], "kode_subzona_dikarang": dikarang or [],
            "low_confidence": False, "n_sitasi": 1}


class TestUkur:
    def test_zona_induk_disebut_terbaca_kena(self):
        h = _ukur(_poin(), _keluaran("KDB melampaui ambang Lampiran VI zona Pertanian"),
                  "Zona Pertanian", None)
        assert h["kena"] is True
        assert h["token_cocok"] == "pertanian"

    def test_zona_tak_disebut_terbaca_miss(self):
        h = _ukur(_poin(), _keluaran("KDB melampaui ambang sesuai Lampiran VI"),
                  "Zona Pertanian", None)
        assert h["kena"] is False
        assert h["token_cocok"] is None

    def test_kode_subzona_diterima_sebagai_jangkar(self):
        h = _ukur(_poin(zona_subzone="P-1"), _keluaran("KDB melampaui ambang sub-zona P-1"),
                  "Zona Pertanian", "P-1")
        assert h["kena"] is True

    def test_jalur_template_tidak_dinilai_kena(self):
        """Narasi template tak pernah menyebut partikular apa pun — menilainya "miss" akan
        menghukum perubahan prompt atas kegagalan yang bukan soal prompt."""
        from app.reasoning.templates import template_low_confidence

        poin = _poin()
        h = _ukur(poin, template_low_confidence(poin), poin.zona, None)
        assert h["jalur"] == "template_low_confidence"
        assert h["kena"] is None

    def test_kode_dikarang_hanya_ditandai_saat_be_tak_mengirim_subzona(self):
        narasi = "KDB melampaui ambang sub-zona P-1 di zona Pertanian"
        tanpa = _ukur(_poin(), _keluaran(narasi), "Zona Pertanian", None)
        assert tanpa["kode_subzona_dikarang"] == ["P-1"], \
            "BE tak mengirim sub-zona, jadi kode apa pun di narasi adalah karangan"
        dengan = _ukur(_poin(zona_subzone="P-1"), _keluaran(narasi), "Zona Pertanian", "P-1")
        assert dengan["kode_subzona_dikarang"] == [], \
            "kalau sub-zonanya memang dikirim BE, menyebutnya sah"

    def test_regex_kode_subzona_tidak_menyapu_angka_biasa(self):
        assert _RE_KODE_SUBZONA.findall("KDB 58-60 persen dan KLB 1-2") == []
        assert _RE_KODE_SUBZONA.findall("sub-zona R-2 dan SPU-3") == ["R-2", "SPU-3"]


class TestRingkasBerpasangan:
    def test_hanya_permohonan_dua_lengan_masuk_berpasangan(self):
        """Permohonan tanpa lengan "sebelum" berjalur LLM TIDAK boleh ikut — kalau ikut, angkanya
        campuran dan tak bisa menyimpulkan apa pun tentang perubahan prompt."""
        hasil = [_baris("APP-A", 1), _baris("APP-B", 1)]
        sebelum = {"APP-A": {"jalur": "llm", "kena": False, "token_cocok": None,
                             "timestamp": "2026-08-01T00:00:00"},
                   "APP-B": {"jalur": "template_low_confidence", "kena": None,
                             "token_cocok": None, "timestamp": "2026-08-01T00:00:00"}}
        h = ringkas(hasil, sebelum, 1)
        assert h["berpasangan"]["n"] == 1
        assert [b["permohonan"] for b in h["pasangan"]] == ["APP-A"]
        assert [b["permohonan"] for b in h["tanpa_pembanding"]] == ["APP-B"]

    def test_permohonan_tanpa_jejak_di_log_masuk_tanpa_pembanding(self):
        h = ringkas([_baris("APP-BARU", 1)], {}, 1)
        assert h["berpasangan"]["n"] == 0
        assert h["berpasangan"]["nilai_baru"] is None, "penyebut nol -> tak terukur, bukan 0%"
        assert len(h["tanpa_pembanding"]) == 1

    def test_kena_baru_menuntut_SELURUH_jalan_menyebut_zona(self):
        """Ambang paling ketat, sengaja: generasi tak deterministik, jadi satu jalan yang kebetulan
        kena bukan bukti bahwa promptnya bekerja."""
        sebelum = {"APP-A": {"jalur": "llm", "kena": False, "token_cocok": None,
                             "timestamp": "2026-08-01T00:00:00"}}
        separuh = ringkas([_baris("APP-A", 1, kena=True), _baris("APP-A", 2, kena=False)],
                          sebelum, 2)
        assert separuh["berpasangan"]["kena_prompt_baru"] == 0
        assert separuh["berpasangan"]["membaik"] == []
        penuh = ringkas([_baris("APP-A", 1, kena=True), _baris("APP-A", 2, kena=True)],
                        sebelum, 2)
        assert penuh["berpasangan"]["kena_prompt_baru"] == 1
        assert penuh["berpasangan"]["membaik"] == ["APP-A"]

    def test_regresi_tercatat_sebagai_memburuk(self):
        """Yang tadinya kena lalu seluruh jalan baru miss harus MUNCUL, bukan tenggelam di rata-rata."""
        sebelum = {"APP-A": {"jalur": "llm", "kena": True, "token_cocok": "pertanian",
                             "timestamp": "2026-08-01T00:00:00"}}
        h = ringkas([_baris("APP-A", 1, kena=False), _baris("APP-A", 2, kena=False)], sebelum, 2)
        assert h["berpasangan"]["memburuk"] == ["APP-A"]
        assert h["berpasangan"]["kena_prompt_lama"] == 1
        assert h["berpasangan"]["kena_prompt_baru"] == 0

    def test_jalan_berjalur_template_tak_masuk_penyebut_jalan_llm(self):
        sebelum = {"APP-A": {"jalur": "llm", "kena": False, "token_cocok": None,
                             "timestamp": "2026-08-01T00:00:00"}}
        h = ringkas([_baris("APP-A", 1, kena=True),
                     _baris("APP-A", 2, jalur="template_low_confidence")], sebelum, 2)
        s = h["pasangan"][0]["sesudah"]
        assert s["jalan_llm"] == 1 and s["jalan_kena"] == 1
        assert s["jalur"]["template_low_confidence"] == 1

    def test_kode_dikarang_diangkat_ke_tingkat_atas(self):
        """Kalau kewajiban menyebut zona berubah jadi tekanan mengarang kode sub-zona, itu harus
        terlihat di ringkasan — bukan terkubur di rincian per permohonan."""
        h = ringkas([_baris("APP-A", 1, dikarang=["R-2"])], {}, 1)
        assert h["kode_subzona_dikarang"] == ["R-2"]


class TestMuatFixture:
    def test_hanya_payload_l2_yang_diambil(self, tmp_path, monkeypatch):
        import eval.replay_prompt_zona as mod

        (tmp_path / "ada_amplop.json").write_text(
            json.dumps({"statusCode": 200, "data": {"gate_hukum": {"tahapan": {}}}}),
            encoding="utf-8")
        (tmp_path / "tanpa_amplop.json").write_text(
            json.dumps({"gate_hukum": {"tahapan": {}}}), encoding="utf-8")
        (tmp_path / "bukan_l2.json").write_text(json.dumps({"halo": 1}), encoding="utf-8")
        (tmp_path / "rusak.json").write_text("{bukan json", encoding="utf-8")
        monkeypatch.setattr(mod, "_FIXTURE", tmp_path)

        nama = [n for n, _ in mod._muat_fixture()]
        assert nama == ["ada_amplop.json", "tanpa_amplop.json"], \
            "payload di bawah 'data' maupun di akar sama-sama diterima; non-L2 & rusak dilewati"
