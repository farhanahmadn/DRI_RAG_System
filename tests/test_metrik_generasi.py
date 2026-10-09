"""tests/test_metrik_generasi.py — verifikasi eval/metrik_generasi.py atas kasus berjawaban diketahui.

Modul yang diuji menghasilkan angka yang akan dikutip sebagai bukti bahwa LLM patuh pada konteks dan
relevan dengan permohonan. Alat ukur yang salah di situ lebih berbahaya daripada tidak mengukur, dan
itu BUKAN risiko teoretis di berkas ini: selama penyetelannya, tiga dari empat metrik sempat
mengeluarkan angka yang salah —

  1. narasi TEMPLATE deterministik ikut dinilai sebagai keluaran LLM (satu teks template muncul di
     11 permohonan berbeda, dan `kekhususan` poin dampak keluar 0.0%),
  2. jendela ±90 karakter di sekitar nama parameter menyerap kata kerja milik parameter TETANGGA
     pada kalimat daftar, sehingga 55% kasus ditandai "arah salah" secara palsu,
  3. kasus AMBIGU masuk penyebut `ketepatan_arah`, sehingga "tak bisa dinilai" terhitung "salah"
     (76.2% padahal seluruh selisihnya ambigu), dan jendela negasi karakter-tetap menandai dua
     kalimat yang benar sebagai menyimpang.

Tiap kelas di bawah mengunci salah satu koreksi itu dengan kalimat yang benar-benar memicunya.
"""

import json
from pathlib import Path

from app.schemas import (
    LangkahKonkretOutput,
    PoinKonteks,
    PoinOutput,
    RekomendasiOutput,
    SitasiOutput,
)
from eval.metrik_generasi import (
    Kasus,
    _arah_intensitas,
    _arah_kategori_itbx,
    _arah_mitigasi_dampak,
    _jalur_narasi,
    _jendela_parameter,
    _klaim_arah,
    _normalisasi,
    _posisi_parameter,
    _proporsi,
    _label_masalah,
    _teks_klausa,
    _ternegasi,
    boilerplate,
    faithfulness_numerik,
    kekhususan,
    ketepatan_arah,
    laju_tolak_guardrail,
    muat_kasus,
    sebaran_jalur,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "l2_sample_lolos.json"


def _parameter(kdb=True, klb=True, kdh=True):
    return {
        "kdb": {"usulan": 47.0, "ambang_maks": 60.0, "ambang_min": None, "memenuhi": kdb,
                "satuan": "persen"},
        "klb": {"usulan": 0.94, "ambang_maks": 1.8, "ambang_min": None, "memenuhi": klb,
                "satuan": "rasio"},
        "kdh": {"usulan": 29.4, "ambang_maks": None, "ambang_min": 20.0, "memenuhi": kdh,
                "satuan": "persen"},
    }


def _konteks(**ubah) -> PoinKonteks:
    dasar = dict(
        poin_id="intensitas",
        kategori="Intensitas Bangunan (KDB/KLB/KDH)",
        tipe_rekomendasi="numerik",
        status="MELAMPAUI_BATAS",
        fakta={"parameter": _parameter(), "target": {}},
        dasar_hukum=[],
        zona="Zona Perumahan",
    )
    dasar.update(ubah)
    return PoinKonteks(**dasar)


def _keluaran(panjang: str, pendek: str = "", saran: str = "", sitasi=None,
              tipe="numerik") -> PoinOutput:
    return PoinOutput(
        poin_id="intensitas", kategori="x", status="MELAMPAUI_BATAS",
        reasoning_pendek=pendek or panjang[:80] or "x",
        reasoning_panjang=panjang,
        sitasi=sitasi or [],
        rekomendasi=RekomendasiOutput(tipe=tipe, target=None, saran=saran or "x",
                                      disclaimer=None, langkah_konkret=[]),
        low_confidence=False,
    )


def _kasus(panjang: str, konteks: PoinKonteks | None = None, poin_id: str = "intensitas",
           permohonan: str = "APP-1", timestamp: str = "2026-09-20T10:00:00",
           saran: str = "", pendek: str = "", sitasi=None, zona="Zona Perumahan",
           subzona=None, diagnosa=None) -> Kasus:
    k = konteks or _konteks()
    return Kasus(
        permohonan=permohonan, timestamp=timestamp, poin_id=poin_id, konteks=k,
        keluaran=_keluaran(panjang, pendek=pendek, saran=saran, sitasi=sitasi),
        zona=zona, zona_subzone=subzona, jalur="llm", diagnosa=diagnosa,
    )


def _riwayat(*sebab_dan_masalah) -> dict:
    """Diagnosa stub: tiap argumen satu percobaan, berupa (sebab, [masalah])."""
    return {"riwayat_percobaan": [
        {"percobaan": i, "sebab": s, "masalah": list(m),
         "exception": "RuntimeError: x" if s == "panggilan_llm_gagal" else None}
        for i, (s, m) in enumerate(sebab_dan_masalah, 1)
    ]}


class TestBantu:
    def test_normalisasi_membuang_tanda_baca(self):
        assert _normalisasi("KDB (58,6%) vs 10%!") == "kdb 58 6 vs 10"

    def test_teks_klausa_mempertahankan_tanda_baca(self):
        """`ketepatan_arah` bersandar pada tanda baca sebagai batas klausa — kalau ikut dibuang,
        lingkup negasinya melebar dan kalimat yang benar ditandai menyimpang."""
        assert _teks_klausa("KDB melampaui.  Namun, KDH memenuhi") == \
            "kdb melampaui. namun, kdh memenuhi"

    def test_proporsi_penyebut_nol_mengembalikan_none(self):
        """0.0 akan terbaca 'gagal total'; yang benar adalah 'tak terukur'."""
        assert _proporsi(0, 0) is None
        assert _proporsi(3, 4) == 0.75


class TestNegasiBerlingkupKlausa:
    def test_negator_di_klausa_yang_sama_terbaca(self):
        t = _teks_klausa("kegiatan ini tidak memenuhi ketentuan")
        assert _ternegasi(t, t.index("memenuhi")) is True

    def test_negator_di_klausa_sebelumnya_tidak_dihitung(self):
        """Titik memutus lingkup: negasi di kalimat sebelumnya tak boleh membalik kalimat ini."""
        t = _teks_klausa("KDB tidak bermasalah. KDH melampaui ambang minimal")
        assert _ternegasi(t, t.index("melampaui")) is False

    def test_konjungsi_kontrastif_memutus_lingkup(self):
        t = _teks_klausa("tidak ada kendala sehingga memerlukan mitigasi")
        assert _ternegasi(t, t.index("memerlukan mitigasi")) is False, \
            "'sehingga' memulai klausa baru, negasi di depannya tak lagi berlaku"

    def test_konjungsi_aditif_tidak_memutus_lingkup(self):
        """"tidak A atau B" menegasikan keduanya — 'atau'/'dan' bukan batas klausa."""
        t = _teks_klausa("tidak melampaui KDB atau KLB")
        assert _ternegasi(t, t.index("melampaui")) is True

    def test_regresi_negator_35_karakter_di_depan(self):
        """Jendela 30 char sempat menandai kalimat ini menyimpang. Lihat docstring modul."""
        t = _teks_klausa("Kategori Sedang berpotensi menimbulkan dampak, "
                         "namun tidak mencapai tingkat tinggi yang memerlukan mitigasi drastis")
        assert _ternegasi(t, t.index("memerlukan mitigasi")) is True

    def test_regresi_negator_88_karakter_di_depan(self):
        """Jendela 80 char pun masih terlalu pendek — batas klausa yang menentukan, bukan cap."""
        t = _teks_klausa("dapat dikategorikan rendah. tidak ada indikasi bahwa kegiatan tersebut "
                         "berada di zona rawan bencana kekeringan atau memerlukan mitigasi tambahan")
        assert _ternegasi(t, t.index("memerlukan mitigasi")) is True


class TestKlaimArah:
    def test_memenuhi_afirmatif_adalah_klaim_patuh(self):
        assert _klaim_arah(_teks_klausa("KDB memenuhi ambang")) == {"patuh"}

    def test_memenuhi_ternegasi_adalah_klaim_langgar(self):
        assert _klaim_arah(_teks_klausa("KDB tidak memenuhi ambang")) == {"langgar"}

    def test_melampaui_afirmatif_adalah_klaim_langgar(self):
        assert _klaim_arah(_teks_klausa("KDB melampaui ambang")) == {"langgar"}

    def test_melampaui_ternegasi_adalah_klaim_patuh(self):
        """Negasi diperlakukan sebagai PEMBALIK, bukan kata terlarang — tanpa ini kalimat benar
        'tidak melampaui ambang' akan terhitung sebagai klaim pelanggaran."""
        assert _klaim_arah(_teks_klausa("KDB tidak melampaui ambang")) == {"patuh"}

    def test_dua_arah_sekaligus_terbaca_keduanya(self):
        assert _klaim_arah(_teks_klausa("KDB memenuhi, KDH melampaui")) == {"patuh", "langgar"}

    def test_tanpa_kata_kerja_arah_tidak_ada_klaim(self):
        assert _klaim_arah(_teks_klausa("KDB sebesar 47 persen")) == set()


class TestJendelaParameter:
    _KALIMAT = ("KDB melebihi 10%, KLB melebihi 1.0, serta KDH kurang dari 88% "
                "sebagaimana Lampiran VI")

    def test_jendela_dipotong_di_parameter_lain(self):
        """Regresi inti: tanpa pemotongan ini, jendela KDH menyerap 'melebihi' milik KLB dan
        parameter yang arahnya benar ditandai salah."""
        t = _teks_klausa(self._KALIMAT)
        pos = _posisi_parameter(t, ["kdb", "klb", "kdh"])
        idx_kdh = next(i for i, p in enumerate(pos) if p[2] == "kdh")
        jendela = _jendela_parameter(t, pos, idx_kdh)
        assert "kdh" in jendela
        assert "melebihi" not in jendela, "kata kerja milik KLB tak boleh masuk jendela KDH"

    def test_jendela_memuat_kata_kerja_miliknya_sendiri(self):
        t = _teks_klausa(self._KALIMAT)
        pos = _posisi_parameter(t, ["kdb", "klb", "kdh"])
        idx_kdb = next(i for i, p in enumerate(pos) if p[2] == "kdb")
        assert "melebihi" in _jendela_parameter(t, pos, idx_kdb)

    def test_nama_panjang_parameter_ikut_terbaca(self):
        t = _teks_klausa("koefisien dasar bangunan melampaui ambang")
        assert [p[2] for p in _posisi_parameter(t, ["kdb"])] == ["kdb"]

    def test_penyebutan_parameter_sama_tidak_memotong(self):
        """Narasi sah menyebut satu parameter dua kali dalam satu klausa."""
        t = _teks_klausa("KDB usulan 58% sedangkan KDB maksimal 10% jadi melampaui")
        pos = _posisi_parameter(t, ["kdb"])
        assert len(pos) == 2
        assert "melampaui" in _jendela_parameter(t, pos, 1)


class TestArahIntensitas:
    def test_semua_parameter_gagal_dan_narasi_benar(self):
        k = _kasus("KDB melebihi 10%, KLB melebihi 1.0, serta KDH kurang dari 88% dan tidak "
                   "memenuhi ketentuan minimal",
                   konteks=_konteks(fakta={"parameter": _parameter(False, False, False),
                                           "target": {}}))
        dinilai, sesuai, ambigu, catatan = _arah_intensitas(k)
        assert catatan == []
        assert sesuai == dinilai and dinilai >= 2

    def test_arah_terbalik_tercatat(self):
        k = _kasus("KDB memenuhi ketentuan dengan baik",
                   konteks=_konteks(fakta={"parameter": _parameter(False, True, True),
                                           "target": {}}))
        _dinilai, _sesuai, _ambigu, catatan = _arah_intensitas(k)
        assert any(c.startswith("kdb:") for c in catatan)

    def test_parameter_tak_disebut_tidak_masuk_penyebut(self):
        k = _kasus("Usulan sudah sesuai Lampiran VI tanpa menyebut parameter apa pun",
                   konteks=_konteks(fakta={"parameter": _parameter(), "target": {}}))
        dinilai, _sesuai, _ambigu, _catatan = _arah_intensitas(k)
        assert dinilai == 0

    def test_ambigu_tidak_masuk_penyebut(self):
        """Regresi: kasus ambigu sempat dihitung sebagai 'salah', bukan 'tak bisa dinilai'."""
        k = _kasus("KDB memenuhi sebagian tetapi KDB melampaui ambang juga",
                   konteks=_konteks(fakta={"parameter": {"kdb": _parameter()["kdb"]},
                                           "target": {}}))
        dinilai, sesuai, ambigu, catatan = _arah_intensitas(k)
        assert (dinilai, sesuai, ambigu, catatan) == (0, 0, 1, [])

    def test_kdh_ambang_minimum_tidak_tertukar_arahnya(self):
        """KDH dilanggar dengan berada DI BAWAH ambang; kosakata arah harus netral terhadap itu."""
        k = _kasus("KDH tidak memenuhi ambang minimal",
                   konteks=_konteks(fakta={"parameter": {"kdh": _parameter(kdh=False)["kdh"]},
                                           "target": {}}))
        dinilai, sesuai, _ambigu, catatan = _arah_intensitas(k)
        assert (dinilai, sesuai, catatan) == (1, 1, [])


class TestArahMitigasiDampak:
    def _k(self, panjang, perlu):
        konteks = _konteks(poin_id="dampak", kategori="Dampak", tipe_rekomendasi="numerik-mitigasi",
                           status="Sedang", fakta={"mitigasi": {"perlu_mitigasi": perlu,
                                                                "arah": []}})
        return _kasus(panjang, konteks=konteks, poin_id="dampak")

    def test_klaim_perlu_saat_fakta_tak_perlu_menyimpang(self):
        bisa, sesuai, catatan = self._arah("Kategori Sedang signifikan sehingga memerlukan "
                                           "tindakan mitigasi", perlu=False)
        assert (bisa, sesuai) == (True, False)
        assert "narasi=perlu" in catatan

    def test_klaim_ternegasi_saat_fakta_tak_perlu_sesuai(self):
        bisa, sesuai, _ = self._arah("Kategori Sedang berdampak, namun tidak mencapai tingkat "
                                     "tinggi yang memerlukan mitigasi", perlu=False)
        assert (bisa, sesuai) == (True, True)

    def test_klaim_perlu_saat_fakta_perlu_sesuai(self):
        bisa, sesuai, _ = self._arah("Kategori Tinggi memerlukan mitigasi", perlu=True)
        assert (bisa, sesuai) == (True, True)

    def test_tanpa_klaim_tidak_bisa_dinilai(self):
        bisa, _sesuai, _ = self._arah("Kategori Sedang menandakan perubahan aliran air", perlu=False)
        assert bisa is False

    def test_kedua_klaim_sekaligus_tidak_bisa_dinilai(self):
        bisa, _sesuai, _ = self._arah("memerlukan mitigasi tetapi tidak memerlukan mitigasi "
                                      "tambahan", perlu=False)
        assert bisa is False

    def _arah(self, panjang, perlu):
        return _arah_mitigasi_dampak(self._k(panjang, perlu))


class TestArahKategoriItbx:
    def _k(self, panjang, status):
        konteks = _konteks(poin_id="itbx", kategori="ITBX", tipe_rekomendasi="kategorikal",
                           status=status, fakta={"lolos": status == "I"})
        return _kasus(panjang, konteks=konteks, poin_id="itbx")

    def test_klaim_cocok_dengan_status(self):
        bisa, sesuai, _ = _arah_kategori_itbx(
            self._k("Kegiatan ini termasuk dalam kategori ITBX I (Diizinkan).", "I"))
        assert (bisa, sesuai) == (True, True)

    def test_klaim_berbeda_dari_status_menyimpang(self):
        bisa, sesuai, catatan = _arah_kategori_itbx(
            self._k("Kegiatan ini termasuk dalam kategori ITBX I (Diizinkan).", "X"))
        assert (bisa, sesuai) == (True, False)
        assert "fakta=X" in catatan

    def test_penyebutan_kategori_lain_di_kalimat_terpisah_tidak_dinilai(self):
        """Hanya pola self-classification yang dihitung; menyebut kategori lain sebagai konteks
        adalah penjelasan yang sah, bukan klaim."""
        bisa, _sesuai, _ = _arah_kategori_itbx(
            self._k("Status permohonan ini X. Kegiatan ITBX T di zona ini meliputi ruko.", "X"))
        assert bisa is False

    def test_klasifikasi_tanpa_huruf_tidak_dinilai(self):
        bisa, _sesuai, _ = _arah_kategori_itbx(
            self._k("Kegiatan ini termasuk kategori yang tidak diizinkan.", "X"))
        assert bisa is False


class TestJalurNarasi:
    def test_template_low_confidence_terdeteksi(self):
        """Dideteksi dengan MEMBANGKITKAN ULANG template dari konteks yang sama, bukan dengan
        mencocokkan string yang disalin ke test — kalau templatenya berubah, deteksinya ikut."""
        from app.reasoning.templates import template_low_confidence

        konteks = _konteks()
        assert _jalur_narasi(konteks, template_low_confidence(konteks)) == \
            "template_low_confidence"

    def test_template_aman_terdeteksi(self):
        from app.reasoning.templates import template_aman

        konteks = _konteks(status="MEMENUHI_SYARAT",
                           fakta={"parameter": _parameter(), "target": {}})
        assert _jalur_narasi(konteks, template_aman(konteks)) == "template_aman"

    def test_narasi_llm_bukan_template(self):
        konteks = _konteks()
        assert _jalur_narasi(konteks, _keluaran("KDB melampaui ambang Lampiran VI zona "
                                               "Perumahan")) == "llm"

    def test_metrik_mutu_hanya_menilai_jalur_llm(self):
        """Regresi: satu teks template pernah muncul di 11 permohonan dan membuat `kekhususan`
        poin dampak keluar 0.0% — yang terukur template, bukan model."""
        from app.reasoning.templates import template_low_confidence

        konteks = _konteks()
        tmpl = Kasus(permohonan="APP-T", timestamp="2026-09-20T10:00:00", poin_id="intensitas",
                     konteks=konteks, keluaran=template_low_confidence(konteks),
                     zona="Zona Perumahan", zona_subzone=None,
                     jalur=_jalur_narasi(konteks, template_low_confidence(konteks)))
        assert tmpl.jalur == "template_low_confidence"
        j = sebaran_jalur([tmpl])
        assert j["proporsi_llm"] == 0.0
        assert kekhususan([k for k in [tmpl] if k.jalur == "llm"])["semua_jangkar"]["nilai"] is None


class TestKekhususan:
    # Konteks dgn HANYA kdb yang gagal: jangkar wajibnya jadi kdb saja, sehingga tes di bawah
    # benar-benar menguji jangkar zona dan bukan ikut gagal karena klb/kdh tak disebut.
    _KDB_GAGAL = {"parameter": {"kdb": _parameter(kdb=False)["kdb"]}, "target": {}}

    def test_jangkar_zona_wajib_untuk_intensitas(self):
        tanpa = kekhususan([_kasus("KDB melampaui ambang Lampiran VI",
                                   konteks=_konteks(fakta=self._KDB_GAGAL))])
        assert tanpa["per_poin"]["intensitas"]["kena"] == 0
        dengan = kekhususan([_kasus("KDB melampaui ambang Lampiran VI zona Perumahan",
                                    konteks=_konteks(fakta=self._KDB_GAGAL))])
        assert dengan["per_poin"]["intensitas"]["kena"] == 1

    def test_jangkar_zona_hanya_informasional_untuk_dampak(self):
        """Ketentuan dampak adalah pasal prosa lintas-zona, jadi menuntut nama zona di sana akan
        menandai jawaban yang benar sebagai salah — sejalan `metrik_atribusi._POIN_BERZONA`."""
        konteks = _konteks(poin_id="dampak", kategori="Dampak",
                           tipe_rekomendasi="numerik-mitigasi", status="Sedang",
                           fakta={"mitigasi": {"perlu_mitigasi": False, "arah": []}})
        h = kekhususan([_kasus("Kategori Sedang sesuai Pasal 53 ayat 3", konteks=konteks,
                               poin_id="dampak")])
        assert h["per_poin"]["dampak"]["kena"] == 1
        assert "dampak/zona_pemohon (informasional)" in h["per_jangkar"]

    def test_kbli_hanya_informasional(self):
        """KBLI mengidentifikasi fakta yang sama dengan nama kegiatan — mewajibkan keduanya
        menghitung satu sinyal relevansi dua kali."""
        konteks = _konteks(poin_id="itbx", kategori="ITBX", tipe_rekomendasi="kategorikal",
                           status="I", fakta={"kegiatan_diusulkan": "Rumah Tunggal",
                                              "kbli_diusulkan": "41011"})
        h = kekhususan([_kasus("Rumah Tunggal diizinkan di zona Perumahan", konteks=konteks,
                               poin_id="itbx")])
        assert h["per_poin"]["itbx"]["kena"] == 1, "tanpa KBLI pun tetap dihitung spesifik"
        assert h["per_jangkar"]["itbx/kbli_diusulkan (informasional)"]["kena"] == 0

    def test_subzona_diterima_sebagai_jangkar_zona(self):
        h = kekhususan([_kasus("KDB melampaui ambang sub-zona R-2", subzona="R-2",
                               konteks=_konteks(fakta=self._KDB_GAGAL))])
        assert h["per_poin"]["intensitas"]["kena"] == 1


class TestFaithfulnessNumerik:
    def test_tanpa_chunk_dilaporkan_tak_terukur(self):
        """Tanpa teks chunk yang disitasi, cek provenance akan menolak angka yang sah — hasilnya
        angka yang SALAH, bukan angka yang hilang. Jadi wajib 'tak terukur'."""
        h = faithfulness_numerik([_kasus("KDB 58% melampaui 10%")], {})
        assert h["terukur"] is False
        assert "alasan" in h

    def test_angka_dari_fakta_poin_terlacak(self):
        k = _kasus("KDB usulan 47.0 persen dengan ambang 60.0 persen",
                   sitasi=[SitasiOutput(citation_id="anchor-0", dokumen="RDTR", pasal="44",
                                        halaman=1, kutipan="x", terverifikasi=True)])
        h = faithfulness_numerik([k], {"dummy": None})
        assert h["terukur"] is True
        assert h["per_narasi"]["nilai"] == 1.0

    def test_angka_karangan_tertangkap(self):
        k = _kasus("KDB usulan 47.0 persen padahal ambang sesungguhnya 93.7 persen")
        h = faithfulness_numerik([k], {"dummy": None})
        assert h["per_narasi"]["nilai"] == 0.0
        assert h["contoh_gagal"][0]["angka_tak_terlacak"]


class TestBoilerplate:
    def test_narasi_identik_lintas_permohonan_berstatus_beda_tertangkap(self):
        panjang = ("Berdasarkan ketentuan yang berlaku poin ini dinilai sesuai dengan aturan "
                   "intensitas pemanfaatan ruang yang ditetapkan dalam lampiran peraturan")
        a = _kasus(panjang, permohonan="APP-1")
        b = _kasus(panjang, permohonan="APP-2",
                   konteks=_konteks(status="MEMENUHI_SYARAT"))
        h = boilerplate([a, b])
        assert h["per_poin"]["intensitas"]["status_berbeda"]["maks"] == 1.0
        assert h["per_poin"]["intensitas"]["status_berbeda"][f"n_di_atas_{h['ambang']}"] == 2

    def test_permohonan_yang_sama_tidak_dipasangkan(self):
        """Dua poin dari permohonan yang sama tak mengatakan apa pun soal kekhususan."""
        panjang = ("Narasi panjang yang cukup untuk membentuk sejumlah lima gram kata agar "
                   "perbandingan jaccard bisa dihitung dengan wajar di sini")
        h = boilerplate([_kasus(panjang, permohonan="APP-1"),
                         _kasus(panjang, permohonan="APP-1",
                                konteks=_konteks(status="MEMENUHI_SYARAT"))])
        assert h["per_poin"]["intensitas"]["status_berbeda"]["maks"] == 0.0

    def test_narasi_berbeda_tidak_tertangkap(self):
        h = boilerplate([
            _kasus("KDB sebesar lima puluh delapan persen melampaui ambang sepuluh persen untuk "
                   "sub zona perumahan kepadatan sedang", permohonan="APP-1"),
            _kasus("Seluruh parameter intensitas berada dalam ambang yang ditetapkan sehingga "
                   "tidak ada catatan untuk poin ini", permohonan="APP-2",
                   konteks=_konteks(status="MEMENUHI_SYARAT")),
        ])
        assert h["per_poin"]["intensitas"]["status_berbeda"]["maks"] < 0.5


class TestMuatKasus:
    def _baris(self, timestamp, panjang, nomor="APP-1", diagnostik=None):
        # Fixture amplop back-end terbungkus {"statusCode", "message", "data"}; L2Assessment-nya
        # ada di bawah "data" — itu bentuk yang juga tersimpan di logs/precheck.jsonl.
        req = json.loads(_FIXTURE.read_text(encoding="utf-8"))["data"]
        req["application_number"] = nomor
        return json.dumps({
            "timestamp": timestamp,
            "request": req,
            **({"diagnostik": diagnostik} if diagnostik is not None else {}),
            "response": {"poin": [{
                "poin_id": "intensitas", "kategori": "Intensitas", "status": "MEMENUHI_SYARAT",
                "reasoning_pendek": panjang[:60], "reasoning_panjang": panjang, "sitasi": [],
                "rekomendasi": {"tipe": "numerik", "target": None, "saran": "x",
                                "disclaimer": None, "langkah_konkret": []},
                "low_confidence": False,
            }]},
        }, ensure_ascii=False)

    def test_dedup_mengambil_jalan_terbaru(self, tmp_path):
        """Log bukan trafik unik — satu permohonan bisa di-replay puluhan kali selama pengembangan,
        dan yang paling sering diulang justru yang paling bermasalah, jadi biasnya tidak acak."""
        p = tmp_path / "precheck.jsonl"
        p.write_text("\n".join([
            self._baris("2026-09-01T10:00:00", "narasi lama tentang intensitas zona perumahan"),
            self._baris("2026-09-20T10:00:00", "narasi baru tentang intensitas zona perumahan"),
        ]) + "\n", encoding="utf-8")
        kasus, lewat = muat_kasus(p)
        assert len(kasus) == 1
        assert kasus[0].keluaran.reasoning_panjang.startswith("narasi baru")
        assert lewat["jalan ulang permohonan yg sama (dedup)"] == 1

    def test_semua_jalan_mematikan_dedup(self, tmp_path):
        p = tmp_path / "precheck.jsonl"
        p.write_text("\n".join([
            self._baris("2026-09-01T10:00:00", "narasi satu tentang intensitas zona perumahan"),
            self._baris("2026-09-20T10:00:00", "narasi dua tentang intensitas zona perumahan"),
        ]) + "\n", encoding="utf-8")
        kasus, _ = muat_kasus(p, semua_jalan=True)
        assert len(kasus) == 2

    def test_sejak_membuang_versi_kode_lama(self, tmp_path):
        p = tmp_path / "precheck.jsonl"
        p.write_text("\n".join([
            self._baris("2026-08-01T10:00:00", "narasi lama", nomor="APP-1"),
            self._baris("2026-09-20T10:00:00", "narasi baru", nomor="APP-2"),
        ]) + "\n", encoding="utf-8")
        kasus, lewat = muat_kasus(p, sejak="2026-09-01")
        assert [k.permohonan for k in kasus] == ["APP-2"]
        assert lewat["sebelum 2026-09-01"] == 1

    def test_narasi_stub_dibuang(self, tmp_path):
        p = tmp_path / "precheck.jsonl"
        p.write_text(self._baris("2026-09-20T10:00:00",
                                 "Reasoning panjang stub yang jelas") + "\n", encoding="utf-8")
        kasus, lewat = muat_kasus(p)
        assert kasus == []
        assert lewat["narasi stub (pytest/fixture)"] == 1

    def test_jalur_ikut_terisi(self, tmp_path):
        p = tmp_path / "precheck.jsonl"
        p.write_text(self._baris("2026-09-20T10:00:00",
                                 "KDB berada dalam ambang zona perumahan") + "\n",
                     encoding="utf-8")
        kasus, _ = muat_kasus(p)
        assert [k.jalur for k in kasus] == ["llm"]

    def test_diagnostik_dibawa_ikut_per_poin(self, tmp_path):
        """Dibawa lewat jalur pemuatan yang SAMA, bukan jalur kedua — supaya seluruh filter
        (stub/mock/era lama/--sejak) dan dedup berlaku sama utk `laju_tolak_guardrail`."""
        riwayat = [{"percobaan": 1, "sebab": "guardrail_menolak", "masalah": ["x kosong."],
                    "exception": None}]
        diagnostik = [
            {"poin_id": "intensitas", "sebab": "berhasil", "riwayat_percobaan": riwayat},
            {"poin_id": "dampak", "sebab": "berhasil", "riwayat_percobaan": []},
        ]
        p = tmp_path / "precheck.jsonl"
        p.write_text(self._baris("2026-09-20T10:00:00", "KDB berada dalam ambang zona perumahan",
                                 diagnostik=diagnostik) + "\n", encoding="utf-8")

        kasus, _ = muat_kasus(p)

        assert len(kasus) == 1
        assert kasus[0].poin_id == "intensitas"
        assert kasus[0].riwayat_percobaan == riwayat

    def test_diagnosa_none_saat_baris_era_lama(self, tmp_path):
        p = tmp_path / "precheck.jsonl"
        p.write_text(self._baris("2026-09-20T10:00:00",
                                 "KDB berada dalam ambang zona perumahan") + "\n",
                     encoding="utf-8")
        kasus, _ = muat_kasus(p)
        assert kasus[0].diagnosa is None
        assert kasus[0].riwayat_percobaan == []

    def test_dedup_membawa_diagnostik_jalan_terbaru(self, tmp_path):
        """Riwayat percobaan harus ikut jalan yang dipilih dedup, bukan tertinggal di jalan lama —
        kalau tidak, laju tolaknya dihitung dari percobaan permohonan versi lain."""
        lama = [{"poin_id": "intensitas", "riwayat_percobaan": [
            {"percobaan": 1, "sebab": "guardrail_menolak", "masalah": ["lama."], "exception": None}]}]
        baru = [{"poin_id": "intensitas", "riwayat_percobaan": [
            {"percobaan": 1, "sebab": "berhasil", "masalah": [], "exception": None}]}]
        p = tmp_path / "precheck.jsonl"
        p.write_text("\n".join([
            self._baris("2026-09-01T10:00:00", "narasi lama tentang intensitas zona perumahan",
                        diagnostik=lama),
            self._baris("2026-09-20T10:00:00", "narasi baru tentang intensitas zona perumahan",
                        diagnostik=baru),
        ]) + "\n", encoding="utf-8")

        kasus, _ = muat_kasus(p)

        assert len(kasus) == 1
        assert kasus[0].riwayat_percobaan[0]["sebab"] == "berhasil"
        assert laju_tolak_guardrail(kasus)["percobaan_pertama"] == {"n": 1, "ditolak": 0,
                                                                   "nilai": 0.0}


class TestKetepatanArahGabungan:
    def test_penyebut_gabungan_tidak_memuat_kasus_ambigu(self):
        k_ambigu = _kasus("KDB memenuhi sebagian tetapi KDB melampaui ambang juga",
                          konteks=_konteks(fakta={"parameter": {"kdb": _parameter()["kdb"]},
                                                  "target": {}}))
        h = ketepatan_arah([k_ambigu])
        assert h["parameter_intensitas"]["n"] == 0
        assert h["parameter_intensitas"]["ambigu_dilewati"] == 1
        assert h["gabungan"]["nilai"] is None, "tak ada yang bisa dinilai -> bukan 0%"


class TestLabelMasalah:
    """Normalisasi label temuan — tanpa ini sebaran masalah cuma jadi daftar satu-satu, karena
    pesan guardrail menyisipkan nilai konkret ke dalam tanda kutip."""

    def test_angka_di_dalam_kutip_dikelompokkan(self):
        a = _label_masalah("Reasoning/saran menyebutkan angka '96,1' yang tidak tercantum")
        b = _label_masalah("Reasoning/saran menyebutkan angka '48' yang tidak tercantum")
        assert a == b
        assert "'X'" in a

    def test_frasa_dan_kategori_dikelompokkan(self):
        a = _label_masalah("Reasoning menyiratkan dampak tinggi ('risiko besar') padahal kategori aktual 'Rendah'.")
        b = _label_masalah("Reasoning menyiratkan dampak tinggi ('sangat signifikan') padahal kategori aktual 'Sedang'.")
        assert a == b

    def test_nomor_aturan_system_prompt_dipertahankan(self):
        """"aturan #8" (provenance angka) dan "aturan #17" (citation_id/titik koma) adalah cek yang
        BERBEDA — menormalkan nomornya akan menyatukan temuan yang penanganannya tak sama."""
        assert "#8" in _label_masalah("dilarang (SYSTEM_PROMPT aturan #8). Angka harus")
        assert "#17" in _label_masalah("DILARANG (SYSTEM_PROMPT aturan #17). Gunakan tanda baca")

    def test_angka_data_di_luar_kutip_dinormalkan(self):
        a = _label_masalah("runoff_change_index=0.42 jatuh di band 'Tinggi' menurut threshold_bands")
        b = _label_masalah("runoff_change_index=1.07 jatuh di band 'Sedang' menurut threshold_bands")
        assert a == b


class TestLajuTolakGuardrail:
    """`laju_tolak_guardrail` — faithfulness MENTAH model, satu-satunya metrik di modul ini yang
    menilai keluaran LLM sebelum dikoreksi.

    Dua kesalahan yang paling mudah terjadi di sini dan dikunci di bawah: (1) menghitung poin yang
    percobaan pertamanya gagal di panggilan LLM / retrieval sebagai "diterima guardrail", sehingga
    kegagalan infrastruktur terbaca sebagai kepatuhan model; (2) melaporkan 0% saat yang sebenarnya
    terjadi adalah log belum punya datanya.
    """

    def test_tanpa_riwayat_melaporkan_tak_terukur_bukan_nol(self):
        """Seluruh log sebelum 2026-10-09 tak punya `riwayat_percobaan`. Jawaban yang benar adalah
        "tak terukur"; 0.0% akan terbaca sebagai "guardrail tak pernah menolak apa pun"."""
        h = laju_tolak_guardrail([_kasus("narasi apa pun"), _kasus("narasi lain")])
        assert h["terukur"] is False
        assert h["n_tanpa_riwayat"] == 2
        assert "percobaan_pertama" not in h, "tanpa data, jangan sajikan angka sama sekali"

    def test_laju_tolak_percobaan_pertama(self):
        kasus = [
            _kasus("a", permohonan="APP-1", diagnosa=_riwayat(("guardrail_menolak", ["x kosong."]),
                                                              ("berhasil", []))),
            _kasus("b", permohonan="APP-2", diagnosa=_riwayat(("berhasil", []))),
            _kasus("c", permohonan="APP-3", diagnosa=_riwayat(("berhasil", []))),
            _kasus("d", permohonan="APP-4", diagnosa=_riwayat(("berhasil", []))),
        ]
        h = laju_tolak_guardrail(kasus)
        assert h["terukur"] is True
        assert h["percobaan_pertama"] == {"n": 4, "ditolak": 1, "nilai": 0.25}

    def test_percobaan_pertama_gagal_di_llm_keluar_dari_penyebut(self):
        """Rate limit/timeout bukan penilaian guardrail. Kalau ikut dihitung sbg "tidak ditolak",
        laju tolaknya turun setiap kali kuota Groq habis — mengukur cuaca, bukan model."""
        kasus = [
            _kasus("a", permohonan="APP-1",
                   diagnosa=_riwayat(("panggilan_llm_gagal", []), ("berhasil", []))),
            _kasus("b", permohonan="APP-2",
                   diagnosa=_riwayat(("guardrail_menolak", ["x kosong."]), ("berhasil", []))),
        ]
        h = laju_tolak_guardrail(kasus)
        assert h["percobaan_pertama"] == {"n": 1, "ditolak": 1, "nilai": 1.0}

    def test_retrieval_gagal_tanpa_riwayat_tidak_terhitung_lolos(self):
        kasus = [
            _kasus("a", permohonan="APP-1", diagnosa={"sebab": "retrieval_provider_gagal",
                                                      "riwayat_percobaan": []}),
            _kasus("b", permohonan="APP-2", diagnosa=_riwayat(("berhasil", []))),
        ]
        h = laju_tolak_guardrail(kasus)
        assert h["percobaan_pertama"] == {"n": 1, "ditolak": 0, "nilai": 0.0}
        assert h["n_tanpa_riwayat"] == 1

    def test_sebaran_masalah_dinormalkan_dan_hanya_dari_percobaan_pertama(self):
        kasus = [
            _kasus("a", permohonan="APP-1", diagnosa=_riwayat(
                ("guardrail_menolak", ["menyebutkan angka '96,1' yang tidak tercantum"]),
                ("guardrail_menolak", ["rekomendasi.saran kosong."]),
                ("berhasil", []))),
            _kasus("b", permohonan="APP-2", diagnosa=_riwayat(
                ("guardrail_menolak", ["menyebutkan angka '48' yang tidak tercantum"]),
                ("berhasil", []))),
        ]
        h = laju_tolak_guardrail(kasus)
        assert h["n_temuan"] == 2
        assert list(h["sebaran_masalah"].values()) == [2], \
            "dua angka berbeda -> satu label setelah normalisasi"
        assert all("kosong" not in lbl for lbl in h["sebaran_masalah"]), \
            "temuan percobaan ke-2 tidak boleh ikut: metrik ini tentang keluaran MENTAH"

    def test_pulih_setelah_tolak_mengukur_efek_guardrail(self):
        kasus = [
            _kasus("a", permohonan="APP-1", diagnosa=_riwayat(("guardrail_menolak", ["x."]),
                                                              ("berhasil", []))),
            _kasus("b", permohonan="APP-2", diagnosa=_riwayat(("guardrail_menolak", ["x."]),
                                                              ("guardrail_menolak", ["x."]),
                                                              ("guardrail_menolak", ["x."]))),
        ]
        h = laju_tolak_guardrail(kasus)
        assert h["pulih_setelah_tolak"] == {"n": 2, "kena": 1, "nilai": 0.5}

    def test_per_percobaan_memisahkan_indeks(self):
        kasus = [_kasus("a", permohonan="APP-1", diagnosa=_riwayat(
            ("guardrail_menolak", ["x."]), ("guardrail_menolak", ["x."]), ("berhasil", [])))]
        h = laju_tolak_guardrail(kasus)
        assert h["per_percobaan"]["1"] == {"n": 1, "ditolak": 1, "nilai": 1.0}
        assert h["per_percobaan"]["3"] == {"n": 1, "ditolak": 0, "nilai": 0.0}

    def test_per_poin_dipisah(self):
        kasus = [
            _kasus("a", poin_id="itbx", permohonan="APP-1",
                   diagnosa=_riwayat(("guardrail_menolak", ["x."]), ("berhasil", []))),
            _kasus("b", poin_id="dampak", permohonan="APP-1",
                   diagnosa=_riwayat(("berhasil", []))),
        ]
        h = laju_tolak_guardrail(kasus)
        assert h["per_poin"]["itbx"]["nilai"] == 1.0
        assert h["per_poin"]["dampak"]["nilai"] == 0.0

    def test_narasi_template_low_confidence_tetap_masuk_hitungan(self):
        """Poin yang guardrail-nya menolak SEMUA percobaan berakhir sbg template_low_confidence.
        Membuangnya (spt yang benar dilakukan metrik 1-4) akan menyisakan hanya yang berhasil dan
        menekan laju tolaknya secara sistematis."""
        k = _kasus("teks template", permohonan="APP-1",
                   diagnosa=_riwayat(("guardrail_menolak", ["x."]), ("guardrail_menolak", ["x."])))
        k.jalur = "template_low_confidence"
        h = laju_tolak_guardrail([k])
        assert h["percobaan_pertama"]["n"] == 1
        assert h["pulih_setelah_tolak"]["nilai"] == 0.0

    def test_riwayat_rusak_diabaikan_tanpa_meledak(self):
        """Log bisa memuat baris dari versi lain / `diagnostik_rusak` — pembacanya tidak boleh
        mengasumsikan bentuknya."""
        kasus = [
            _kasus("a", permohonan="APP-1", diagnosa={"riwayat_percobaan": "bukan daftar"}),
            _kasus("b", permohonan="APP-2", diagnosa={"riwayat_percobaan": ["bukan dict", 3]}),
            _kasus("c", permohonan="APP-3", diagnosa=_riwayat(("berhasil", []))),
        ]
        h = laju_tolak_guardrail(kasus)
        assert h["percobaan_pertama"] == {"n": 1, "ditolak": 0, "nilai": 0.0}
        assert h["n_tanpa_riwayat"] == 2
