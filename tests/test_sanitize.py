"""tests/test_sanitize.py — Tahap 6: pagar PII eksplisit (app/sanitize.py) BENAR dipanggil di titik
pembentukan query retrieval (RetrieverAsli.search/get_by_reference) & prompt LLM (build_user_prompt),
bukan cuma diasumsikan lewat desain.
"""

from __future__ import annotations

import pytest

from app import sanitize
from app.reasoning.prompts import build_user_prompt
from app.retrieval.base import RetrievalFilters
from app.retrieval.retriever import RetrieverAsli
from app.schemas import PoinKonteks

_NIK = "3471012345670001"
_APP_NUMBER = "APP-2026-9999"
_PHONE = "081234567890"
_EMAIL = "petugas@sleman.go.id"
_COORD = "-7.783200, 110.478300"


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


# ---------------------------------------------------------------------- unit: sanitize_query_text
class TestSanitizeQueryText:
    def test_lolos_apa_adanya_utk_query_bersih(self):
        assert sanitize.sanitize_query_text("kdb") == "kdb"
        assert sanitize.sanitize_query_text("RDTR Sleman Tengah Pasal 43") == "RDTR Sleman Tengah Pasal 43"

    @pytest.mark.parametrize("bocor", [_NIK, _APP_NUMBER, _PHONE, _EMAIL, _COORD])
    def test_menolak_query_berpola_pii(self, bocor):
        with pytest.raises(sanitize.PIIDetectedError):
            sanitize.sanitize_query_text(f"kdb {bocor}")


# ------------------------------------------------------------------------- unit: sanitize_freetext
class TestSanitizeFreetext:
    def test_teks_bersih_tak_berubah(self):
        teks = "Kegiatan tergolong Terbatas (T) di zona Zona Perumahan sesuai Matriks ITBX RDTR."
        assert sanitize.sanitize_freetext(teks) == teks

    def test_redaksi_nik_dan_telepon(self):
        teks = f"Hubungi {_PHONE} utk verifikasi NIK {_NIK}."
        hasil = sanitize.sanitize_freetext(teks)
        assert _NIK not in hasil
        assert _PHONE not in hasil
        assert "[REDACTED:" in hasil

    def test_string_kosong_aman(self):
        assert sanitize.sanitize_freetext("") == ""


# --------------------------------------------------------------------------- unit: sanitize_fakta
class TestSanitizeFakta:
    def test_buang_key_di_luar_allowlist(self):
        fakta = {"lolos": True, "reason": "ok", "application_number": _APP_NUMBER, "nik": _NIK}
        hasil = sanitize.sanitize_fakta("itbx", fakta)
        assert "application_number" not in hasil
        assert "nik" not in hasil
        assert hasil["lolos"] is True

    def test_key_allowlist_legit_tetap_lolos(self):
        fakta = {
            "lolos": True, "kbli_diusulkan": "0111", "kegiatan_diusulkan": "WARUNG",
            "reason": "Lolos karena Terbatas (T)",
        }
        hasil = sanitize.sanitize_fakta("itbx", fakta)
        assert hasil["kbli_diusulkan"] == "0111"
        assert hasil["kegiatan_diusulkan"] == "WARUNG"

    def test_scrub_freetext_di_dalam_allowlist(self):
        fakta = {"lolos": True, "reason": f"Sesuai catatan, hubungi {_PHONE}."}
        hasil = sanitize.sanitize_fakta("itbx", fakta)
        assert _PHONE not in hasil["reason"]

    def test_scrub_freetext_list_keterangan_ketentuan(self):
        fakta = {"lolos": True, "reason": "ok", "keterangan_ketentuan": [f"Contoh: {_NIK}", "Syarat biasa"]}
        hasil = sanitize.sanitize_fakta("itbx", fakta)
        assert _NIK not in hasil["keterangan_ketentuan"][0]
        assert hasil["keterangan_ketentuan"][1] == "Syarat biasa"

    def test_poin_id_tak_dikenal_return_kosong(self):
        assert sanitize.sanitize_fakta("poin_id_asing", {"apa": "saja"}) == {}

    def test_forbidden_field_tetap_terbuang_meski_lolos_ke_allowlist(self):
        # Pertahanan lapis-2: seandainya field terlarang KEBETULAN masuk allowlist di masa depan
        # (human error saat edit ALLOWED_FAKTA_FIELDS), tetap harus terbuang.
        allowlist_asli = sanitize.ALLOWED_FAKTA_FIELDS["itbx"]
        try:
            sanitize.ALLOWED_FAKTA_FIELDS["itbx"] = allowlist_asli | {"application_number"}
            hasil = sanitize.sanitize_fakta("itbx", {"lolos": True, "application_number": _APP_NUMBER})
            assert "application_number" not in hasil
        finally:
            sanitize.ALLOWED_FAKTA_FIELDS["itbx"] = allowlist_asli


# ------------------------------------------------------- integrasi: build_user_prompt (prompt LLM)
class TestBuildUserPromptSanitasi:
    def test_field_terlarang_tak_pernah_masuk_prompt(self):
        poin = _poin(fakta={
            "lolos": True, "reason": "Lolos karena Diizinkan.",
            "application_number": _APP_NUMBER, "nik": _NIK, "nama_pemohon": "Budi Santoso",
        })
        prompt = build_user_prompt(poin, [], None, None)
        assert _APP_NUMBER not in prompt
        assert _NIK not in prompt
        assert "Budi Santoso" not in prompt

    def test_pola_pii_di_reason_diredaksi_di_prompt(self):
        poin = _poin(fakta={
            "lolos": True,
            "reason": f"Sesuai keterangan, hubungi {_PHONE} utk verifikasi NIK {_NIK}.",
        })
        prompt = build_user_prompt(poin, [], None, None)
        assert _NIK not in prompt
        assert _PHONE not in prompt
        assert "[REDACTED:" in prompt

    def test_field_legit_tetap_muncul_di_prompt(self):
        poin = _poin(status="T", fakta={
            "lolos": True, "kbli_diusulkan": "0111", "kegiatan_diusulkan": "WARUNG",
            "reason": "Lolos karena Terbatas (T) di zona Zona Perumahan.",
        })
        prompt = build_user_prompt(poin, [], None, None)
        assert "WARUNG" in prompt
        assert "Terbatas (T)" in prompt


# --------------------------------------------------- integrasi: titik pembentukan query retrieval
class TestRetrieverAsliSanitasi:
    def test_search_menolak_query_berpola_pii_sebelum_apa_pun_lain(self, monkeypatch):
        # Tidak perlu mock DB/embedding sama sekali — sanitize HARUS jadi baris pertama di search(),
        # jadi PIIDetectedError harus muncul sebelum kode lain (koneksi DB dst) sempat jalan.
        def gagal_kalau_dipanggil(*a, **kw):
            raise AssertionError("encode_dense_one tidak boleh dipanggil kalau query ditolak sanitasi")

        monkeypatch.setattr("app.retrieval.retriever.encode_dense_one", gagal_kalau_dipanggil)
        rt = RetrieverAsli()
        with pytest.raises(sanitize.PIIDetectedError):
            rt.search(f"kdb {_NIK}", RetrievalFilters())

    def test_search_memanggil_sanitize_utk_query_bersih(self, monkeypatch):
        panggilan = []
        asli = sanitize.sanitize_query_text

        def spy(text, **kw):
            panggilan.append(text)
            return asli(text, **kw)

        monkeypatch.setattr(sanitize, "sanitize_query_text", spy)
        monkeypatch.setattr("app.retrieval.db.connect", lambda dsn=None: object())

        def berhenti_di_sini(*a, **kw):
            raise RuntimeError("STOP_SETELAH_SANITASI")

        monkeypatch.setattr("app.retrieval.retriever.encode_dense_one", berhenti_di_sini)

        rt = RetrieverAsli()
        with pytest.raises(RuntimeError, match="STOP_SETELAH_SANITASI"):
            rt.search("kdb", RetrievalFilters())
        assert panggilan == ["kdb"]  # sanitasi terbukti dipanggil, DAN sebelum encode_dense_one

    def test_get_by_reference_menolak_referensi_berpola_pii(self, monkeypatch):
        def gagal_kalau_dipanggil(*a, **kw):
            raise AssertionError("db.match_dokumen tidak boleh dipanggil kalau referensi ditolak sanitasi")

        monkeypatch.setattr("app.retrieval.db.match_dokumen", gagal_kalau_dipanggil)
        rt = RetrieverAsli()
        with pytest.raises(sanitize.PIIDetectedError):
            rt.get_by_reference([f"RDTR Sleman Tengah Pasal 43 {_NIK}"])

    def test_get_by_reference_memanggil_sanitize_utk_referensi_bersih(self, monkeypatch):
        panggilan = []
        asli = sanitize.sanitize_query_text

        def spy(text, **kw):
            panggilan.append(text)
            return asli(text, **kw)

        monkeypatch.setattr(sanitize, "sanitize_query_text", spy)
        # Referensi bersih -> sanitize LOLOS -> get_by_reference lanjut sampai buka koneksi DB.
        # Stub `db.connect` (bukan DB sungguhan) supaya lanjut ke match_dokumen, yg dibuat gagal
        # utk membuktikan alur benar-benar sampai sana (bukan berhenti diam-diam lebih awal).
        monkeypatch.setattr("app.retrieval.db.connect", lambda dsn=None: object())

        def berhenti_di_sini(*a, **kw):
            raise RuntimeError("STOP_SETELAH_SANITASI")

        monkeypatch.setattr("app.retrieval.db.match_dokumen", berhenti_di_sini)

        rt = RetrieverAsli()
        with pytest.raises(RuntimeError, match="STOP_SETELAH_SANITASI"):
            rt.get_by_reference(["RDTR Sleman Tengah Pasal 43"])
        assert panggilan == ["RDTR Sleman Tengah Pasal 43"]
