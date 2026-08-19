import logging
import os

import pytest
from dotenv import load_dotenv
from openai import RateLimitError

load_dotenv()


def test_get_client_pakai_timeout_dan_max_retries_dari_konfigurasi(monkeypatch):
    # Config timeout/max_retries dibaca sekali saat modul di-load (mirip pola LLM_BASE_URL, beda
    # dari LLM_MODEL yang sengaja dibaca ulang tiap panggilan utk bakeoff). Test ini verifikasi
    # WIRING-nya benar (client dibangun pakai konstanta modul), bukan nilai numerik env spesifik.
    import app.reasoning.llm_client as llm_client_module

    monkeypatch.setenv("GROQ_API_KEY", "fake-key-utk-test")
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    monkeypatch.setattr(llm_client_module, "_rotator", None)

    client = llm_client_module._get_client()

    assert client.timeout == llm_client_module._TIMEOUT_S
    assert client.max_retries == llm_client_module._MAX_RETRIES
    # monkeypatch otomatis kembalikan _rotator ke nilai semula saat test ini selesai


def test_get_client_default_timeout_dan_retries_masuk_akal():
    import app.reasoning.llm_client as llm_client_module

    # default 30s / 2 retries kalau env tidak diisi — cukup responsif utk precheck sinkron,
    # tidak selama default SDK (~10 menit) yang terlalu lama utk web request.
    assert llm_client_module._TIMEOUT_S <= 60
    assert 0 <= llm_client_module._MAX_RETRIES <= 5


class TestOverrideEnvTerbatasSaatImport:
    """Bug ditemukan live full-suite (2026-08-19, Docker lokal mati): import ulang llm_client.py
    dgn load_dotenv(override=True) POLOS menimpa SELURUH isi .env, termasuk DATABASE_URL/RETRIEVER
    yang sengaja diproteksi tests/conftest.py — suite offline diam-diam coba konek Postgres
    sungguhan, hang bermenit-menit (dari <10s jadi >1000s). Sebelumnya lolos tanpa ketahuan krn
    Docker lokal kebetulan selalu jalan (koneksi ke DB asli SUKSES diam-diam, bukan hang).

    Fix: override hanya per-key milik modul ini (_ENV_KUNCI_LLM_CLIENT), bukan seluruh file .env.
    Test ini reload modul scr paksa (import ulang memicu ulang kode level-modul, termasuk override)
    supaya bug regresi ini tertangkap OFFLINE, tak perlu Docker mati dulu baru ketahuan lagi.
    """

    def test_reimport_tak_menimpa_var_yg_diproteksi_modul_lain(self, monkeypatch):
        import importlib

        import app.reasoning.llm_client as llm_client_module

        # Simulasikan proteksi tests/conftest.py: var ini SENGAJA diset ke nilai non-.env sebelum
        # modul di-import ulang — modul lain (bukan llm_client.py) yang "punya" var ini.
        monkeypatch.setenv("DATABASE_URL", "")
        monkeypatch.setenv("RETRIEVER", "mock")
        # Pastikan .env sungguhan (dibaca test ini via load_dotenv scr manual) MEMANG punya nilai
        # lain utk kedua var ini — kalau tidak, test ini false-negative (tak menguji apa pun).
        from dotenv import dotenv_values
        nilai_dotenv = dotenv_values()
        if not nilai_dotenv.get("DATABASE_URL") and not nilai_dotenv.get("RETRIEVER"):
            pytest.skip(".env tak punya DATABASE_URL/RETRIEVER utk dibandingkan — tak bisa uji bug ini")

        try:
            importlib.reload(llm_client_module)
            assert os.environ.get("DATABASE_URL") == ""  # TETAP kosong, bukan ketimpa isi .env
            assert os.environ.get("RETRIEVER") == "mock"  # TETAP mock, bukan ketimpa "asli"
        finally:
            importlib.reload(llm_client_module)  # pulihkan state modul utk test lain

    def test_reimport_tetap_refresh_key_groq_milik_sendiri(self, monkeypatch):
        # Kontrol positif — key yg MEMANG milik modul ini (GROQ_API_KEY) HARUS tetap ter-refresh
        # dari .env saat re-import, walau ada nilai basi di os.environ (kasus asli PM2 di VPS).
        import importlib

        import app.reasoning.llm_client as llm_client_module

        from dotenv import dotenv_values
        nilai_asli = dotenv_values().get("GROQ_API_KEY")
        if not nilai_asli:
            pytest.skip(".env tak punya GROQ_API_KEY — tak bisa uji refresh")

        monkeypatch.setenv("GROQ_API_KEY", "nilai-basi-simulasi-cache-pm2")
        try:
            importlib.reload(llm_client_module)
            assert os.environ.get("GROQ_API_KEY") == nilai_asli  # ketimpa isi .env, bukan basi
        finally:
            importlib.reload(llm_client_module)


class TestMuatDaftarKunci:
    def test_groq_api_keys_csv_diprioritaskan(self, monkeypatch):
        import app.reasoning.llm_client as llm_client_module

        monkeypatch.setenv("GROQ_API_KEYS", "kunci-a, kunci-b ,kunci-c")
        monkeypatch.setenv("GROQ_API_KEY", "kunci-tunggal-lama")
        assert llm_client_module._muat_daftar_kunci() == ["kunci-a", "kunci-b", "kunci-c"]

    def test_fallback_ke_groq_api_key_tunggal(self, monkeypatch):
        import app.reasoning.llm_client as llm_client_module

        monkeypatch.delenv("GROQ_API_KEYS", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "kunci-tunggal")
        assert llm_client_module._muat_daftar_kunci() == ["kunci-tunggal"]

    def test_tak_ada_kunci_sama_sekali_return_kosong(self, monkeypatch):
        import app.reasoning.llm_client as llm_client_module

        monkeypatch.delenv("GROQ_API_KEYS", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert llm_client_module._muat_daftar_kunci() == []

    def test_groq_api_keys_kosong_fallback_ke_tunggal(self, monkeypatch):
        # GROQ_API_KEYS="" (diset tapi kosong) -> BUKAN dianggap "ada 1 kunci kosong", fallback.
        import app.reasoning.llm_client as llm_client_module

        monkeypatch.setenv("GROQ_API_KEYS", "  ,  ")
        monkeypatch.setenv("GROQ_API_KEY", "kunci-tunggal")
        assert llm_client_module._muat_daftar_kunci() == ["kunci-tunggal"]


class TestRotasiKunciGroq:
    def test_kunci_awal_index_0_pada_panggilan_pertama(self):
        from app.reasoning.llm_client import _RotasiKunciGroq

        r = _RotasiKunciGroq(["kunci-1", "kunci-2"], base_url=None)
        idx = r.kunci_awal()
        assert idx == 0
        assert r.client_utk(idx).api_key == "kunci-1"

    def test_kunci_awal_round_robin_setiap_panggilan(self):
        # 2026-08-19: beda dari desain lama (sticky, semua panggilan pakai kunci yg sama sampai
        # gagal) — SEKARANG setiap panggilan generate() baru dapat kunci AWAL berikutnya scr
        # berurutan, supaya beban tersebar sejak awal (bukan menumpuk di kunci pertama).
        from app.reasoning.llm_client import _RotasiKunciGroq

        r = _RotasiKunciGroq(["kunci-1", "kunci-2", "kunci-3"], base_url=None)
        assert [r.kunci_awal() for _ in range(5)] == [0, 1, 2, 0, 1]  # muter balik stlh kunci ke-3

    def test_kunci_awal_thread_safe_tak_ada_index_dobel(self):
        # Increment counter dilindungi lock -> N panggilan concurrent hasilkan N index BERBEDA
        # (0..N-1 tiap siklus), bukan ada yang kebagian index sama krn race condition.
        import threading

        from app.reasoning.llm_client import _RotasiKunciGroq

        r = _RotasiKunciGroq(["kunci-1", "kunci-2", "kunci-3", "kunci-4"], base_url=None)
        hasil = []
        lock_hasil = threading.Lock()

        def _ambil():
            idx = r.kunci_awal()
            with lock_hasil:
                hasil.append(idx)

        threads = [threading.Thread(target=_ambil) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(hasil) == [0, 1, 2, 3]  # 4 kunci, 4 panggilan -> masing2 index kepakai persis 1x

    def test_tanpa_kunci_raise(self):
        from app.reasoning.llm_client import _RotasiKunciGroq

        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            _RotasiKunciGroq([], base_url=None)

    def test_client_di_cache_bukan_dibangun_ulang(self):
        from app.reasoning.llm_client import _RotasiKunciGroq

        r = _RotasiKunciGroq(["kunci-1"], base_url=None)
        client_a = r.client_utk(0)
        client_b = r.client_utk(0)
        assert client_a is client_b

    def test_client_utk_index_di_luar_jangkauan_dibungkus_modulo(self):
        from app.reasoning.llm_client import _RotasiKunciGroq

        r = _RotasiKunciGroq(["kunci-1", "kunci-2"], base_url=None)
        assert r.client_utk(2).api_key == "kunci-1"  # 2 % 2 == 0
        assert r.client_utk(3).api_key == "kunci-2"  # 3 % 2 == 1


class TestKlasifikasiRateLimit:
    def test_per_hari_tpd(self):
        from app.reasoning.llm_client import _klasifikasi_rate_limit

        pesan = "Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199992..."
        assert _klasifikasi_rate_limit(pesan) == "tpd"

    def test_per_menit_tpm(self):
        from app.reasoning.llm_client import _klasifikasi_rate_limit

        pesan = "Rate limit reached ... on tokens per minute (TPM): Limit 8000, Used 6990..."
        assert _klasifikasi_rate_limit(pesan) == "tpm"

    def test_per_menit_rpm(self):
        from app.reasoning.llm_client import _klasifikasi_rate_limit

        pesan = "Rate limit reached ... on requests per minute (RPM): Limit 30..."
        assert _klasifikasi_rate_limit(pesan) == "tpm"

    def test_format_tak_dikenal_dianggap_tpm_lebih_aman(self):
        from app.reasoning.llm_client import _klasifikasi_rate_limit

        assert _klasifikasi_rate_limit("pesan aneh yang tak sesuai format apapun") == "tak_diketahui"


class TestEkstrakTungguDetik:
    def test_ekstrak_angka_dari_pesan(self):
        from app.reasoning.llm_client import _ekstrak_tunggu_detik

        pesan = "... Please try again in 34.019999999s. Need more tokens?..."
        assert _ekstrak_tunggu_detik(pesan) == pytest.approx(34.02, abs=0.01)

    def test_dibatasi_tunggu_maks(self):
        from app.reasoning.llm_client import _ekstrak_tunggu_detik, _TUNGGU_MAKS_S

        pesan = "... Please try again in 99999s. ..."
        assert _ekstrak_tunggu_detik(pesan) == _TUNGGU_MAKS_S

    def test_tak_ada_angka_pakai_default(self):
        from app.reasoning.llm_client import _ekstrak_tunggu_detik, _TUNGGU_DEFAULT_S

        assert _ekstrak_tunggu_detik("pesan tanpa saran waktu sama sekali") == _TUNGGU_DEFAULT_S


class TestGenerateRotasiSaatRateLimit:
    """Simulasi RateLimitError dari client Groq via monkeypatch — TIDAK memanggil Groq nyata."""

    @staticmethod
    def _buat_rate_limit_error(pesan: str) -> RateLimitError:
        import httpx

        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(429, request=request, json={"error": {"message": pesan}})
        return RateLimitError(message=pesan, response=response, body=None)

    def test_tpd_rotasi_kunci_lalu_sukses(self, monkeypatch, caplog):
        import app.reasoning.llm_client as llm_client_module

        monkeypatch.setattr(llm_client_module, "_rotator", None)
        monkeypatch.setenv("GROQ_API_KEYS", "kunci-1,kunci-2")
        monkeypatch.setenv("LLM_MODEL", "model-tes")
        monkeypatch.setattr(llm_client_module.time, "sleep", lambda s: None)  # jangan benar2 tidur

        panggilan = {"n": 0, "kunci_dipakai": []}

        def _stub_panggil(client, messages, json_schema, schema_name, temperature, max_tokens):
            panggilan["n"] += 1
            panggilan["kunci_dipakai"].append(client.api_key)
            if panggilan["n"] == 1:
                raise self._buat_rate_limit_error(
                    "Rate limit reached for model x on tokens per day (TPD): Limit 200000, Used 199992"
                )

            class _Msg:
                content = '{"ok": true}'

            class _Choice:
                message = _Msg()

            class _Completion:
                choices = [_Choice()]

            return _Completion()

        monkeypatch.setattr(llm_client_module, "_panggil_llm_sekali", _stub_panggil)

        with caplog.at_level(logging.WARNING, logger="app.reasoning.llm_client"):
            hasil = llm_client_module.generate("prompt", {"type": "object"}, schema_name="tes")

        assert hasil == {"ok": True}
        assert panggilan["n"] == 2
        assert panggilan["kunci_dipakai"] == ["kunci-1", "kunci-2"]  # rotasi beneran pindah kunci
        assert "Rotasi GROQ_API_KEY" in caplog.text

    def test_tpm_tunggu_lalu_ulang_kunci_sama(self, monkeypatch):
        import app.reasoning.llm_client as llm_client_module

        monkeypatch.setattr(llm_client_module, "_rotator", None)
        monkeypatch.setenv("GROQ_API_KEYS", "kunci-1,kunci-2")
        monkeypatch.setenv("LLM_MODEL", "model-tes")

        tidur_dipanggil = {"detik": []}
        monkeypatch.setattr(llm_client_module.time, "sleep", lambda s: tidur_dipanggil["detik"].append(s))

        panggilan = {"n": 0, "kunci_dipakai": []}

        def _stub_panggil(client, messages, json_schema, schema_name, temperature, max_tokens):
            panggilan["n"] += 1
            panggilan["kunci_dipakai"].append(client.api_key)
            if panggilan["n"] == 1:
                raise self._buat_rate_limit_error(
                    "Rate limit reached for model x on tokens per minute (TPM): Limit 8000, Used 6990. "
                    "Please try again in 1.5s."
                )

            class _Msg:
                content = '{"ok": true}'

            class _Choice:
                message = _Msg()

            class _Completion:
                choices = [_Choice()]

            return _Completion()

        monkeypatch.setattr(llm_client_module, "_panggil_llm_sekali", _stub_panggil)

        hasil = llm_client_module.generate("prompt", {"type": "object"}, schema_name="tes")

        assert hasil == {"ok": True}
        assert panggilan["n"] == 2
        # TPM -> kunci SAMA diulang, BUKAN rotasi ke kunci-2.
        assert panggilan["kunci_dipakai"] == ["kunci-1", "kunci-1"]
        assert tidur_dipanggil["detik"] == [pytest.approx(1.5, abs=0.01)]

    def test_tpd_semua_kunci_habis_raise(self, monkeypatch):
        import app.reasoning.llm_client as llm_client_module

        monkeypatch.setattr(llm_client_module, "_rotator", None)
        monkeypatch.setenv("GROQ_API_KEYS", "kunci-1,kunci-2")
        monkeypatch.setenv("LLM_MODEL", "model-tes")
        monkeypatch.setattr(llm_client_module.time, "sleep", lambda s: None)

        def _stub_selalu_gagal(client, messages, json_schema, schema_name, temperature, max_tokens):
            raise self._buat_rate_limit_error(
                "Rate limit reached for model x on tokens per day (TPD): Limit 200000, Used 200000"
            )

        monkeypatch.setattr(llm_client_module, "_panggil_llm_sekali", _stub_selalu_gagal)

        with pytest.raises(RateLimitError):
            llm_client_module.generate("prompt", {"type": "object"}, schema_name="tes")

    def test_kunci_tunggal_tpd_langsung_raise_tanpa_loop_tak_terbatas(self, monkeypatch):
        # Kasus paling umum sekarang (1 kunci di .env) — HARUS tetap gagal cepat spt perilaku lama,
        # bukan berputar tanpa henti mencoba "rotasi" ke kunci yang sama.
        import app.reasoning.llm_client as llm_client_module

        monkeypatch.setattr(llm_client_module, "_rotator", None)
        monkeypatch.delenv("GROQ_API_KEYS", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "kunci-satu-satunya")
        monkeypatch.setenv("LLM_MODEL", "model-tes")
        monkeypatch.setattr(llm_client_module.time, "sleep", lambda s: None)

        panggilan = {"n": 0}

        def _stub_selalu_gagal(client, messages, json_schema, schema_name, temperature, max_tokens):
            panggilan["n"] += 1
            raise self._buat_rate_limit_error(
                "Rate limit reached for model x on tokens per day (TPD): Limit 200000, Used 200000"
            )

        monkeypatch.setattr(llm_client_module, "_panggil_llm_sekali", _stub_selalu_gagal)

        with pytest.raises(RateLimitError):
            llm_client_module.generate("prompt", {"type": "object"}, schema_name="tes")
        assert panggilan["n"] == 1  # 1 kunci -> 1 percobaan, langsung menyerah

    def test_panggilan_berturut_turut_mulai_dari_kunci_berbeda(self, monkeypatch):
        # 2026-08-19: ditemukan live di VPS — desain lama sticky bikin kunci pertama terus
        # dipakai sampai nyaris limit berulang, kunci lain menganggur. Sekarang tiap generate()
        # BARU (bukan retry di dalamnya) harus mulai dari kunci berikutnya scr round-robin.
        import app.reasoning.llm_client as llm_client_module

        monkeypatch.setattr(llm_client_module, "_rotator", None)
        monkeypatch.setenv("GROQ_API_KEYS", "kunci-1,kunci-2,kunci-3")
        monkeypatch.setenv("LLM_MODEL", "model-tes")

        kunci_dipakai = []

        def _stub_sukses(client, messages, json_schema, schema_name, temperature, max_tokens):
            kunci_dipakai.append(client.api_key)

            class _Msg:
                content = '{"ok": true}'

            class _Choice:
                message = _Msg()

            class _Completion:
                choices = [_Choice()]

            return _Completion()

        monkeypatch.setattr(llm_client_module, "_panggil_llm_sekali", _stub_sukses)

        for _ in range(4):
            llm_client_module.generate("prompt", {"type": "object"}, schema_name="tes")

        assert kunci_dipakai == ["kunci-1", "kunci-2", "kunci-3", "kunci-1"]


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY tidak diset — skip smoke test panggilan LLM nyata.",
)
def test_generate_returns_valid_json():
    from app.reasoning.llm_client import generate

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    result = generate(
        prompt="Balas dengan JSON {\"ok\": true} saja, tanpa penjelasan apapun.",
        json_schema=schema,
        schema_name="ok_check",
        # max_tokens 300, BUKAN nilai kecil spt 50 (cukup utk llama-3.3-70b-versatile lama, model
        # non-reasoning) — model reasoning skrg (mis. openai/gpt-oss-20b default sejak 2026-08-15)
        # menghabiskan sebagian max_tokens utk trace berpikir TERSEMBUNYI sebelum JSON terlihat;
        # budget kecil bikin generation kepotong kosong sebelum JSON sempat ditulis
        # (json_validate_failed, failed_generation="") — dikonfirmasi reproduksi live, BUKAN bug
        # produksi (generator.py/assemble.py pakai default 1024, bakeoff nyata json=100%).
        max_tokens=300,
    )

    assert isinstance(result, dict)
    assert "ok" in result
    assert isinstance(result["ok"], bool)
