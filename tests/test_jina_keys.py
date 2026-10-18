"""Failover berurutan API key Jina: satu key dipakai sampai habis, baru pindah (bukan round-robin)."""

import httpx
import pytest

from app.retrieval import _provider_http as http
from app.retrieval.jina_keys import KumpulanKunciJina, SemuaKunciJinaHabisError, kunci_habis, get_kumpulan


class TestKunciHabis:
    @pytest.mark.parametrize("kode", [401, 402, 403])
    def test_status_penolakan_dianggap_habis(self, kode):
        assert kunci_habis(kode, "")

    def test_429_saldo_habis_tapi_429_biasa_tidak(self):
        assert kunci_habis(429, "Insufficient balance for this request")
        assert not kunci_habis(429, "Too many requests, slow down")
        assert not kunci_habis(500, "balance") and not kunci_habis(200, "")


class TestKumpulan:
    def test_sticky_bukan_round_robin(self):
        k = KumpulanKunciJina(["a", "b", "c"])
        assert [k.aktif()[0] for _ in range(5)] == [0, 0, 0, 0, 0]

    def test_pindah_hanya_saat_habis_dan_tidak_kembali(self):
        k = KumpulanKunciJina(["a", "b", "c"])
        k.tandai_habis(0, "uji")
        assert k.aktif() == (1, "b") and k.aktif() == (1, "b")
        k.tandai_habis(1, "uji")
        assert k.aktif() == (2, "c")

    def test_semua_habis_raise(self):
        k = KumpulanKunciJina(["a", "b"])
        k.tandai_habis(0, "x"); k.tandai_habis(1, "x")
        with pytest.raises(SemuaKunciJinaHabisError):
            k.aktif()

    def test_key_habis_pulih_setelah_cooldown(self):
        k = KumpulanKunciJina(["a", "b"], cooldown_s=0.0)
        k.tandai_habis(0, "x")
        assert k.aktif()[0] == 0  # cooldown 0: langsung boleh dicoba lagi

    def test_muat_dari_env(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEYS", " k1 , k2,,k3 ")
        monkeypatch.setenv("JINA_API_KEY", "lama")
        assert get_kumpulan().jumlah() == 3
        monkeypatch.delenv("JINA_API_KEYS")
        assert get_kumpulan().jumlah() == 1


class _Resp:
    def __init__(self, status, teks="", data=None):
        self.status_code, self.text, self._data = status, teks, data or {"ok": True}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=httpx.Request("POST", "http://x"),
                                        response=httpx.Response(self.status_code))

    def json(self):
        return self._data


class _Client:
    dipakai: list = []
    skenario: dict = {}

    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False

    def post(self, url, headers=None, json=None):
        tok = headers["Authorization"].split()[-1]
        _Client.dipakai.append(tok)
        return _Resp(*_Client.skenario.get(tok, (200,)))


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    _Client.dipakai, _Client.skenario = [], {}
    monkeypatch.setattr(httpx, "Client", _Client)
    http._state.clear()


def _kirim(k):
    return http.post_json("jina-uji", "http://x", headers={"Content-Type": "application/json"},
                          json_body={}, timeout_s=1, max_retries=0, kunci=k)


class TestPostJsonFailover:
    def test_pindah_ke_key_berikutnya_saat_saldo_habis_tanpa_retry(self):
        k = KumpulanKunciJina(["a", "b", "c"])
        _Client.skenario = {"a": (402, "Payment required")}
        assert _kirim(k) == {"ok": True}
        assert _Client.dipakai == ["a", "b"]
        _Client.dipakai.clear()
        _kirim(k)
        assert _Client.dipakai == ["b"]  # tidak kembali ke key a

    def test_key_ditolak_401_juga_pindah(self):
        k = KumpulanKunciJina(["a", "b"])
        _Client.skenario = {"a": (401, "bad key")}
        _kirim(k)
        assert _Client.dipakai == ["a", "b"]

    def test_semua_habis_gagal_jelas_dan_tidak_membuka_circuit(self):
        k = KumpulanKunciJina(["a", "b"])
        _Client.skenario = {"a": (402, ""), "b": (402, "")}
        with pytest.raises(SemuaKunciJinaHabisError):
            _kirim(k)
        assert http._state.get("jina-uji", {}).get("failures", 0) == 0

    def test_429_laju_biasa_bukan_habis(self):
        k = KumpulanKunciJina(["a", "b"])
        _Client.skenario = {"a": (429, "Too many requests")}
        with pytest.raises(RuntimeError):
            _kirim(k)  # max_retries=0: gagal biasa, key a TIDAK ditandai habis
        assert k.aktif()[0] == 0
