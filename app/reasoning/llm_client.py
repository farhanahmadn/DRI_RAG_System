"""Client LLM OpenAI-compatible tipis — satu-satunya tempat provider (Groq/Ollama/vLLM) dikonfigurasi.

Berpindah provider = ganti GROQ_API_KEY/LLM_BASE_URL/LLM_MODEL di .env, kode di sini dan
pemanggilnya TIDAK berubah (CLAUDE.md § Tech stack, LLM serving).

Modul ini murni transport (panggil LLM, paksa JSON valid, rotasi API key saat rate limit). Retry
bisnis / fallback template saat guardrail gagal adalah tanggung jawab `app/reasoning/generator.py`
+ `guardrail.py` (Fase 2), bukan di sini. `timeout`/`max_retries` di sini adalah hardening
TRANSPORT (koneksi macet/5xx sesaat) — beda lapis dari retry semantik guardrail (regenerasi
terarah saat output gagal validasi).

Rotasi multi-API-key (2026-08-18): sistem MASIH tahap demo, Groq free-tier ("on_demand") — TPM
cuma 8000, gampang habis begitu 3 poin digenerate BERSAMAAN (ThreadPoolExecutor). Ditemukan live:
1 request penuh (3 poin) sendirian sudah bisa >12rb token/menit dibutuhkan. Sambil menunggu upgrade
tier/langganan (keputusan bisnis, bukan kode), GROQ_API_KEYS (jamak) memungkinkan beberapa akun
demo bergantian menyerap beban — BUKAN solusi produksi jangka panjang, cuma jembatan sampai
langganan resmi (lihat docs/STATUS.md).
"""

import json
import logging
import os
import re
import threading
import time

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI, RateLimitError

from app.reasoning import observability

load_dotenv()

logger = logging.getLogger(__name__)

_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))
_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# Rotasi rate-limit — SEMUA opsional, default masuk akal (pola sama spt LLM_TIMEOUT_S/dst).
# Batas berapa kali NUNGGU pakai kunci yang SAMA (utk TPM/RPM, limit per-menit — biasanya reset
# dlm hitungan detik/menit) sebelum menyerah & mencoba rotasi kunci sbg upaya terakhir.
_MAKS_TUNGGU_PER_KUNCI = int(os.getenv("GROQ_RATE_LIMIT_MAKS_TUNGGU_PER_KUNCI", "2"))
# Batas atas detik tidur SEKALI nunggu, apapun saran "try again in Ns" dari Groq — cegah blokir
# terlalu lama kalau saran Groq ternyata sangat besar (mis. TPD yg salah terdeteksi sbg TPM).
_TUNGGU_MAKS_S = float(os.getenv("GROQ_RATE_LIMIT_TUNGGU_MAKS_S", "60"))
# Dipakai kalau pesan error tak menyebutkan "try again in Ns" sama sekali (jaga-jaga format Groq
# berubah) — angka kasar, bukan hasil ukur presisi.
_TUNGGU_DEFAULT_S = float(os.getenv("GROQ_RATE_LIMIT_TUNGGU_DEFAULT_S", "10"))

# Pola pesan RateLimitError Groq, mis.:
#   "...on tokens per minute (TPM): Limit 8000... Please try again in 34.019999999s..."
#   "...on tokens per day (TPD): Limit 200000..."
# "per minute"/"per day" dicek scr teks (BUKAN cuma singkatan TPM/RPM/TPD/RPD) supaya tahan kalau
# Groq ganti singkatan tapi frasa "per minute"/"per day" tetap konsisten.
_RE_PER_MENIT = re.compile(r"per minute", re.IGNORECASE)
_RE_PER_HARI = re.compile(r"per day", re.IGNORECASE)
_RE_COBA_LAGI_DETIK = re.compile(r"try again in ([\d.]+)\s*s", re.IGNORECASE)


def _muat_daftar_kunci() -> list[str]:
    """GROQ_API_KEYS (comma-separated, utk rotasi) diprioritaskan; fallback ke GROQ_API_KEY
    tunggal (perilaku lama, tetap didukung penuh — array isi 1 elemen secara efektif)."""
    jamak = os.getenv("GROQ_API_KEYS")
    if jamak:
        kunci = [k.strip() for k in jamak.split(",") if k.strip()]
        if kunci:
            return kunci
    tunggal = os.getenv("GROQ_API_KEY")
    return [tunggal] if tunggal else []


class _RotasiKunciGroq:
    """Rotasi multi-API-key Groq, thread-safe (poin digenerate paralel via ThreadPoolExecutor,
    lihat app/reasoning/assemble.py). Index & cache client per-kunci dilindungi satu lock —
    operasi di dalamnya murni in-memory (bangun objek OpenAI klien), bukan I/O, jadi lock singkat.
    """

    def __init__(self, keys: list[str], base_url: str | None):
        if not keys:
            raise RuntimeError(
                "GROQ_API_KEY/GROQ_API_KEYS tidak ditemukan di environment. "
                "Salin .env.example ke .env dan isi minimal satu key."
            )
        self._keys = keys
        self._base_url = base_url
        self._idx = 0
        self._lock = threading.Lock()
        self._clients: dict[int, OpenAI] = {}

    def jumlah_kunci(self) -> int:
        return len(self._keys)

    def client_aktif(self) -> tuple[int, OpenAI]:
        with self._lock:
            idx = self._idx
            client = self._clients.get(idx)
            if client is None:
                client = OpenAI(
                    api_key=self._keys[idx],
                    base_url=self._base_url,
                    timeout=_TIMEOUT_S,
                    max_retries=_MAX_RETRIES,
                )
                self._clients[idx] = client
            return idx, client

    def rotasi_dari(self, idx_lama: int) -> int:
        """Pindah ke kunci berikutnya HANYA kalau index global belum berubah sejak `idx_lama`
        dibaca caller (cegah 2 thread yang gagal bersamaan merotasi 2x berturut-turut utk 1
        insiden yang sama). Return index aktif SETELAH panggilan ini (baru atau tetap lama kalau
        thread lain sudah duluan merotasi)."""
        with self._lock:
            if self._idx == idx_lama:
                self._idx = (self._idx + 1) % len(self._keys)
                logger.warning(
                    "Rotasi GROQ_API_KEY: index %d -> %d (dari %d kunci total, alasan: rate limit TPD/RPD)",
                    idx_lama, self._idx, len(self._keys),
                )
            return self._idx


_rotator: _RotasiKunciGroq | None = None


def _get_rotator() -> _RotasiKunciGroq:
    global _rotator
    if _rotator is None:
        _rotator = _RotasiKunciGroq(_muat_daftar_kunci(), os.getenv("LLM_BASE_URL"))
    return _rotator


def _get_client() -> OpenAI:
    """Kompat mundur — client utk kunci AKTIF saat ini (index 0 kalau belum pernah rotasi).
    Dipertahankan utk caller lama & test yang tak perlu tahu soal rotasi sama sekali."""
    _, client = _get_rotator().client_aktif()
    return client


def _model() -> str:
    model = os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("LLM_MODEL tidak ditemukan di environment.")
    return model


def _klasifikasi_rate_limit(pesan: str) -> str:
    """"tpd" (per hari -> rotasi kunci) / "tpm" (per menit -> tunggu) / "tak_diketahui" (format
    pesan tak dikenali -> diperlakukan spt tpm, lebih aman menunggu drpd rotasi tanpa dasar)."""
    per_hari = bool(_RE_PER_HARI.search(pesan))
    per_menit = bool(_RE_PER_MENIT.search(pesan))
    if per_hari and not per_menit:
        return "tpd"
    if per_menit:
        return "tpm"
    return "tak_diketahui"


def _ekstrak_tunggu_detik(pesan: str) -> float:
    m = _RE_COBA_LAGI_DETIK.search(pesan)
    if not m:
        return _TUNGGU_DEFAULT_S
    try:
        return min(float(m.group(1)), _TUNGGU_MAKS_S)
    except ValueError:
        return _TUNGGU_DEFAULT_S


def _panggil_llm_sekali(client: OpenAI, messages: list[dict], json_schema: dict, schema_name: str,
                         temperature: float, max_tokens: int):
    """Satu percobaan panggilan (strict json_schema, fallback json_object kalau ditolak) — TIDAK
    ADA retry/rotasi di sini, murni satu request. Dipisah dari `generate()` supaya loop rotasi di
    `generate()` tetap simpel (ulang panggil fungsi ini, bukan duplikasi logic strict/fallback)."""
    try:
        return client.chat.completions.create(
            model=_model(),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema, "strict": True},
            },
        )
    except BadRequestError:
        fallback_messages = list(messages)
        fallback_messages.append(
            {
                "role": "user",
                "content": (
                    "Balas HANYA dengan JSON valid yang mengikuti skema berikut, tanpa teks lain:\n"
                    f"{json.dumps(json_schema, ensure_ascii=False)}"
                ),
            }
        )
        return client.chat.completions.create(
            model=_model(),
            messages=fallback_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )


def generate(
    prompt: str,
    json_schema: dict,
    *,
    schema_name: str = "response",
    system: str | None = None,
    temperature: float = 0.0,
    # 2048 (bukan 1024) sejak migrasi ke openai/gpt-oss-20b (2026-08-15) — model REASONING,
    # menghabiskan sebagian max_tokens utk trace berpikir tersembunyi SEBELUM JSON terlihat. 1024
    # terbukti live kehabisan di tengah jalan utk prompt lebih besar (mis. poin intensitas):
    # 'max completion tokens reached before generating a valid document' -> json_validate_failed
    # -> retry habis -> low_confidence, padahal bukan soal kualitas model.
    max_tokens: int = 2048,
) -> dict:
    """Panggil LLM dan kembalikan JSON valid (dict) sesuai `json_schema`.

    Coba structured output (`response_format=json_schema`, strict) dulu. Kalau provider/model
    menolak mode ini, fallback sekali ke `json_object` dengan schema disisipkan ke prompt, supaya
    tetap portable ke model yang belum dukung strict json_schema.

    Rate limit (RateLimitError) ditangani DI SINI (transport), terpisah dari retry semantik
    guardrail: limit per-MENIT (TPM/RPM) -> tunggu sesuai saran Groq lalu ulang KUNCI YANG SAMA;
    limit per-HARI (TPD/RPD) -> rotasi ke kunci berikutnya (GROQ_API_KEYS) SEGERA, tanpa nunggu
    (kuota harian tak akan reset dlm sesi ini). Exception lain (BadRequestError sesudah fallback,
    timeout, dst) diteruskan apa adanya ke pemanggil (generator.py/guardrail.py) spt biasa.
    """
    rotator = _get_rotator()
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    mulai = time.perf_counter()
    hasil: dict | None = None
    error_msg: str | None = None
    model_terpakai = "unknown"

    try:
        tunggu_di_kunci_ini = 0
        kunci_dicoba = set()
        while True:
            idx, client = rotator.client_aktif()
            kunci_dicoba.add(idx)
            try:
                completion = _panggil_llm_sekali(client, messages, json_schema, schema_name, temperature, max_tokens)
                break
            except RateLimitError as exc:
                jenis = _klasifikasi_rate_limit(str(exc))
                if jenis == "tpd":
                    idx_baru = rotator.rotasi_dari(idx)
                    tunggu_di_kunci_ini = 0
                    if len(kunci_dicoba) >= rotator.jumlah_kunci() and idx_baru in kunci_dicoba:
                        # Semua kunci yang ada sudah dicoba & masih limit -> tak ada lagi yg bisa
                        # dirotasi, menyerah (guardrail.py akan menangkap ini spt exception biasa).
                        raise
                    continue
                # TPM/RPM (atau format tak dikenali, diperlakukan spt TPM demi aman) -> tunggu.
                tunggu_di_kunci_ini += 1
                if tunggu_di_kunci_ini > _MAKS_TUNGGU_PER_KUNCI:
                    # Sudah nunggu berkali-kali di kunci ini tanpa membaik -> coba kunci lain
                    # sbg upaya terakhir (barangkali kunci lain memang lebih longgar).
                    idx_baru = rotator.rotasi_dari(idx)
                    tunggu_di_kunci_ini = 0
                    if len(kunci_dicoba) >= rotator.jumlah_kunci() and idx_baru in kunci_dicoba:
                        raise
                    continue
                detik = _ekstrak_tunggu_detik(str(exc))
                logger.warning(
                    "Rate limit TPM/RPM (kunci index %d) — tunggu %.1fs lalu ulang (percobaan %d/%d): %s",
                    idx, detik, tunggu_di_kunci_ini, _MAKS_TUNGGU_PER_KUNCI, exc,
                )
                time.sleep(detik)
                continue

        model_terpakai = _model()
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("LLM mengembalikan konten kosong.")
        try:
            hasil = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM tidak mengembalikan JSON valid: {content!r}") from exc

        return hasil
    except Exception as exc:
        error_msg = str(exc)
        raise
    finally:
        latensi = time.perf_counter() - mulai
        # os.getenv langsung (bukan _model()) — finally tidak boleh raise baru yang menutupi
        # exception asli kalau LLM_MODEL entah bagaimana hilang di tengah jalan.
        model_untuk_log = model_terpakai if model_terpakai != "unknown" else (os.getenv("LLM_MODEL") or "unknown")
        observability.catat_generation(schema_name, prompt, hasil, model_untuk_log, latensi, error=error_msg)
