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

from dotenv import dotenv_values, load_dotenv
from openai import BadRequestError, OpenAI, RateLimitError

from app import batas_waktu
from app.reasoning import observability

load_dotenv()
# Override TERBATAS (2026-08-19, ditemukan live VPS) — HANYA key yang modul ini pakai, BUKAN
# load_dotenv(override=True) polos. python-dotenv SECARA DEFAULT tak menimpa env var yang sudah
# ada di proses (mis. dicache PM2 dari shell saat `pm2 start` pertama kali dijalankan) — restart
# proses SAJA tak menjamin .env yang baru diedit benar2 kepakai. TAPI load_dotenv(override=True)
# polos PERNAH DICOBA & DIBATALKAN: itu menimpa SELURUH isi .env sekaligus, termasuk var yang
# SENGAJA diproteksi modul lain saat test (tests/conftest.py memaksa RETRIEVER=mock/
# DATABASE_URL="" — override=True global menimpa balik keduanya begitu llm_client.py di-import,
# suite offline diam-diam coba konek Postgres sungguhan & hang bermenit-menit, ketemu live pas
# Docker lokal kebetulan mati). Override manual per-key di bawah cuma "milik" modul ini.
_ENV_KUNCI_LLM_CLIENT = (
    "GROQ_API_KEY", "GROQ_API_KEYS", "LLM_BASE_URL", "LLM_MODEL", "LLM_TIMEOUT_S", "LLM_MAX_RETRIES",
    "GROQ_RATE_LIMIT_MAKS_TUNGGU_PER_KUNCI", "GROQ_RATE_LIMIT_TUNGGU_MAKS_S", "GROQ_RATE_LIMIT_TUNGGU_DEFAULT_S",
    "GROQ_KARANTINA_TPD_S", "GROQ_KARANTINA_RESTRICTED_S",
)
for _k, _v in dotenv_values().items():
    if _k in _ENV_KUNCI_LLM_CLIENT:
        os.environ[_k] = _v

logger = logging.getLogger(__name__)

_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))

# Karantina kunci (insiden 2026-10-15): kunci yang BARU SAJA terbukti mati tak dicoba lagi sampai
# masa karantina habis. Tanpa ini setiap generate() baru menemukan ulang kunci mati lewat satu
# round-trip yang terbuang, dan `organization_restricted` meledak ke guardrail menghabiskan percobaan.
_KARANTINA_TPD_S = float(os.getenv("GROQ_KARANTINA_TPD_S", "900"))
_KARANTINA_RESTRICTED_S = float(os.getenv("GROQ_KARANTINA_RESTRICTED_S", "3600"))


class LayananTidakTersediaError(RuntimeError):
    """Infrastruktur LLM tak bisa melayani SAMA SEKALI saat ini (semua kunci Groq habis kuota harian
    atau organisasinya dibatasi). Beda dari kegagalan isi: mengulang percobaan tak berguna sampai
    operator memperbaiki akun/kuota. guardrail.py berhenti mencoba begitu menangkap ini."""
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
    lihat app/reasoning/assemble.py).

    2026-08-19 — diubah dari "sticky index global" (semua panggilan pakai kunci yang sama sampai
    kunci itu gagal) jadi ROUND-ROBIN PROAKTIF: setiap `generate()` BARU (bukan retry di
    dalamnya) mulai dari kunci berikutnya scr berurutan. Ditemukan live di VPS: desain lama
    reaktif murni — TPM (limit per-menit) sengaja menunggu+ulang KUNCI YANG SAMA (sesuai spek),
    baru pindah kunci sbg upaya terakhir setelah gagal berkali-kali di kunci itu. Akibatnya di
    traffic sustained, kunci pertama terus-menerus dipakai sampai nyaris limit BERULANG KALI,
    sementara kunci lain menganggur — rotasi cuma cadangan darurat, bukan menyebar beban.
    Round-robin proaktif menyebar beban ke SEMUA kunci sejak awal (total TPM efektif ~jumlah_kunci
    x limit per kunci di bawah traffic sustained), TPM-tunggu/TPD-lompat di DALAM satu panggilan
    tetap sama persis (lihat generate() di bawah) — cuma titik AWAL yang berubah, bukan
    perilaku saat rate limit terjadi.
    """

    def __init__(self, keys: list[str], base_url: str | None):
        if not keys:
            raise RuntimeError(
                "GROQ_API_KEY/GROQ_API_KEYS tidak ditemukan di environment. "
                "Salin .env.example ke .env dan isi minimal satu key."
            )
        self._keys = keys
        self._base_url = base_url
        self._counter = 0
        self._lock = threading.Lock()
        self._clients: dict[int, OpenAI] = {}
        self._karantina: dict[int, float] = {}  # index kunci -> waktu monotonic karantina berakhir

    def jumlah_kunci(self) -> int:
        return len(self._keys)

    def _tersedia_nolock(self, idx: int) -> bool:
        sampai = self._karantina.get(idx)
        if sampai is None:
            return True
        if time.monotonic() >= sampai:
            del self._karantina[idx]  # masa karantina habis -> boleh dicoba lagi
            return True
        return False

    def karantina(self, idx: int, detik: float, alasan: str) -> None:
        """Tandai kunci mati sementara. Yang dicatat HANYA nomor urut, tidak pernah isi kunci."""
        with self._lock:
            baru = idx not in self._karantina
            self._karantina[idx % len(self._keys)] = time.monotonic() + detik
        if baru:
            logger.warning("Kunci Groq index %d dikarantina %.0f detik (alasan: %s)", idx, detik, alasan)

    def jumlah_tersedia(self) -> int:
        with self._lock:
            return sum(1 for i in range(len(self._keys)) if self._tersedia_nolock(i))

    def kunci_berikutnya(self, idx: int) -> int | None:
        """Kunci tersedia berikutnya SESUDAH `idx` (melingkar, tidak termasuk `idx`), atau None."""
        n = len(self._keys)
        with self._lock:
            for langkah in range(1, n):
                kandidat = (idx + langkah) % n
                if self._tersedia_nolock(kandidat):
                    return kandidat
        return None

    def kunci_awal(self) -> int:
        """Index kunci AWAL utk satu panggilan generate() baru — round-robin (increment atomik),
        BUKAN selalu kunci pertama. Kunci yang sedang dikarantina dilewati; kalau SEMUA mati, langsung
        gagal dgn pesan jelas (tanpa membuang satu round-trip pun)."""
        with self._lock:
            n = len(self._keys)
            for _ in range(n):
                idx = self._counter % n
                self._counter += 1
                if self._tersedia_nolock(idx):
                    return idx
        raise LayananTidakTersediaError(
            f"Semua {len(self._keys)} kunci Groq sedang dikarantina (kuota harian habis atau organisasi "
            "dibatasi). Periksa akun Groq (python -m scripts.check_api_keys)."
        )

    def client_utk(self, idx: int) -> OpenAI:
        idx = idx % len(self._keys)
        with self._lock:
            client = self._clients.get(idx)
            if client is None:
                client = OpenAI(
                    api_key=self._keys[idx],
                    base_url=self._base_url,
                    timeout=_TIMEOUT_S,
                    max_retries=_MAX_RETRIES,
                )
                self._clients[idx] = client
            return client


_rotator: _RotasiKunciGroq | None = None


def _get_rotator() -> _RotasiKunciGroq:
    global _rotator
    if _rotator is None:
        _rotator = _RotasiKunciGroq(_muat_daftar_kunci(), os.getenv("LLM_BASE_URL"))
    return _rotator


def _get_client() -> OpenAI:
    """Kompat mundur — satu client (kunci berikutnya scr round-robin). Dipertahankan utk caller
    lama & test yang tak perlu tahu soal rotasi sama sekali."""
    rotator = _get_rotator()
    return rotator.client_utk(rotator.kunci_awal())


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


def _org_dibatasi(exc: Exception) -> bool:
    """Groq 400 `organization_restricted`: akun/organisasi kunci ini dibatasi. Masalah kunci, bukan isi."""
    pesan = str(exc).lower()
    return "organization_restricted" in pesan or "has been restricted" in pesan


# Sisa anggaran minimum (detik) agar satu panggilan LLM masih layak dimulai.
_MIN_SISA_PANGGILAN_S = 5.0


def _batasi_client(client: OpenAI) -> OpenAI:
    """Dengan anggaran waktu aktif: timeout per panggilan dibatasi sisa anggaran dan retry internal SDK
    dimatikan (retry semantik ditangani guardrail, dan tiap retry SDK memakan waktu yang tak terhitung).
    Tanpa anggaran aktif, client dikembalikan apa adanya."""
    sisa = batas_waktu.sisa()
    if sisa is None:
        return client
    batas_waktu.periksa(_MIN_SISA_PANGGILAN_S, "panggilan LLM")
    return client.with_options(timeout=min(_TIMEOUT_S, sisa - 1.0), max_retries=0)


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
    except BadRequestError as exc:
        if _org_dibatasi(exc):
            raise  # fallback json_object akan gagal identik; hemat satu round-trip
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


# Model (gpt-oss) kadang memakai varian Unicode yg tampak sama dgn tanda hubung/spasi biasa
# (APP-2026-0095: "sub-zona P-1" bertanda hubung U+2011). Visual identik, tapi pencarian teks/salin-tempel di FE
# tak menemukannya. Dinormalisasi ke ASCII di satu pintu keluaran LLM, sebelum guardrail.
_PETA_KARAKTER = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-",
    "\u00a0": " ", "\u202f": " ", "\u2007": " ", "\u2009": " ",
})


# Label enum teknis yang kadang disalin LLM ke narasi (APP-2026-UJI: "Intensitas MELAMPAUI_BATAS"),
# dan salah-ekspansi "RTH" -> "ruang terbuka terbuka". Diganti di satu pintu keluaran LLM.
_PETA_LABEL_TEKNIS = (
    (re.compile(r"\bMELAMPAUI_BATAS\b"), "melampaui batas"),
    (re.compile(r"\bMEMENUHI_SYARAT\b"), "memenuhi syarat"),
    (re.compile(r"\bTIDAK_DIEVALUASI\b"), "tidak dievaluasi"),
    (re.compile(r"ruang\s+terbuka\s+terbuka", re.IGNORECASE), "ruang terbuka hijau"),
)


def _normalisasi_karakter(nilai):
    if isinstance(nilai, str):
        # Model kadang menulis dua karakter backslash+n alih-alih baris baru di dalam string JSON
        # (APP-2026-UJI-4: "...turun ke 60%.\\n2. Menambah RTH" tampil mentah di saran/narasi).
        teks = nilai.translate(_PETA_KARAKTER).replace("\\n", "\n")
        for pola, pengganti in _PETA_LABEL_TEKNIS:
            teks = pola.sub(pengganti, teks)
        return teks
    if isinstance(nilai, list):
        return [_normalisasi_karakter(v) for v in nilai]
    if isinstance(nilai, dict):
        return {k: _normalisasi_karakter(v) for k, v in nilai.items()}
    return nilai


_MAKS_TOKEN_EKSKALASI = int(os.getenv("LLM_MAX_TOKENS_EKSKALASI", "8192"))


def _keluaran_kosong_token_habis(exc: BadRequestError) -> bool:
    """Groq 400 `json_validate_failed` dgn `failed_generation` KOSONG: model reasoning menghabiskan
    seluruh max_tokens utk berpikir sebelum menulis satu karakter JSON pun. Beda dari JSON rusak
    (failed_generation berisi teks). Mengulang dgn max_tokens SAMA akan gagal identik — log VPS
    2026-10-14: itbx multi-zona gagal 3/3 persis begitu."""
    pesan = str(exc)
    return "json_validate_failed" in pesan and "'failed_generation': ''" in pesan


def _generate_tanpa_antre(
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
        # Round-robin: titik AWAL kunci berbeda tiap panggilan generate() (lihat kunci_awal()) —
        # menyebar beban dasar ke semua kunci. Rate limit DI DALAM satu panggilan ini masih
        # ditangani reaktif spt sebelumnya: TPM/RPM -> tunggu+ulang KUNCI SAMA; TPD/RPD -> lompat
        # SEGERA ke kunci berikutnya (idx lokal, tak perlu koordinasi lintas-thread — tiap
        # panggilan generate() punya jalur/kunci sendiri sejak kunci_awal()).
        idx = rotator.kunci_awal()
        tunggu_di_kunci_ini = 0
        kunci_berbeda_dicoba = 1
        sudah_eskalasi_token = False
        while True:
            client = _batasi_client(rotator.client_utk(idx))
            try:
                completion = _panggil_llm_sekali(client, messages, json_schema, schema_name, temperature, max_tokens)
                break
            except BadRequestError as exc:
                if _org_dibatasi(exc):
                    rotator.karantina(idx, _KARANTINA_RESTRICTED_S, "organization_restricted")
                    berikutnya = rotator.kunci_berikutnya(idx)
                    if berikutnya is None or kunci_berbeda_dicoba >= rotator.jumlah_kunci():
                        raise LayananTidakTersediaError(
                            "Tak ada kunci Groq yang bisa dipakai: organisasi kunci dibatasi "
                            "(organization_restricted) atau kuota habis. Hubungi support Groq / ganti kunci."
                        ) from exc
                    logger.warning("Rotasi GROQ_API_KEY: index %d -> %d (alasan: organization_restricted)",
                                   idx, berikutnya)
                    idx = berikutnya
                    kunci_berbeda_dicoba += 1
                    tunggu_di_kunci_ini = 0
                    continue
                if sudah_eskalasi_token or max_tokens >= _MAKS_TOKEN_EKSKALASI or not _keluaran_kosong_token_habis(exc):
                    raise
                sudah_eskalasi_token = True
                logger.warning(
                    "Keluaran LLM kosong (kemungkinan token habis utk reasoning) — ulang dgn max_tokens %d -> %d",
                    max_tokens, _MAKS_TOKEN_EKSKALASI,
                )
                max_tokens = _MAKS_TOKEN_EKSKALASI
                continue
            except RateLimitError as exc:
                jenis = _klasifikasi_rate_limit(str(exc))
                pindah_kunci = False
                if jenis == "tpd":
                    # Kuota harian tak akan reset dlm sesi ini -> lompat segera, tanpa nunggu.
                    rotator.karantina(idx, _KARANTINA_TPD_S, "tpd")
                    pindah_kunci = True
                else:
                    # TPM/RPM (atau format tak dikenali, diperlakukan spt TPM demi aman) -> tunggu
                    # dulu di kunci yang sama; pindah kunci HANYA sbg upaya terakhir kalau sudah
                    # nunggu berkali-kali tanpa membaik.
                    tunggu_di_kunci_ini += 1
                    if tunggu_di_kunci_ini > _MAKS_TUNGGU_PER_KUNCI:
                        pindah_kunci = True
                    else:
                        detik = _ekstrak_tunggu_detik(str(exc))
                        logger.warning(
                            "Rate limit TPM/RPM (kunci index %d) — tunggu %.1fs lalu ulang (percobaan %d/%d): %s",
                            idx, detik, tunggu_di_kunci_ini, _MAKS_TUNGGU_PER_KUNCI, exc,
                        )
                        batas_waktu.periksa(detik + _MIN_SISA_PANGGILAN_S, "menunggu rate limit Groq")
                        time.sleep(detik)
                        continue

                if pindah_kunci:
                    berikutnya = rotator.kunci_berikutnya(idx)
                    if kunci_berbeda_dicoba >= rotator.jumlah_kunci() or berikutnya is None:
                        # Semua kunci yang ada sudah dicoba/dikarantina & masih limit -> tak ada lagi
                        # yg bisa dirotasi, menyerah (guardrail.py akan menangkap ini spt biasa).
                        raise
                    idx_lama = idx
                    idx = berikutnya
                    kunci_berbeda_dicoba += 1
                    tunggu_di_kunci_ini = 0
                    logger.warning(
                        "Rotasi GROQ_API_KEY: index %d -> %d (kunci ke-%d/%d dicoba, alasan: %s)",
                        idx_lama, idx, kunci_berbeda_dicoba, rotator.jumlah_kunci(), jenis,
                    )
                    continue

        model_terpakai = _model()
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("LLM mengembalikan konten kosong.")
        try:
            hasil = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM tidak mengembalikan JSON valid: {content!r}") from exc
        hasil = _normalisasi_karakter(hasil)

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


# Antrean panggilan LLM per-poin: tiga poin yang berangkat bersamaan membawa >10K token dalam hitungan
# detik, padahal Groq free tier membatasi 8K token/menit per ORGANISASI (hasil uji nyata 2026-10-16:
# ratusan 429). Default 1 = poin diproses satu per satu (retrieval tetap paralel). Naikkan lewat
# LLM_MAX_CONCURRENT bila akun berbayar. Sintesis (narasi/kesimpulan) tidak diantre: ia baru jalan setelah
# semua poin selesai dan kecil.
_SKEMA_SINTESIS = frozenset({"kesimpulan", "narasi_rekomendasi"})
_antrean_llm = threading.BoundedSemaphore(max(1, int(os.getenv("LLM_MAX_CONCURRENT", "1"))))


def generate(prompt: str, json_schema: dict, **kwargs) -> dict:
    """Pintu masuk panggilan LLM: antre untuk poin, langsung untuk sintesis. Lihat _generate_tanpa_antre."""
    if kwargs.get("schema_name") in _SKEMA_SINTESIS:
        return _generate_tanpa_antre(prompt, json_schema, **kwargs)
    with _antrean_llm:
        # Waktu antre ikut anggaran permintaan: jangan mulai panggilan bila sisa waktu sudah habis.
        batas_waktu.periksa(_MIN_SISA_PANGGILAN_S, "panggilan LLM")
        return _generate_tanpa_antre(prompt, json_schema, **kwargs)
