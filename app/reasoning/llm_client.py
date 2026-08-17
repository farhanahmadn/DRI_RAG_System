"""Client LLM OpenAI-compatible tipis — satu-satunya tempat provider (Groq/Ollama/vLLM) dikonfigurasi.

Berpindah provider = ganti GROQ_API_KEY/LLM_BASE_URL/LLM_MODEL di .env, kode di sini dan
pemanggilnya TIDAK berubah (CLAUDE.md § Tech stack, LLM serving).

Modul ini murni transport (panggil LLM, paksa JSON valid). Retry bisnis / fallback template saat
guardrail gagal adalah tanggung jawab `app/reasoning/generator.py` + `guardrail.py` (Fase 2), bukan
di sini. `timeout`/`max_retries` di sini adalah hardening TRANSPORT (koneksi macet/5xx sesaat) —
beda lapis dari retry semantik guardrail (regenerasi terarah saat output gagal validasi).
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

from app.reasoning import observability

load_dotenv()

_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))
_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY tidak ditemukan di environment. Salin .env.example ke .env dan isi key."
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=_TIMEOUT_S,
            max_retries=_MAX_RETRIES,
        )
    return _client


def _model() -> str:
    model = os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("LLM_MODEL tidak ditemukan di environment.")
    return model


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
    """
    client = _get_client()
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    mulai = time.perf_counter()
    hasil: dict | None = None
    error_msg: str | None = None

    try:
        try:
            completion = client.chat.completions.create(
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
            completion = client.chat.completions.create(
                model=_model(),
                messages=fallback_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

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
        model_untuk_log = os.getenv("LLM_MODEL") or "unknown"
        observability.catat_generation(schema_name, prompt, hasil, model_untuk_log, latensi, error=error_msg)
