"""Client LLM OpenAI-compatible tipis — satu-satunya tempat provider (Groq/Ollama/vLLM) dikonfigurasi.

Berpindah provider = ganti GROQ_API_KEY/LLM_BASE_URL/LLM_MODEL di .env, kode di sini dan
pemanggilnya TIDAK berubah (CLAUDE.md § Tech stack, LLM serving).

Modul ini murni transport (panggil LLM, paksa JSON valid). Retry bisnis / fallback template saat
guardrail gagal adalah tanggung jawab `app/reasoning/generator.py` + `guardrail.py` (Fase 2), bukan
di sini.
"""

import json
import os

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()

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
        _client = OpenAI(api_key=api_key, base_url=base_url)
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
    max_tokens: int = 1024,
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
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM tidak mengembalikan JSON valid: {content!r}") from exc
