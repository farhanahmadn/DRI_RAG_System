"""Cek status SETIAP kunci Groq dan kunci Jina — tool diagnosis operasional, hemat token.

Ditulis setelah insiden 2026-10-15: banyak kunci Groq kena TPD habis / `organization_restricted` dan Jina
403, tapi satu-satunya petunjuknya log panjang di tengah request. Skrip ini menjawab "kunci mana yang
sehat?" tanpa menjalankan pipeline: satu panggilan chat 1 token per kunci Groq, satu embedding pendek
untuk Jina. KUNCI TIDAK PERNAH DICETAK — hanya nomor urut dan 4 karakter terakhir.

Jalankan dari root repo:  python -m scripts.check_api_keys
Butuh: .env berisi GROQ_API_KEY(S), LLM_MODEL, LLM_BASE_URL, JINA_API_KEY.
"""

from __future__ import annotations

import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

from app.reasoning.llm_client import _muat_daftar_kunci  # noqa: E402

_TIMEOUT = 30.0


def _topeng(kunci: str) -> str:
    return "..." + kunci[-4:] if len(kunci) >= 4 else "(pendek)"


def _kode_error(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error", {})
        return f"{err.get('code') or err.get('type') or resp.status_code}: {str(err.get('message'))[:140]}"
    except (ValueError, AttributeError):
        return f"HTTP {resp.status_code}: {resp.text[:140]}"


def _klasifikasi(resp: httpx.Response) -> str:
    """SEHAT / TPD_HABIS / TPM / DIBATASI / KUNCI_DITOLAK / ERROR_LAIN."""
    if resp.status_code == 200:
        return "SEHAT"
    teks = resp.text.lower()
    if resp.status_code == 429:
        return "TPD_HABIS" if "per day" in teks else "TPM_PENUH"
    if "organization_restricted" in teks or "has been restricted" in teks:
        return "DIBATASI"
    if resp.status_code in (401, 403):
        return "KUNCI_DITOLAK"
    return "ERROR_LAIN"


def cek_groq() -> int:
    kunci_list = _muat_daftar_kunci()
    base = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    model = os.getenv("LLM_MODEL")
    print(f"=== Groq: {len(kunci_list)} kunci, model={model!r}, base={base} ===")
    if not kunci_list or not model:
        print("  GROQ_API_KEY(S) atau LLM_MODEL belum diisi.")
        return 0
    sehat = 0
    for i, kunci in enumerate(kunci_list):
        try:
            resp = httpx.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {kunci}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1},
                timeout=_TIMEOUT,
            )
        except httpx.HTTPError as exc:
            print(f"  [{i}] {_topeng(kunci)}  JARINGAN  {type(exc).__name__}: {exc}")
            continue
        status = _klasifikasi(resp)
        sehat += status == "SEHAT"
        sisa = resp.headers.get("x-ratelimit-remaining-tokens")
        detail = f"sisa token menit ini: {sisa}" if status == "SEHAT" and sisa else _kode_error(resp)
        print(f"  [{i}] {_topeng(kunci)}  {status:<14} {detail}")
    print(f"  -> {sehat}/{len(kunci_list)} kunci sehat\n")
    return sehat


def cek_jina() -> bool:
    """Cek tiap key Jina (JINA_API_KEYS atau JINA_API_KEY). True bila ADA key yang sehat."""
    from app.retrieval.jina_keys import muat_kunci_dari_env

    kunci_list = muat_kunci_dari_env()
    print(f"=== Jina (embedding): {len(kunci_list)} kunci, dipakai berurutan (pindah saat habis) ===")
    if not kunci_list:
        print("  JINA_API_KEYS/JINA_API_KEY belum diisi.
")
        return False
    sehat = 0
    for i, kunci in enumerate(kunci_list):
        try:
            resp = httpx.post(
                "https://api.jina.ai/v1/embeddings",
                headers={"Authorization": f"Bearer {kunci}", "Content-Type": "application/json"},
                json={"model": os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3"),
                      "input": ["tes"], "task": "retrieval.query",
                      "dimensions": int(os.getenv("EMBEDDING_DIM", "1024"))},
                timeout=_TIMEOUT,
            )
        except httpx.HTTPError as exc:
            print(f"  [{i}] {_topeng(kunci)}  JARINGAN  {type(exc).__name__}: {exc}")
            continue
        if resp.status_code == 200:
            sehat += 1
            print(f"  [{i}] {_topeng(kunci)}  SEHAT  (embedding berhasil)")
            continue
        pesan = resp.text.lower()
        sebab = "SALDO_HABIS" if "balance" in pesan or resp.status_code == 402 else             "KUNCI_DITOLAK" if resp.status_code in (401, 403) else "ERROR_LAIN"
        print(f"  [{i}] {_topeng(kunci)}  {sebab}  HTTP {resp.status_code}: {_kode_error(resp)}")
        if sebab == "SALDO_HABIS":
            print("      -> isi saldo di https://jina.ai/api-dashboard/key-manager")
    print(f"  -> {sehat}/{len(kunci_list)} kunci sehat
")
    return sehat > 0


def main() -> None:
    sehat_groq = cek_groq()
    jina_ok = cek_jina()
    print("=== RINGKASAN ===")
    print(f"  Groq: {'ADA kunci sehat' if sehat_groq else 'TIDAK ADA kunci sehat'} | "
          f"Jina: {'sehat' if jina_ok else 'BERMASALAH'}")
    print(json.dumps({"groq_sehat": sehat_groq, "jina_sehat": jina_ok}))


if __name__ == "__main__":
    main()
