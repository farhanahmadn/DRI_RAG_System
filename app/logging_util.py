"""Logging request/response precheck — untuk pembelajaran yang ditunda (CLAUDE.md § Konvensi kode).

"Log setiap input & output (skor, reasoning, sitasi, rekomendasi) sejak awal." Format JSONL,
best-effort (kegagalan logging tidak boleh menggagalkan precheck itu sendiri — lihat pemanggil di
`app/reasoning/assemble.py`).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import L2Assessment, OutputL3

DEFAULT_LOG_PATH = Path("logs/precheck.jsonl")


def log_precheck(
    request: L2Assessment,
    response: OutputL3,
    *,
    diagnostik: list[dict] | None = None,
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> None:
    """Append satu baris JSON (request + response [+ diagnostik]) ke berkas log JSONL.

    `diagnostik` = satu entri per poin (lihat `app/reasoning/guardrail.py::DiagnosaPoin`): berapa
    kali dicoba, berapa chunk pendukung yang didapat, temuan guardrail terakhir, exception LLM
    terakhir, dan label `sebab`. Ditulis DI SINI, bukan di `response`, karena ini data operasional
    — `OutputL3` adalah kontrak dgn back-end/reviewer dan tidak boleh membengkak oleh diagnostik.

    Kenapa perlu (2026-09-07): 51% permohonan nyata punya minimal satu poin `low_confidence`, tapi
    log lama cuma menyimpan request+response akhir sehingga "guardrail menolak", "LLM kena rate
    limit", dan "retrieval kosong" tak terbedakan — tak bisa diperbaiki secara terarah.
    Opsional (default None) supaya pemanggil lama & test tetap jalan tanpa perubahan.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request": request.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
    }
    if diagnostik is not None:
        record["diagnostik"] = diagnostik

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
