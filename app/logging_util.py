"""Logging request/response precheck — untuk pembelajaran yang ditunda (CLAUDE.md § Konvensi kode).

"Log setiap input & output (skor, reasoning, sitasi, rekomendasi) sejak awal." Format JSONL,
best-effort (kegagalan logging tidak boleh menggagalkan precheck itu sendiri — lihat pemanggil di
`app/reasoning/assemble.py`).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import JejakAturanRequest, OutputPreCheck

DEFAULT_LOG_PATH = Path("logs/precheck.jsonl")


def log_precheck(
    request: JejakAturanRequest,
    response: OutputPreCheck,
    *,
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> None:
    """Append satu baris JSON (request + response) ke berkas log JSONL."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request": request.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
