"""Ekspor skema OpenAPI FastAPI ke openapi.json (root repo) — utk editor.swagger.io / Postman
tanpa perlu server jalan terus-menerus (lihat docs/INTEGRASI_BACKEND.md).

Jalankan: `python scripts/export_openapi.py` dari root repo.
"""

import json
from pathlib import Path

from app.api.main import app

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "openapi.json"


def main() -> None:
    skema = app.openapi()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(skema, f, indent=2, ensure_ascii=False)

    print(f"OpenAPI schema ditulis ke {OUTPUT_PATH}")
    print(f"Jumlah paths: {len(skema.get('paths', {}))}")


if __name__ == "__main__":
    main()
