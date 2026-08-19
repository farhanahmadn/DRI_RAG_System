"""app/ingest/parse.py — PDF -> markdown (LlamaParse / Docling), page-aware & versioned.

Bagian dari pipeline RAG (wilayah tim RAG). Langkah 3 di rencana: parse dokumen sumber
(RDTR 390 hlm + UU/Permen/RTRW) jadi markdown dengan TABEL UTUH, lalu simpan ke
`data/parsed/<versi>/` (di-version, tidak di-commit isi mentahnya bila besar — lihat .gitignore).

KENAPA page-aware: `halaman` adalah syarat sitasi (SitasiOutput.halaman). Tiap halaman ditandai
`<!-- PAGE n -->` supaya chunker (Langkah 5) bisa menautkan nomor halaman yang AKURAT ke tiap chunk.

Dua backend (bandingkan dulu di halaman matriks tersulit sebelum bakar 390 hlm — lihat rencana):
  * llamaparse : cloud, kualitas tabel padat terbaik. Butuh LLAMA_CLOUD_API_KEY di .env.
  * docling    : lokal/gratis/privasi, fallback offline.

CLI:
  # Bake-off: parse HANYA halaman tabel tersulit dengan kedua backend, lalu bandingkan visual.
  python -m app.ingest.parse --backend llamaparse --input data/raw/rdtr.pdf --pages 108-115 --out data/parsed/bakeoff
  python -m app.ingest.parse --backend docling    --input data/raw/rdtr.pdf --pages 108-115 --out data/parsed/bakeoff

  # Parse penuh setelah backend dikunci (input boleh file tunggal atau folder berisi *.pdf):
  python -m app.ingest.parse --backend llamaparse --input data/raw --out data/parsed/v1 --mode agentic

Prasyarat:  pip install -e ".[rag,parse]"   (llama-cloud-services + docling)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:  # .env di root repo (LLAMA_CLOUD_API_KEY, dst) — BUKAN override=True, skrip CLI manual
    # (bukan proses long-running PM2), tak butuh proteksi itu (lihat app/reasoning/llm_client.py
    # utk kasus yg benar2 butuh, dgn override TERBATAS per key)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv opsional
    pass


PAGE_MARKER = "<!-- PAGE {n} -->"

# Peta mode ramah -> ParsingMode LlamaParse (diverifikasi dari enum SDK llama-cloud-services).
# Kredit/halaman (perkiraan): without_llm=1, with_llm=3, with_agent=10, document_with_agent=45.
_LLAMA_MODE = {
    "fast": "parse_page_without_llm",
    "balanced": "parse_page_with_llm",
    "agentic": "parse_page_with_agent",          # DEFAULT: terbaik utk matriks ITBX / KDB-KLB padat
    "agentic_plus": "parse_document_with_agent",
}


@dataclass
class PageMd:
    """Satu halaman hasil parse."""

    page: int
    md: str


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------
def parse_pages_arg(pages: str | None) -> list[int] | None:
    """'108-115,120' -> [108,...,115,120] (1-indexed, sesuai nomor halaman manusia). None = semua."""
    if not pages:
        return None
    out: list[int] = []
    for part in pages.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _pdf_inputs(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf"))
        if not pdfs:
            sys.exit(f"Tidak ada *.pdf di folder {input_path}")
        return pdfs
    if not input_path.exists():
        sys.exit(f"Input tidak ditemukan: {input_path}")
    return [input_path]


# ---------------------------------------------------------------------------
# Backend: LlamaParse (cloud)
# ---------------------------------------------------------------------------
def parse_llamaparse(pdf_path: Path, pages: list[int] | None, mode: str, language: str) -> list[PageMd]:
    from llama_cloud_services import LlamaParse  # noqa: PLC0415 (impor lazy — hanya bila backend dipakai)

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        sys.exit("LLAMA_CLOUD_API_KEY belum diset (taruh di .env root repo).")

    parse_mode = _LLAMA_MODE.get(mode)
    if parse_mode is None:
        sys.exit(f"--mode tidak dikenal: {mode}. Pilih: {', '.join(_LLAMA_MODE)}")

    kwargs: dict = {
        "api_key": api_key,
        "result_type": "markdown",
        "parse_mode": parse_mode,
        "language": language,           # OCR hint; 'id' untuk dokumen Indonesia
        "split_by_page": True,
    }
    if pages:
        # SDK memakai target_pages 0-indexed, dipisah koma.
        kwargs["target_pages"] = ",".join(str(p - 1) for p in pages)

    parser = LlamaParse(**kwargs)
    json_result = parser.get_json_result(str(pdf_path))  # List[dict], tiap dict punya 'pages'

    out: list[PageMd] = []
    for doc in json_result:
        for pg in doc.get("pages", []):
            # nomor halaman dari SDK biasanya 1-indexed; fallback md->text.
            page_no = pg.get("page")
            md = pg.get("md") or pg.get("text") or ""
            out.append(PageMd(page=int(page_no) if page_no is not None else len(out) + 1, md=md))
    if not out:
        sys.exit("LlamaParse tidak mengembalikan halaman apa pun — cek key/mode/target_pages.")
    return out


# ---------------------------------------------------------------------------
# Backend: Docling (lokal)
# ---------------------------------------------------------------------------
def parse_docling(pdf_path: Path, pages: list[int] | None, _mode: str, _language: str) -> list[PageMd]:
    from docling.document_converter import DocumentConverter  # noqa: PLC0415

    converter = DocumentConverter()

    # Kalau halaman spesifik diminta: konversi per-halaman supaya penanda halaman tepat
    # (Docling markdown standar tidak menyisipkan batas halaman). Untuk parse penuh (pages=None),
    # konversi sekali lalu—bila perlu—ekspor per halaman via page_range.
    out: list[PageMd] = []
    if pages:
        for p in pages:
            res = converter.convert(source=str(pdf_path), page_range=(p, p))
            out.append(PageMd(page=p, md=res.document.export_to_markdown()))
        return out

    # Parse penuh: tentukan jumlah halaman lalu ekspor per halaman (penanda akurat).
    res_all = converter.convert(source=str(pdf_path))
    n_pages = len(getattr(res_all.document, "pages", []) or [])
    if n_pages == 0:
        # tak bisa deteksi jumlah halaman -> satu blob (chunker tetap jalan, halaman kurang presisi)
        return [PageMd(page=1, md=res_all.document.export_to_markdown())]
    for p in range(1, n_pages + 1):
        res = converter.convert(source=str(pdf_path), page_range=(p, p))
        out.append(PageMd(page=p, md=res.document.export_to_markdown()))
    return out


_BACKENDS = {"llamaparse": parse_llamaparse, "docling": parse_docling}


# ---------------------------------------------------------------------------
# Tulis output
# ---------------------------------------------------------------------------
def render_markdown(pages: list[PageMd]) -> str:
    blocks: list[str] = []
    for pg in pages:
        blocks.append(PAGE_MARKER.format(n=pg.page))
        blocks.append(pg.md.strip())
    return "\n\n".join(blocks).strip() + "\n"


def write_output(
    pages: list[PageMd], out_dir: Path, backend: str, source: Path, mode: str, pages_arg: str | None
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    suffix = f"_p{pages_arg.replace(',', '_')}" if pages_arg else ""
    # Bake-off: pisahkan per backend supaya mudah dibandingkan berdampingan.
    sub = out_dir / backend if out_dir.name == "bakeoff" else out_dir
    sub.mkdir(parents=True, exist_ok=True)
    md_path = sub / f"{stem}{suffix}.md"
    md_path.write_text(render_markdown(pages), encoding="utf-8")

    meta = {
        "source": str(source),
        "backend": backend,
        "mode": mode if backend == "llamaparse" else "-",
        "pages_arg": pages_arg,
        "n_pages_out": len(pages),
        "page_numbers": [p.page for p in pages],
        "parsed_at": datetime.now(timezone.utc).isoformat(),
    }
    md_path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Parse PDF regulasi -> markdown page-aware (LlamaParse/Docling).")
    ap.add_argument("--backend", required=True, choices=list(_BACKENDS), help="llamaparse | docling")
    ap.add_argument("--input", required=True, type=Path, help="File PDF atau folder berisi *.pdf")
    ap.add_argument("--out", required=True, type=Path, help="Folder output (mis. data/parsed/v1 atau data/parsed/bakeoff)")
    ap.add_argument("--pages", default=None, help="Rentang halaman 1-indexed, mis. '108-115,120'. Kosong = semua.")
    ap.add_argument("--mode", default="agentic", choices=list(_LLAMA_MODE), help="Mode LlamaParse (diabaikan untuk docling).")
    ap.add_argument("--language", default="id", help="Hint bahasa OCR (LlamaParse). Default 'id'.")
    args = ap.parse_args(argv)

    pages = parse_pages_arg(args.pages)
    backend_fn = _BACKENDS[args.backend]

    for pdf in _pdf_inputs(args.input):
        print(f"[parse] {args.backend} :: {pdf.name} :: pages={args.pages or 'ALL'} :: mode={args.mode}")
        page_list = backend_fn(pdf, pages, args.mode, args.language)
        md_path = write_output(page_list, args.out, args.backend, pdf, args.mode, args.pages)
        print(f"  -> {md_path}  ({len(page_list)} halaman)")


if __name__ == "__main__":
    main()
