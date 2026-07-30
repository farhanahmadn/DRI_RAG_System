"""app/ingest/tabel_chunks.py — tabel-chunk citeable dari lampiran (Langkah 4b).

Menangkap konten lampiran yang PENTING untuk sitasi (bukan sekadar matriks relasional):
  * Lampiran V.B — "Tabel Penjelasan": syarat T (terbatas) & B (bersyarat) + sarana-prasarana minimal
    per zona. Sitasi utama saat indikator Kegiatan = T/B. -> 1 tabel-chunk per zona.
  * Lampiran VI  — ringkasan intensitas KDB/KLB/KDH per zona (dari intensitas_zona.jsonl). Sitasi utk
    indikator KDB/KLB/KDH. -> 1 tabel-chunk per zona.

Output: chunks_tabel.jsonl (bentuk Chunk, level='tabel'), dimuat ingest.py bersama chunks_prosa.jsonl.
Matriks ITBX (V.A) TETAP relasional (matriks_kegiatan) — terlalu granular utk di-embed; klasifikasi
I/T/B/X-nya sudah citeable via prosa (Pasal ketentuan kegiatan).

CLI:
  python -m app.ingest.tabel_chunks --input data/parsed/v1 --structured data/parsed/v1/structured --only tengah
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

from app.ingest.chunk import _short_name
from app.ingest.split import DOKUMEN, ZONA_KANONIK, load_pages

_RE_LAMP_VB = re.compile(r"^#{0,6}\s*LAMPIRAN\s+V\.B\b", re.I)
_RE_LAMP_NEXT = re.compile(r"^#{0,6}\s*LAMPIRAN\s+VI\b", re.I)
_RE_ZONA_CODE = re.compile(r"\(([A-Z0-9][A-Z0-9\-\s]*)\)\s*$")


def _dok_for_file(stem: str) -> dict | None:
    s = stem.lower()
    return next((d for d in DOKUMEN if d["file_match"] in s), None)


def _cells(line: str) -> list[str]:
    p = line.split("|")
    if p and p[0].strip() == "":
        p = p[1:]
    if p and p[-1].strip() == "":
        p = p[:-1]
    return [re.sub(r"<br\s*/?>", "\n", c).strip() for c in p]


def _slug(code: str) -> str:
    return re.sub(r"\s+", "-", code.strip().lower())


# ---------------------------------------------------------------------------
# Lampiran V.B — syarat T/B per zona
# ---------------------------------------------------------------------------
def extract_vb(pages: dict[int, str], dok: dict) -> list[dict]:
    # kumpulkan baris tabel di region V.B (dari heading V.B sampai LAMPIRAN VI)
    in_vb = False
    seq: list[tuple[int, str]] = []  # (page, line) baris tabel dalam region
    for p in sorted(pages):
        for line in pages[p].splitlines():
            if _RE_LAMP_VB.match(line):
                in_vb = True
                continue
            if in_vb and _RE_LAMP_NEXT.match(line):
                in_vb = False
                break
            if in_vb and line.strip().startswith("|"):
                seq.append((p, line))
        if not in_vb and seq:
            # sudah lewati region V.B
            if any(_RE_LAMP_NEXT.match(l) for l in pages[p].splitlines()):
                break

    # pecah per zona: baris awal "| N. | Zona X (CODE) | ... |"
    zones: "OrderedDict[str, dict]" = OrderedDict()
    cur = None
    short = _short_name(dok["dokumen_id"])
    for page, line in seq:
        cs = _cells(line)
        if not cs:
            continue
        # header/separator?
        if cs[0] in ("No", "no") or any("Ketentuan Umum" in c for c in cs) or set(cs[0]) <= {"-"}:
            continue
        is_zone_start = bool(re.match(r"^\d+\.?$", cs[0])) and len(cs) >= 2 and cs[1].strip()
        if is_zone_start:
            klas = cs[1].strip()
            m = _RE_ZONA_CODE.search(klas)
            code = m.group(1).strip() if m else None
            if code and code not in ZONA_KANONIK:
                # normalisasi ringan (spasi ganda)
                code = re.sub(r"\s+", " ", code)
            key = code or klas
            cur = {"code": code, "nama": klas, "page": page, "teks_parts": [klas]}
            # tambahkan isi kolom syarat pada baris awal
            cur["teks_parts"] += [c for c in cs[2:] if c]
            zones[key] = cur
        elif cur is not None:
            # baris lanjutan: ambil teks kolom syarat (kolom ke-3 dst, & kolom klasifikasi bila terisi)
            extra = [c for c in cs[1:] if c]
            cur["teks_parts"] += extra

    out = []
    for key, z in zones.items():
        teks = "\n".join(t for t in z["teks_parts"] if t).strip()
        if len(teks) < 40:
            continue
        code = z["code"]
        cid = f"{dok['dokumen_id']}-vb-{_slug(code) if code else _slug(key)}"
        label = f"Lampiran V.B — Penjelasan Ketentuan Kegiatan (T/B) {z['nama']}"
        out.append({
            "id": cid, "level": "tabel", "parent_id": None,
            "dokumen_id": dok["dokumen_id"], "dokumen": dok["nama"],
            "pasal": None, "ayat": None, "halaman": z["page"],
            "teks": f"{label}:\n{teks}",
            "teks_prefixed": f"[{short} | Lampiran V.B | Zona {code or z['nama']}] {teks[:1500]}",
            "istilah_kode": "Lampiran V.B", "zona": code, "jenis": "RDTR",
            "tanggal_berlaku": dok["tanggal_berlaku"], "tanggal_dicabut": dok["tanggal_dicabut"],
            "to_embed": True,
        })
    return out


# ---------------------------------------------------------------------------
# Lampiran VI — ringkasan intensitas per zona (dari intensitas_zona.jsonl)
# ---------------------------------------------------------------------------
def build_intensitas_chunks(intensitas_rows: list[dict], dok: dict) -> list[dict]:
    short = _short_name(dok["dokumen_id"])
    by_zona: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in intensitas_rows:
        if r["dokumen_id"] != dok["dokumen_id"]:
            continue
        by_zona.setdefault(r["zona"], []).append(r)

    def _rng(vals):
        v = sorted({x for x in vals if x is not None})
        if not v:
            return "-"
        return f"{v[0]:g}" if len(v) == 1 else f"{v[0]:g}–{v[-1]:g}"

    out = []
    for zona, rows in by_zona.items():
        nama = rows[0].get("nama_zona") or zona
        kdb = _rng([r["kdb_maks"] for r in rows])
        klb = _rng([r["klb_maks"] for r in rows])
        kdh = _rng([r["kdh_min"] for r in rows])
        teks = (f"Lampiran VI — Ketentuan Intensitas Pemanfaatan Ruang Zona {zona} ({nama}): "
                f"KDB maksimum {kdb}%; KLB maksimum {klb}; KDH minimum {kdh}%. "
                f"Nilai bervariasi menurut kawasan resapan air (resapan/non-resapan) dan hierarki jalan "
                f"(arteri/kolektor/lokal/lingkungan).")
        out.append({
            "id": f"{dok['dokumen_id']}-vi-{_slug(zona)}", "level": "tabel", "parent_id": None,
            "dokumen_id": dok["dokumen_id"], "dokumen": dok["nama"],
            "pasal": None, "ayat": None, "halaman": rows[0].get("halaman"),
            "teks": teks,
            "teks_prefixed": f"[{short} | Lampiran VI | Zona {zona}] {teks}",
            "istilah_kode": "Lampiran VI", "zona": zona, "jenis": "RDTR",
            "tanggal_berlaku": dok["tanggal_berlaku"], "tanggal_dicabut": dok["tanggal_dicabut"],
            "to_embed": True,
        })
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate tabel-chunk citeable dari Lampiran V.B + VI.")
    ap.add_argument("--input", required=True, type=Path, help="File .md atau folder berisi *.md")
    ap.add_argument("--structured", required=True, type=Path, help="Folder structured (intensitas_zona.jsonl, output)")
    ap.add_argument("--only", default=None, help="Filter file_match (mis. 'tengah').")
    args = ap.parse_args(argv)

    mds = sorted(args.input.glob("*.md")) if args.input.is_dir() else [args.input]
    inten_path = args.structured / "intensitas_zona.jsonl"
    inten = [json.loads(l) for l in inten_path.read_text(encoding="utf-8").splitlines()] if inten_path.exists() else []

    all_chunks: list[dict] = []
    for md_path in mds:
        dok = _dok_for_file(md_path.stem)
        if dok is None or (args.only and args.only.lower() not in dok["file_match"]):
            continue
        pages = load_pages(md_path)
        vb = extract_vb(pages, dok)
        vi = build_intensitas_chunks(inten, dok)
        all_chunks += vb + vi
        print(f"[tabel] {dok['dokumen_id']:20s} V.B={len(vb)} VI={len(vi)}")

    out_path = args.structured / "chunks_tabel.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[tabel] TOTAL={len(all_chunks)} -> {out_path}")


if __name__ == "__main__":
    main()
