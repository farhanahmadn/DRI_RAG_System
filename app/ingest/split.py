"""app/ingest/split.py — pilah data hasil parse: matriks/tabel -> RELASIONAL, sisanya -> prosa (RAG).

Langkah 4 pipeline. Prinsip (CLAUDE.md + rencana): matriks besar (ITBX kegiatan per zona, intensitas
KDB/KLB/KDH per zona) adalah TABEL RELASIONAL OTORITATIF untuk rule engine — BUKAN embedding. Modul ini
mengekstrak matriks itu dari markdown page-aware jadi baris relasional (JSONL), lalu Langkah 7 (ingest.py)
memuatnya ke Postgres apa adanya. Prosa pasal ditangani chunker (Langkah 5), bukan di sini.

Registry `dokumen` (alias -> kanonik) juga dibangun di sini — dipakai get_by_reference (jalur UTAMA sitasi).

Divalidasi terhadap RDTR Sleman Tengah: ITBX 358 kegiatan x 32 zona = 11.456 baris; intensitas 31 zona
x (resapan/non-resapan) x (arteri/kolektor/lokal/lingkungan) = 248 baris. Spot-check sel cocok 100%.

CLI:
  python -m app.ingest.split --input data/parsed/v1 --out data/parsed/v1/structured
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Registry dokumen (3 RDTR wilayah). Nomor Perbup & tanggal diambil dari teks parse.
# Barat/Timur: hanya batang tubuh (tanpa lampiran matriks) -> tak ada baris relasional,
# tetap masuk registry untuk prosa RAG + get_by_reference. Matriks menyusul saat lampirannya ada.
# ---------------------------------------------------------------------------
DOKUMEN: list[dict] = [
    {
        "dokumen_id": "rdtr-sleman-barat",
        "nama": "Peraturan Bupati Sleman Nomor 57 Tahun 2021 tentang RDTR Kawasan Sleman Barat Tahun 2021-2041",
        "jenis": "RDTR",
        "tanggal_berlaku": "2021-12-29",
        "tanggal_dicabut": None,
        "aliases": ["RDTR Sleman Barat", "RDTR Kawasan Sleman Barat", "Perbup Sleman 57/2021",
                    "Peraturan Bupati Sleman Nomor 57 Tahun 2021"],
        "file_match": "barat",
    },
    {
        "dokumen_id": "rdtr-sleman-tengah",
        "nama": "Peraturan Bupati Sleman Nomor 80 Tahun 2023 tentang RDTR Kawasan Sleman Tengah Tahun 2023-2043",
        "jenis": "RDTR",
        "tanggal_berlaku": "2023-12-21",   # OCR-approx (dok. hasil scan); tahun & masa berlaku pasti 2023-2043
        "tanggal_dicabut": None,
        "aliases": ["RDTR Sleman Tengah", "RDTR Kawasan Sleman Tengah", "Perbup Sleman 80/2023",
                    "Peraturan Bupati Sleman Nomor 80 Tahun 2023"],
        "file_match": "tengah",
    },
    {
        "dokumen_id": "rdtr-sleman-timur",
        "nama": "Peraturan Bupati Sleman Nomor 3 Tahun 2021 tentang RDTR Kawasan Sleman Timur Tahun 2021-2040",
        "jenis": "RDTR",
        "tanggal_berlaku": "2021-01-15",
        "tanggal_dicabut": None,
        "aliases": ["RDTR Sleman Timur", "RDTR Kawasan Sleman Timur", "Perbup Sleman 3/2021",
                    "Peraturan Bupati Sleman Nomor 3 Tahun 2021"],
        "file_match": "timur",
    },
]

# Kode zona kanonik (32) dari matriks Tengah; dipakai juga untuk normalisasi.
ZONA_KANONIK = {
    "BA", "PS", "RTH-2", "RTH-3", "RTH-4", "RTH-7", "CA", "TWA", "CB", "BJ", "P-1", "P-1 LP2B",
    "PTL", "KPI", "W", "R-2", "R-3", "R-4", "SPU-1", "SPU-2", "SPU-3", "RTNH", "C-1", "K-1",
    "K-2", "K-3", "KT", "PL-3", "PL-4", "PP", "TR", "HK",
}
_IZIN = {"I", "T", "B", "X"}
_HIERARKI = ["arteri", "kolektor", "lokal", "lingkungan"]


# ---------------------------------------------------------------------------
# Util markdown page-aware
# ---------------------------------------------------------------------------
def load_pages(md_path: Path) -> dict[int, str]:
    md = md_path.read_text(encoding="utf-8")
    pages: dict[int, str] = {}
    cur: int | None = None
    for seg in re.split(r"(<!-- PAGE \d+ -->)", md):
        m = re.match(r"<!-- PAGE (\d+) -->", seg)
        if m:
            cur = int(m.group(1))
            pages[cur] = ""
        elif cur is not None:
            pages[cur] += seg
    return pages


def _cells(line: str) -> list[str]:
    parts = line.split("|")
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return [re.sub(r"<br/>", " ", c).strip() for c in parts]


def _norm_zona(z: str) -> str:
    z = re.sub(r"<br/>", " ", z).strip()
    return "RTNH" if z.replace(" ", "").upper() == "RTNH" else z


def _num(x: str) -> float | None:
    x = x.strip()
    if x in ("-", "", "—", "–"):
        return None
    x = x.replace("%", "").replace(",", ".").strip()
    try:
        return float(x)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Extractor 1: Matriks ITBX (kegiatan x zona -> izin I/T/B/X)
# ---------------------------------------------------------------------------
def extract_itbx(pages: dict[int, str], dok: dict) -> list[dict]:
    rows: list[dict] = []
    kategori: str | None = None
    for p in sorted(pages):
        lines = [l for l in pages[p].splitlines() if l.strip().startswith("|")]
        zones: list[str] | None = None
        for l in lines:
            cs = _cells(l)
            if any("NAMA KEGIATAN" in c.upper() for c in cs):
                zones = [_norm_zona(c) for c in cs[2:]]
                break
        if not zones:
            continue
        for l in lines:
            cs = _cells(l)
            if not cs:
                continue
            c0 = cs[0].replace("*", "").strip()
            if re.fullmatch(r"[A-Z]", c0) and len(cs) >= 2 and cs[1].startswith("**"):
                kategori = c0
                continue
            if not re.fullmatch(r"\d{3}", c0):
                continue
            kode = c0
            nama = cs[1].replace("*", "").strip()
            vals = cs[2:]
            for i, z in enumerate(zones):
                if i < len(vals) and z in ZONA_KANONIK:
                    v = vals[i].replace("*", "").strip().upper()
                    if v in _IZIN:
                        rows.append({
                            "dokumen_id": dok["dokumen_id"], "zona": z, "kode_kegiatan": kode,
                            "nama_kegiatan": nama, "kategori": kategori, "izin": v,
                            "keterangan": None, "lampiran": "Lampiran V.A", "halaman": p,
                            "tanggal_berlaku": dok["tanggal_berlaku"], "tanggal_dicabut": dok["tanggal_dicabut"],
                        })
    return rows


# ---------------------------------------------------------------------------
# Extractor 2: Intensitas (zona x resapan x hierarki jalan -> KDB/KLB/KDH)
# 24 kolom nilai: KDB[nonR A/K/L/Lk, R A/K/L/Lk], KLB[...], KDH[...]
# ---------------------------------------------------------------------------
def extract_intensitas(pages: dict[int, str], dok: dict) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(pages):
        for l in [l for l in pages[p].splitlines() if l.strip().startswith("|")]:
            cs = _cells(l)
            if len(cs) < 26:
                continue
            kode = _norm_zona(cs[1])
            if kode not in ZONA_KANONIK:
                continue
            nama = cs[0].strip()
            vals = cs[2:26]
            if len(vals) < 24:
                continue
            kdb, klb, kdh = vals[0:8], vals[8:16], vals[16:24]
            for ri, resapan in enumerate((False, True)):     # 0-3 non-resapan, 4-7 resapan
                for hi, hierarki in enumerate(_HIERARKI):
                    idx = ri * 4 + hi
                    rows.append({
                        "dokumen_id": dok["dokumen_id"], "zona": kode, "nama_zona": nama,
                        "kawasan_resapan": resapan, "hierarki_jalan": hierarki,
                        "kdb_maks": _num(kdb[idx]), "klb_maks": _num(klb[idx]), "kdh_min": _num(kdh[idx]),
                        "lampiran": "Lampiran VI", "halaman": p,
                        "tanggal_berlaku": dok["tanggal_berlaku"], "tanggal_dicabut": dok["tanggal_dicabut"],
                    })
    return rows


def _dok_for_file(stem: str) -> dict | None:
    s = stem.lower()
    for d in DOKUMEN:
        if d["file_match"] in s:
            return d
    return None


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Pilah matriks -> relasional (JSONL) + registry dokumen.")
    ap.add_argument("--input", required=True, type=Path, help="File .md hasil parse atau folder berisi *.md")
    ap.add_argument("--out", required=True, type=Path, help="Folder output structured (mis. data/parsed/v1/structured)")
    args = ap.parse_args(argv)

    mds = sorted(args.input.glob("*.md")) if args.input.is_dir() else [args.input]
    all_matriks: list[dict] = []
    all_inten: list[dict] = []

    for md_path in mds:
        dok = _dok_for_file(md_path.stem)
        if dok is None:
            print(f"[split] LEWATI (tak ada di registry): {md_path.name}")
            continue
        pages = load_pages(md_path)
        mk = extract_itbx(pages, dok)
        iz = extract_intensitas(pages, dok)
        all_matriks += mk
        all_inten += iz
        note = "" if (mk or iz) else "  (batang tubuh saja — tanpa lampiran matriks)"
        print(f"[split] {dok['dokumen_id']:20s} halaman={len(pages)}  matriks={len(mk)}  intensitas={len(iz)}{note}")

    # registry dokumen (buang field internal file_match)
    reg = [{k: v for k, v in d.items() if k != "file_match"} for d in DOKUMEN]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dokumen.json").write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(args.out / "matriks_kegiatan.jsonl", all_matriks)
    _write_jsonl(args.out / "intensitas_zona.jsonl", all_inten)

    print(f"\n[split] TOTAL matriks_kegiatan={len(all_matriks)}  intensitas_zona={len(all_inten)}  dokumen={len(reg)}")
    print(f"[split] -> {args.out}/(dokumen.json, matriks_kegiatan.jsonl, intensitas_zona.jsonl)")


if __name__ == "__main__":
    main()
