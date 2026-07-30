"""app/ingest/chunk.py — chunking STRUKTURAL prosa pasal (Langkah 5).

Bukan semantic chunking (CLAUDE.md § Chunking): segmentasi regex per Pasal(parent)/Ayat(child),
pola parent-child + metadata kaya + prefiks kontekstual. Tabel/matriks TIDAK di sini (itu relasional,
lihat split.py). Output = record berbentuk `Chunk` (app/retrieval/base.py) + kolom pipeline, JSONL,
untuk dimuat & di-embed oleh ingest.py (Langkah 7).

Aturan (dari rencana):
  * Pasal = chunk induk (level='pasal', parent_id=None). Teks induk = seluruh isi pasal (konteks penuh
    untuk small-to-big / get_parent).
  * Ayat "(n)" = chunk anak (level='ayat', parent_id = id pasal). INI yang utama di-embed & diretrieve.
  * Pasal 1 (Ketentuan Umum) memakai daftar definisi bernomor "1. ... 2. ..." -> tiap definisi 1 chunk
    (level='ayat', ayat=nomor definisi). "Pasal 1 (definisi) = 1 ayat 1 chunk".
  * Pasal tanpa ayat -> satu chunk pasal (childless) yang ikut di-embed.
  * id STABIL & deterministik (dipakai sebagai citation_id): '<dok>-p<pasal>[-a<ayat>]'.
  * halaman akurat dari penanda '<!-- PAGE n -->' (syarat sitasi).
  * to_embed: ayat + pasal-childless + (tabel, nanti) = True; pasal-induk-berayat = False (parent
    disimpan utk get_parent, tak perlu ikut ranking — hindari duplikat hit parent/child).

CLI:
  python -m app.ingest.chunk --input data/parsed/v1 --out data/parsed/v1/structured --only timur
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.ingest.split import DOKUMEN

# heading pasal: baris berisi HANYA "Pasal N" (opsional prefiks markdown #). Anchored -> tak menangkap
# rujukan inline "...dalam Pasal 132 ayat (4)".
_RE_PASAL_HEAD = re.compile(r"^#{0,6}\s*Pasal\s+(\d+)\s*$")
# Batas akhir batang tubuh: heading LAMPIRAN (matriks/intensitas/peta) ATAU blok penutup Perbup
# ("Ditetapkan/Diundangkan di Sleman" -> tanda tangan). Prosa berhenti di sini supaya pasal terakhir
# tidak "menelan" tanda tangan + ratusan halaman lampiran. Matriks ditangani relasional (split.py).
_RE_STOP = re.compile(
    r"^#{0,6}\s*(LAMPIRAN\s+[IVXLC]+(\.[A-Za-z0-9]+)?\s*$|(Ditetapkan|Diundangkan)\s+di\s+Sleman)", re.I)
_RE_AYAT = re.compile(r"^\s*\((\d+)\)\s+")          # "(1) ..."
_RE_DEFINISI = re.compile(r"^\s*(\d+)\.\s+")         # "1. ..." (Pasal 1 definisi)
_RE_PAGE = re.compile(r"^<!-- PAGE (\d+) -->\s*$")
# noise OCR yang dibuang saat berdiri sendiri di satu baris
_RE_NOISE = re.compile(r"^\s*(QR\s*Code|BERITA DAERAH[^\n]*|\d{1,3}|-\s*\d{1,3}\s*-)\s*$", re.I)


def _short_name(dokumen_id: str) -> str:
    # 'rdtr-sleman-timur' -> 'RDTR Sleman Timur'
    parts = dokumen_id.split("-")
    return " ".join(p.upper() if p == "rdtr" else p.capitalize() for p in parts)


def _dok_for_file(stem: str) -> dict | None:
    s = stem.lower()
    return next((d for d in DOKUMEN if d["file_match"] in s), None)


def _iter_lines_with_page(md: str):
    """Yield (page, line) menelusuri markdown; page dilacak dari penanda halaman."""
    page = 1
    for raw in md.splitlines():
        pm = _RE_PAGE.match(raw)
        if pm:
            page = int(pm.group(1))
            continue
        yield page, raw


def _clean(text: str) -> str:
    lines = []
    for l in text.splitlines():
        if _RE_NOISE.match(l):
            continue
        lines.append(l.rstrip())
    # rapikan baris kosong beruntun
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return out


def _split_pasal_blocks(md: str):
    """Bagi markdown jadi blok per Pasal. Return list of dict(pasal, page_start, lines[(page,line)])."""
    blocks = []
    cur = None
    for page, line in _iter_lines_with_page(md):
        if _RE_STOP.match(line):
            # lampiran / penutup Perbup -> batang tubuh selesai; tutup blok berjalan & stop.
            break
        m = _RE_PASAL_HEAD.match(line)
        if m:
            if cur:
                blocks.append(cur)
            cur = {"pasal": m.group(1), "page": page, "lines": []}
        elif cur is not None:
            cur["lines"].append((page, line))
    if cur:
        blocks.append(cur)
    return blocks


# Penanda definisi INLINE (Pasal 1): "N. <Term berkapital>", toleran spasi OCR di angka ("11 7.").
_RE_DEF_INLINE = re.compile(r"(\d[\d ]{0,4})\.\s+(?=[A-Z\"*])")


def _segment_definisi(lines: list[tuple[int, str]]):
    """Pasal 1 (Ketentuan Umum): definisi bisa tercetak sebagai paragraf mengalir (bukan baris
    terpisah) & nomornya kadah terpecah OCR ('11 7.'). Pecah berdasarkan penanda inline 'N. Term',
    dengan nomor menaik (buang sub-daftar bersarang & angka dalam kalimat)."""
    # gabung teks + peta halaman per karakter (agar tiap definisi dapat halaman yang tepat)
    text_parts, pagemap = [], []
    for idx, (page, line) in enumerate(lines):
        text_parts.append(line)
        pagemap.extend([page] * len(line))
        if idx < len(lines) - 1:
            pagemap.append(page)  # untuk newline penyambung
    text = "\n".join(text_parts)

    bounds = []  # (pos, nomor)
    last = 0
    for m in _RE_DEF_INLINE.finditer(text):
        num = int(m.group(1).replace(" ", ""))
        if last < num <= last + 3:  # sekuensial-ish: buang reset bersarang & lompatan jauh
            bounds.append((m.start(), num))
            last = num
    if not bounds:
        return _clean(text), []
    intro = _clean(text[: bounds[0][0]])
    ayats = []
    for i, (pos, num) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
        page = pagemap[pos] if pos < len(pagemap) else (lines[0][0] if lines else 1)
        ayats.append({"no": str(num), "page": page, "teks": _clean(text[pos:end])})
    return intro, ayats


def _segment_ayat(lines: list[tuple[int, str]], definisi_mode: bool):
    """Bagi baris-baris sebuah pasal jadi (intro, [(ayat_no, page, teks), ...]).
    definisi_mode=True -> definisi Pasal 1 (inline); else pakai '(1)'/'(2)'."""
    if definisi_mode:
        return _segment_definisi(lines)
    rx = _RE_AYAT
    intro: list[str] = []
    ayats: list[dict] = []
    cur = None
    last_no = 0  # nomor ayat/definisi legal SELALU menaik; sub-daftar bersarang (reset ke 1,2..) = isi
    for page, line in lines:
        m = rx.match(line)
        if m and int(m.group(1)) > last_no:
            if cur:
                ayats.append(cur)
            last_no = int(m.group(1))
            cur = {"no": m.group(1), "page": page, "buf": [line]}
        elif cur is not None:
            cur["buf"].append(line)
        else:
            intro.append(line)
    if cur:
        ayats.append(cur)
    intro_txt = _clean("\n".join(intro))
    out = [{"no": a["no"], "page": a["page"], "teks": _clean("\n".join(a["buf"]))} for a in ayats]
    return intro_txt, out


def chunk_document(md_path: Path, dok: dict) -> list[dict]:
    md = md_path.read_text(encoding="utf-8")
    short = _short_name(dok["dokumen_id"])
    dokumen_nama = dok["nama"]
    chunks: list[dict] = []

    for blk in _split_pasal_blocks(md):
        pasal = blk["pasal"]
        definisi_mode = pasal == "1"
        intro, ayats = _segment_ayat(blk["lines"], definisi_mode)

        pid = f"{dok['dokumen_id']}-p{pasal}"
        # teks induk = konteks penuh pasal (intro + semua ayat) untuk get_parent/small-to-big
        full_parts = [f"Pasal {pasal}"]
        if intro:
            full_parts.append(intro)
        for a in ayats:
            full_parts.append(a["teks"])
        full_teks = "\n".join(p for p in full_parts if p).strip()

        has_children = bool(ayats)
        # chunk PASAL (induk). Childless -> ikut embed; berayat -> simpan saja (to_embed=False).
        chunks.append({
            "id": pid, "level": "pasal", "parent_id": None,
            "dokumen_id": dok["dokumen_id"], "dokumen": dokumen_nama,
            "pasal": pasal, "ayat": None, "halaman": blk["page"],
            "teks": full_teks,
            "teks_prefixed": f"[{short} | Pasal {pasal}] {intro or full_teks}",
            "zona": None, "jenis": "RDTR", "istilah_kode": None,
            "tanggal_berlaku": dok["tanggal_berlaku"], "tanggal_dicabut": dok["tanggal_dicabut"],
            "to_embed": not has_children,
        })
        # chunk AYAT (anak)
        for a in ayats:
            label = f"Pasal {pasal} ayat ({a['no']})" if not definisi_mode else f"Pasal {pasal} angka {a['no']}"
            chunks.append({
                "id": f"{pid}-a{a['no']}", "level": "ayat", "parent_id": pid,
                "dokumen_id": dok["dokumen_id"], "dokumen": dokumen_nama,
                "pasal": pasal, "ayat": a["no"], "halaman": a["page"],
                "teks": a["teks"],
                "teks_prefixed": f"[{short} | {label}] {a['teks']}",
                "zona": None, "jenis": "RDTR", "istilah_kode": None,
                "tanggal_berlaku": dok["tanggal_berlaku"], "tanggal_dicabut": dok["tanggal_dicabut"],
                "to_embed": True,
            })
    return chunks


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Chunking struktural prosa Pasal/Ayat -> JSONL Chunk.")
    ap.add_argument("--input", required=True, type=Path, help="File .md atau folder berisi *.md")
    ap.add_argument("--out", required=True, type=Path, help="Folder output (structured)")
    ap.add_argument("--only", default=None, help="Filter file_match dokumen (mis. 'timur'). Kosong = semua.")
    args = ap.parse_args(argv)

    mds = sorted(args.input.glob("*.md")) if args.input.is_dir() else [args.input]
    all_chunks: list[dict] = []
    for md_path in mds:
        dok = _dok_for_file(md_path.stem)
        if dok is None:
            continue
        if args.only and args.only.lower() not in dok["file_match"]:
            continue
        ch = chunk_document(md_path, dok)
        n_pasal = sum(1 for c in ch if c["level"] == "pasal")
        n_ayat = sum(1 for c in ch if c["level"] == "ayat")
        n_emb = sum(1 for c in ch if c["to_embed"])
        print(f"[chunk] {dok['dokumen_id']:20s} pasal={n_pasal} ayat={n_ayat} total={len(ch)} to_embed={n_emb}")
        all_chunks += ch

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "chunks_prosa.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\n[chunk] TOTAL chunks={len(all_chunks)} -> {out_path}")


if __name__ == "__main__":
    main()
