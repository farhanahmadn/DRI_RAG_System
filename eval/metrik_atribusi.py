"""eval/metrik_atribusi.py — metrik atribusi sitasi, dihitung dari `logs/precheck.jsonl` + korpus.

Kenapa modul ini ada. `eval/eval_rag.py` hanya mengukur leg RETRIEVAL: apakah chunk yang benar masuk
top-k. Ia tidak menjawab pertanyaan yang justru paling khas dari sistem ini — apakah sitasi yang
akhirnya muncul di luaran benar-benar menunjuk chunk yang disodorkan, dan apakah kutipannya benar
ada di chunk itu. Verifikasinya bisa mekanis karena tiap sitasi membawa `citation_id` yang menunjuk
chunk konkret, jadi tak ada anotator yang dibutuhkan.

CARA MEMBACA ANGKANYA — ini bagian terpenting di berkas ini.

Log BUKAN kumpulan trafik produksi yang bersih. Diperiksa langsung, dari 444 permohonan era 3-poin
di log hanya **57** yang sitasinya berasal dari retriever NYATA; **335** berasal dari
`MockRetriever` (replay/test lokal dengan `RETRIEVER=mock`), 47 tanpa sitasi sama sekali. Kalau
semuanya dirata-ratakan tanpa dipilah, yang terukur adalah MockRetriever, bukan sistem. Karena itu
tiap metrik di sini dihitung HANYA pada himpunan tempat ia memang bisa dihitung, dan laporan selalu
mencetak penyebutnya beserta alasan pengecualian. Angka tanpa penyebut tidak boleh dikutip.

Empat metrik, seluruhnya mekanis:

1. `presisi_korpus`    — sitasi ber-id pola chunk korpus yang BENAR ada di tabel `chunks`.
                         Mengukur halusinasi rujukan.
2. `presisi_anchor`    — sitasi `anchor-N` yang N-nya memang indeks sah pada `dasar_hukum` poin itu
                         (tersimpan di `request.gate_hukum.tahapan.<poin>.dasar_hukum`).
3. `groundedness`      — `kutipan` yang teksnya memang muncul di chunk yang disitasi, setelah
                         normalisasi spasi/tanda baca. Kutipan karangan tertangkap di sini. Mode
                         gagal yang sudah terlihat di log dihitung terpisah: LLM mengutip JUDUL
                         DOKUMEN alih-alih isi pasal, padahal `terverifikasi` tetap true.
4. `recall_zona`       — poin yang menyitasi setidaknya satu chunk milik keluarga zona pemohon.
                         Langsung menangkap kelas bug APP-2026-2428 (sitasi Cagar Alam untuk
                         pemohon Zona Pertanian). Dihitung per-poin, bukan per-permohonan: `dampak`
                         memang sah menyitasi pasal lintas-zona, jadi memaksakannya di sana akan
                         menghukum perilaku yang benar.

Sebagai pemeriksaan silang, `kesepakatan_flag` membandingkan flag `terverifikasi` yang dirakit
sistem sendiri dengan verifikasi mekanis di sini. Flag itu hanya memeriksa keberadaan id, jadi
selisihnya terhadap `groundedness` adalah ukuran seberapa jauh "terverifikasi" boleh dipercaya.

WAKTU ADALAH BAGIAN DARI METRIK. Log merentang beberapa versi kode, jadi merata-ratakan seluruh
periode akan mencampur perilaku sebelum dan sesudah perbaikan. Terbukti saat modul ini pertama
dijalankan: `recall_zona` poin intensitas keluar 16.7%, dan 7 kasus sitasi lintas keluarga zona
di dalamnya ternyata APP-2026-6191 tanggal 6-7 Agustus — sedangkan filter `zona_prefix` yang
memperbaikinya baru masuk 10 Agustus (commit 2c32808). Angka itu mengukur bug yang sudah
ditambal, bukan keadaan sistem. Karena itu `--sejak` wajib dipakai saat angkanya dikutip, dan
laporan selalu mencetak rentang tanggal yang ikut terhitung.

CLI:
  python -m eval.metrik_atribusi
  python -m eval.metrik_atribusi --log logs/precheck.jsonl --out eval/atribusi.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_AKAR = Path(__file__).parent.parent
_LOG = _AKAR / "logs" / "precheck.jsonl"
_OUT = Path(__file__).parent / "atribusi.json"

# Poin desain 3-poin yang berjalan sekarang. Permohonan era 8-indikator lama (LP2B-01, KDB-01, ...)
# dibuang: jalur reasoning-nya sudah tidak ada, jadi mengukurnya hanya mengaburkan angka.
_POIN_SEKARANG = ("itbx", "intensitas", "dampak")

# Poin yang sumber otoritatifnya memang tabel per-zona (Lampiran VI utk intensitas, V.B utk itbx),
# sehingga "menyitasi chunk zona pemohon" bermakna. `dampak` sengaja di luar daftar — ketentuannya
# tersaji sebagai pasal prosa lintas-zona (mis. Pasal 53 kawasan resapan air), jadi menuntut sitasi
# ber-zona di sana akan menandai jawaban yang benar sebagai salah.
_POIN_BERZONA = ("intensitas", "itbx")

_RE_ANCHOR = re.compile(r"^anchor-(\d+)$")
_RE_BUKAN_KATA = re.compile(r"[^0-9a-z]+")
# Kutipan sangat pendek tak bisa dinilai groundedness-nya secara bermakna (substring apa pun mudah
# ketemu). Batas ini ditetapkan sebelum melihat hasilnya, supaya bukan hasil penyetelan ke angka.
_MIN_CHAR_KUTIPAN = 25


def _dokumen_mock() -> frozenset[str]:
    """Nama dokumen yang dipakai MockRetriever — penanda baris log hasil replay/test, bukan produksi.

    Dibaca dari mock-nya sendiri, bukan disalin sebagai string: kalau fixture mock berganti nama
    dokumen, deteksi ini ikut, ketimbang diam-diam menganggap baris mock sebagai produksi.
    """
    try:
        from app.retrieval.mock import _build_mock_chunks

        return frozenset(c.dokumen for c in _build_mock_chunks() if c.dokumen)
    except Exception:
        return frozenset()


def _normalisasi(teks: str) -> str:
    """Turunkan teks ke bentuk yang bisa dibandingkan: huruf kecil, tanpa tanda baca, spasi tunggal.

    Perlu karena LLM merapikan kutipan — mengubah "a. pengembangan baru;" jadi "pengembangan baru"
    atau menormalkan tanda hubung. Yang kita uji adalah apakah ISI-nya ada di chunk, bukan apakah
    karakternya identik.
    """
    return _RE_BUKAN_KATA.sub(" ", (teks or "").lower()).strip()


def _jenis_id(cid: str | None) -> str:
    if not cid:
        return "kosong"
    if _RE_ANCHOR.match(cid):
        return "anchor"
    if cid.startswith("rdtr-sleman-"):
        return "korpus"
    return "lain"


@dataclass
class Sitasi:
    permohonan: str
    poin_id: str
    citation_id: str
    kutipan: str
    terverifikasi: bool
    jenis: str
    dasar_hukum_n: int          # jumlah anchor yang tersedia utk poin ini
    zona_induk: str | None
    zona_subzone: str | None
    dari_mock: bool


@dataclass
class Permohonan:
    id: str
    timestamp: str
    zona_induk: str | None
    zona_subzone: str | None
    dari_mock: bool
    sitasi: list[Sitasi] = field(default_factory=list)
    poin_dinilai: dict[str, str] = field(default_factory=dict)   # poin_id -> status


def muat_permohonan(path: Path, sejak: str | None = None) -> tuple[list[Permohonan], dict]:
    """Baca log, ambil hanya era 3-poin, dan laporkan apa saja yang dibuang beserta sebabnya.

    `sejak` (ISO, mis. "2026-09-07") membuang permohonan yang dijalankan versi kode lebih tua.
    Tanpa itu, metrik mencampur perilaku lintas versi — lihat catatan WAKTU di docstring modul.
    """
    dok_mock = _dokumen_mock()
    keluar: list[Permohonan] = []
    lewat = Counter()

    for baris in path.read_text(encoding="utf-8").splitlines():
        if not baris.strip():
            continue
        try:
            d = json.loads(baris)
        except Exception:
            lewat["baris tak terbaca"] += 1
            continue
        if sejak and str(d.get("timestamp") or "")[:10] < sejak:
            lewat[f"sebelum {sejak}"] += 1
            continue
        resp = d.get("response") or {}
        poin = [p for p in (resp.get("poin") or []) if isinstance(p, dict)]
        if not any(p.get("poin_id") in _POIN_SEKARANG for p in poin):
            lewat["era 8-indikator lama"] += 1
            continue

        req = d.get("request") or {}
        lok = req.get("lokasi") or {}
        tahapan = ((req.get("gate_hukum") or {}).get("tahapan") or {})
        semua_sitasi = [s for p in poin for s in (p.get("sitasi") or []) if isinstance(s, dict)]
        dari_mock = any((s.get("dokumen") or "") in dok_mock for s in semua_sitasi)

        pm = Permohonan(
            id=str(req.get("application_number") or req.get("application_id") or "?"),
            timestamp=str(d.get("timestamp") or "")[:19],
            zona_induk=lok.get("rdtr_zone"),
            zona_subzone=lok.get("rdtr_subzone"),
            dari_mock=dari_mock,
        )
        for p in poin:
            pid = p.get("poin_id")
            if pid not in _POIN_SEKARANG:
                continue
            pm.poin_dinilai[pid] = p.get("status") or ""
            n_anchor = len(((tahapan.get(pid) or {}).get("dasar_hukum")) or [])
            for s in (p.get("sitasi") or []):
                if not isinstance(s, dict):
                    continue
                cid = s.get("citation_id")
                pm.sitasi.append(Sitasi(
                    permohonan=pm.id, poin_id=pid, citation_id=cid or "",
                    kutipan=s.get("kutipan") or "", terverifikasi=bool(s.get("terverifikasi")),
                    jenis=_jenis_id(cid), dasar_hukum_n=n_anchor,
                    zona_induk=pm.zona_induk, zona_subzone=pm.zona_subzone,
                    dari_mock=(s.get("dokumen") or "") in dok_mock,
                ))
        keluar.append(pm)

    if not dok_mock:
        lewat["PERINGATAN: penanda dokumen mock tak terbaca"] += 1
    return keluar, dict(lewat)


def _muat_chunks(ids: set[str]) -> dict[str, dict]:
    """Ambil teks & zona chunk yang disitasi. Tanpa DB, metrik yang butuh teks dilewati — bukan
    diganti nilai default yang kelihatan seperti hasil."""
    if not ids:
        return {}
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        return {}
    try:
        import psycopg

        with psycopg.connect(dsn) as conn:
            baris = conn.execute(
                "SELECT id, zona, level, teks FROM chunks WHERE id = ANY(%s)", (sorted(ids),)
            ).fetchall()
        return {r[0]: {"zona": r[1], "level": r[2], "teks": r[3]} for r in baris}
    except Exception as exc:  # DB mati -> lapor apa adanya, jangan mengarang
        print(f"[atribusi] DB tak terjangkau ({exc}); metrik yang butuh teks chunk dilewati")
        return {}


def _keluarga(kode: str | None) -> str | None:
    """'R-2' -> 'R'. Sejalan dgn generator._keluarga_zona utk kode sub-zona."""
    if not kode:
        return None
    return kode.split("-")[0].upper()


def _prefix_zona_pemohon(zona_induk: str | None, zona_subzone: str | None) -> str | None:
    if zona_subzone:
        return _keluarga(zona_subzone)
    from app.reasoning.generator import _zona_prefix_dari_nama

    p = _zona_prefix_dari_nama(zona_induk)
    return p.upper() if p else None


def _proporsi(pembilang: int, penyebut: int) -> float | None:
    """None kalau penyebutnya nol — 0.0 akan terbaca sebagai 'gagal total', bukan 'tak terukur'."""
    return (pembilang / penyebut) if penyebut else None


def hitung(permohonan: list[Permohonan], chunks: dict[str, dict]) -> dict:
    nyata = [p for p in permohonan if not p.dari_mock]
    sitasi_nyata = [s for p in nyata for s in p.sitasi if not s.dari_mock]

    # --- 1. presisi sitasi ber-id korpus -----------------------------------------------------
    korpus = [s for s in sitasi_nyata if s.jenis == "korpus"]
    korpus_ada = [s for s in korpus if s.citation_id in chunks] if chunks else []

    # --- 2. presisi sitasi anchor -------------------------------------------------------------
    anchor = [s for s in sitasi_nyata if s.jenis == "anchor"]
    anchor_sah = [s for s in anchor
                  if int(_RE_ANCHOR.match(s.citation_id).group(1)) < s.dasar_hukum_n]

    # --- 3. groundedness kutipan ---------------------------------------------------------------
    dinilai_kutipan, grounded, judul_dokumen, terlalu_pendek = [], [], [], []
    for s in korpus_ada:
        chunk = chunks[s.citation_id]
        k = _normalisasi(s.kutipan)
        if len(k) < _MIN_CHAR_KUTIPAN:
            terlalu_pendek.append(s)
            continue
        dinilai_kutipan.append(s)
        if k in _normalisasi(chunk["teks"]):
            grounded.append(s)
        elif _normalisasi("Peraturan Bupati Sleman") in k or _normalisasi("RDTR Kawasan") in k:
            # Mode gagal nyata di log: yang dikutip adalah JUDUL dokumen, bukan isi pasal —
            # formalnya "ada rujukan", substansinya tidak menjelaskan apa pun.
            judul_dokumen.append(s)

    # --- 4. recall sitasi terhadap zona pemohon ------------------------------------------------
    rz_total, rz_kena, rz_lewat = 0, 0, Counter()
    per_poin_zona: dict[str, dict] = {}
    for pid in _POIN_BERZONA:
        total = kena = 0
        for p in nyata:
            if pid not in p.poin_dinilai or p.poin_dinilai[pid] == "Tidak Dinilai":
                rz_lewat[f"{pid}: poin tak dinilai/absen"] += 1
                continue
            prefix = _prefix_zona_pemohon(p.zona_induk, p.zona_subzone)
            if not prefix:
                rz_lewat[f"{pid}: zona pemohon tak terpetakan"] += 1
                continue
            id_poin = [s.citation_id for s in p.sitasi
                       if s.poin_id == pid and not s.dari_mock and s.jenis == "korpus"]
            if not chunks or not any(c in chunks for c in id_poin):
                rz_lewat[f"{pid}: tak ada sitasi ber-id korpus utk dinilai"] += 1
                continue
            total += 1
            if any(_keluarga(chunks[c]["zona"]) == prefix for c in id_poin if c in chunks):
                kena += 1
        per_poin_zona[pid] = {"n": total, "kena": kena, "nilai": _proporsi(kena, total)}
        rz_total += total
        rz_kena += kena

    # --- 5. kesepakatan flag `terverifikasi` dgn verifikasi mekanis ---------------------------
    sepakat = beda = 0
    for s in korpus:
        mekanis = bool(chunks) and s.citation_id in chunks
        if s.terverifikasi == mekanis:
            sepakat += 1
        else:
            beda += 1

    tstamp = sorted(p.timestamp for p in nyata if p.timestamp)
    return {
        "dibuat": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cakupan": {
            "rentang_tanggal_dinilai": [tstamp[0], tstamp[-1]] if tstamp else None,
            "permohonan_era_sekarang": len(permohonan),
            "permohonan_retriever_nyata": len(nyata),
            "permohonan_dari_mock": sum(1 for p in permohonan if p.dari_mock),
            "sitasi_dinilai": len(sitasi_nyata),
            "sitasi_per_jenis_id": dict(Counter(s.jenis for s in sitasi_nyata)),
            "chunk_tersedia_dari_db": len(chunks),
        },
        "presisi_korpus": {
            "n": len(korpus), "kena": len(korpus_ada),
            "nilai": _proporsi(len(korpus_ada), len(korpus)) if chunks else None,
        },
        "presisi_anchor": {
            "n": len(anchor), "kena": len(anchor_sah),
            "nilai": _proporsi(len(anchor_sah), len(anchor)),
        },
        "groundedness_kutipan": {
            "n": len(dinilai_kutipan), "kena": len(grounded),
            "nilai": _proporsi(len(grounded), len(dinilai_kutipan)),
            "kutipan_judul_dokumen": len(judul_dokumen),
            "kutipan_terlalu_pendek_dilewati": len(terlalu_pendek),
        },
        "recall_zona": {
            "n": rz_total, "kena": rz_kena, "nilai": _proporsi(rz_kena, rz_total),
            "per_poin": per_poin_zona,
            "dilewati": dict(rz_lewat),
        },
        "kesepakatan_flag_terverifikasi": {
            "n": len(korpus), "sepakat": sepakat, "beda": beda,
            "nilai": _proporsi(sepakat, len(korpus)) if chunks else None,
        },
    }


def _pct(e: dict) -> str:
    n = e.get("nilai")
    return "tak terukur" if n is None else f"{n:.1%}"


def cetak(h: dict, lewat: dict) -> None:
    c = h["cakupan"]
    print("\n=== CAKUPAN (baca ini sebelum angka mana pun) ===")
    rt = c.get("rentang_tanggal_dinilai")
    print(f"  rentang tanggal yang terhitung    : {rt[0]} .. {rt[1]}" if rt
          else "  rentang tanggal                   : (tak ada permohonan terpakai)")
    print(f"  permohonan era 3-poin di log      : {c['permohonan_era_sekarang']}")
    print(f"  dipakai (retriever nyata)         : {c['permohonan_retriever_nyata']}")
    print(f"  DIBUANG (MockRetriever)           : {c['permohonan_dari_mock']}")
    for k, v in lewat.items():
        print(f"  dibuang ({k}){'':{max(0, 22 - len(k))}}: {v}")
    print(f"  sitasi yang dinilai               : {c['sitasi_dinilai']}  {c['sitasi_per_jenis_id']}")

    print("\n=== METRIK ATRIBUSI ===")
    for kunci, label in (
        ("presisi_korpus", "Presisi sitasi (id korpus ada di DB)"),
        ("presisi_anchor", "Presisi sitasi anchor (indeks dasar_hukum sah)"),
        ("groundedness_kutipan", "Groundedness kutipan (teks ada di chunk)"),
        ("recall_zona", "Recall sitasi thd keluarga zona pemohon"),
        ("kesepakatan_flag_terverifikasi", "Kesepakatan flag 'terverifikasi'"),
    ):
        e = h[kunci]
        inti = e.get("kena", e.get("sepakat"))
        print(f"  {label:48s} {_pct(e):>12s}   (n={e['n']}, kena={inti})")

    g = h["groundedness_kutipan"]
    if g["kutipan_judul_dokumen"]:
        print(f"\n  !! {g['kutipan_judul_dokumen']} kutipan berisi JUDUL DOKUMEN, bukan isi pasal — "
              f"formalnya bersitasi, substansinya tidak menjelaskan apa pun.")
    if g["kutipan_terlalu_pendek_dilewati"]:
        print(f"  (i) {g['kutipan_terlalu_pendek_dilewati']} kutipan <{_MIN_CHAR_KUTIPAN} char "
              f"dilewati — terlalu pendek untuk dinilai.")

    rz = h["recall_zona"]
    print("\n  recall zona per poin:")
    for pid, e in rz["per_poin"].items():
        print(f"    {pid:11s} {_pct(e):>12s}  (n={e['n']}, kena={e['kena']})")
    if rz["dilewati"]:
        print("  yang tak bisa dinilai:")
        for k, v in rz["dilewati"].items():
            print(f"    {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Metrik atribusi sitasi dari log precheck + korpus.")
    ap.add_argument("--log", type=Path, default=_LOG)
    ap.add_argument("--out", type=Path, default=_OUT)
    ap.add_argument("--sejak", default=None,
                    help="hanya permohonan sejak tanggal ini (ISO, mis. 2026-09-07) — pakai "
                         "ini saat angkanya dikutip, supaya tak mencampur versi kode")
    args = ap.parse_args()

    if not args.log.exists():
        raise SystemExit(f"log tidak ada: {args.log}")

    permohonan, lewat = muat_permohonan(args.log, sejak=args.sejak)
    id_disitasi = {s.citation_id for p in permohonan for s in p.sitasi if s.jenis == "korpus"}
    chunks = _muat_chunks(id_disitasi)
    hasil = hitung(permohonan, chunks)
    hasil["berkas_log"] = args.log.name
    hasil["sejak"] = args.sejak
    hasil["dilewati"] = lewat

    cetak(hasil, lewat)
    args.out.write_text(json.dumps(hasil, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[atribusi] -> {args.out}")


if __name__ == "__main__":
    main()
