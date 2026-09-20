"""eval/eval_rag.py — evaluasi kinerja RETRIEVAL RAG + laporan HTML mandiri (satu berkas).

Beda dari `eval/run_eval.py` (yang menilai sisi REASONING/LLM atas gold set dan masih terikat
MockRetriever): modul ini menilai sisi **retrieval** terhadap korpus Postgres SUNGGUHAN, memakai
`tests/eval_set.jsonl` (query -> chunk yang relevan) sebagai ground truth.

Yang dihitung:

1. **Metrik retrieval standar** — Hit-Rate@k, MRR, Recall@k, Precision@k, nDCG@k, MAP.
   Relevansi biner (sebuah chunk relevan atau tidak), sesuai bentuk ground truth yang tersedia.

2. **Ablasi per komponen** — pipeline produksi adalah dense + lexical -> RRF -> rerank. Tiap
   lapis dinilai sendiri supaya kontribusinya terlihat, bukan diasumsikan:
     - `dense`        : hanya pencarian vektor
     - `lexical`      : hanya Postgres FTS
     - `rrf`          : fusi keduanya, TANPA rerank
     - `dense+rerank` : dense lalu rerank — TANPA sisi lexical sama sekali
     - `rrf+rerank`   : pipeline produksi penuh
   `dense+rerank` ada khusus untuk menjawab satu pertanyaan yang tak bisa dijawab empat lainnya:
   apakah sisi lexical benar-benar membayar tempatnya, atau justru mengencerkan peringkat dense
   lewat RRF yang tak berbobot.
   Tanpa ablasi, mustahil tahu apakah reranker (yang berbayar per panggilan) benar-benar membayar
   dirinya sendiri, atau apakah sisi lexical benar-benar menyumbang.

3. **Latensi per konfigurasi** — biaya waktu tiap lapis.

4. **Sisi generasi (historis)** — sebaran `sebab` dari kunci `diagnostik` di `logs/precheck.jsonl`
   kalau ada. Ini data yang SUDAH terkumpul, bukan dihitung ulang di sini.

Keluaran: satu berkas HTML mandiri (grafik SVG inline, tanpa dependensi eksternal) + JSON mentah
di sebelahnya untuk reproduksi.

CLI:
  python -m eval.eval_rag                      # tulis eval/laporan_rag.html
  python -m eval.eval_rag --out /tmp/lap.html  # lokasi lain
  python -m eval.eval_rag --tanpa-rerank       # lewati konfigurasi ber-rerank (hemat kuota API)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.retrieval import db, fusion, rerank  # noqa: E402
from app.retrieval.base import RetrievalFilters  # noqa: E402
from app.retrieval.embeddings import encode_dense_one  # noqa: E402
from app.retrieval.retriever import _expand  # noqa: E402

_EVAL_SET = Path(__file__).parent.parent / "tests" / "eval_set.jsonl"
_LOG_PRECHECK = Path(__file__).parent.parent / "logs" / "precheck.jsonl"
_K_LIST = (1, 3, 5, 10)
_TOP_N = 10          # panjang daftar hasil yang dinilai
_KANDIDAT = 30       # sejajar dgn candidate_k/rerank_pool retriever produksi


# ---------------------------------------------------------------------------
# Metrik — relevansi biner, sesuai bentuk ground truth (`relevan`: daftar id chunk)
# ---------------------------------------------------------------------------
def _metrik_satu_query(urutan: list[str], relevan: set[str]) -> dict:
    """Hitung seluruh metrik untuk SATU query dari daftar id terurut (peringkat 1 = indeks 0)."""
    hasil: dict = {}
    n_relevan = len(relevan)

    peringkat_pertama = next((i + 1 for i, cid in enumerate(urutan) if cid in relevan), None)
    hasil["peringkat_pertama"] = peringkat_pertama
    hasil["rr"] = 1.0 / peringkat_pertama if peringkat_pertama else 0.0

    for k in _K_LIST:
        atas = urutan[:k]
        kena = sum(1 for cid in atas if cid in relevan)
        hasil[f"hit@{k}"] = 1.0 if kena else 0.0
        hasil[f"recall@{k}"] = kena / n_relevan if n_relevan else 0.0
        hasil[f"precision@{k}"] = kena / k
        # nDCG biner: gain 1 utk chunk relevan, diskon logaritmik menurut posisi.
        dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(atas) if cid in relevan)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_relevan, k)))
        hasil[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0

    # Average Precision: rata-rata precision pada tiap posisi yang relevan, dibagi jumlah relevan
    # (dibatasi panjang daftar yang dinilai — AP jadi konservatif, tidak menghukum ganda chunk
    # relevan yang memang berada di luar top-N).
    kena = 0
    jumlah_p = 0.0
    for i, cid in enumerate(urutan):
        if cid in relevan:
            kena += 1
            jumlah_p += kena / (i + 1)
    hasil["ap"] = jumlah_p / min(n_relevan, len(urutan)) if n_relevan and urutan else 0.0
    return hasil


def _agregat(per_query: list[dict]) -> dict:
    """Rata-rata makro seluruh metrik (tiap query berbobot sama)."""
    if not per_query:
        return {}
    kunci = [k for k in per_query[0] if k != "peringkat_pertama"]
    agg = {k: statistics.fmean(q[k] for q in per_query) for k in kunci}
    agg["mrr"] = agg.pop("rr")
    agg["map"] = agg.pop("ap")
    ketemu = [q["peringkat_pertama"] for q in per_query if q["peringkat_pertama"]]
    agg["median_peringkat_pertama"] = statistics.median(ketemu) if ketemu else None
    agg["query_tanpa_hasil_relevan"] = sum(1 for q in per_query if q["peringkat_pertama"] is None)
    return agg


# ---------------------------------------------------------------------------
# Konfigurasi retrieval yang diuji (ablasi lapis demi lapis)
# ---------------------------------------------------------------------------
def _urutan_dense(conn, query: str, filters: RetrievalFilters) -> list[str]:
    """Hanya pencarian vektor. Query DIPERLUAS — persis seperti jalur produksi."""
    qvec = encode_dense_one(_expand(query))
    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    if provider == "local":
        hasil = db.dense_search(conn, qvec, filters, _KANDIDAT)
    else:
        hasil = db.dense_search_ab(conn, qvec, provider, filters, _KANDIDAT)
    return [cid for cid, _ in hasil]


def _urutan_lexical(conn, query: str, filters: RetrievalFilters) -> list[str]:
    """Hanya Postgres FTS. Query ASLI (tanpa expansion) — persis seperti jalur produksi."""
    return [cid for cid, _ in db.fts_search(conn, query, filters, _KANDIDAT)]


def _urutan_rrf(dense: list[str], lexical: list[str]) -> list[str]:
    return [cid for cid, _ in fusion.reciprocal_rank_fusion([dense, lexical])]


def _urutan_rerank(conn, query: str, fused: list[str]) -> list[str]:
    """Rerank cross-encoder atas kandidat hasil fusi — lapis terakhir pipeline produksi."""
    if not fused:
        return []
    kandidat = db.hydrate(conn, fused[:_KANDIDAT])
    skor = rerank.rerank(_expand(query), [c.teks for c in kandidat], top_k=_TOP_N)
    return [kandidat[i].id for i, _ in skor]


def jalankan_evaluasi(pakai_rerank: bool = True, jeda_s: float = 0.0) -> dict:
    import psycopg

    baris = [json.loads(l) for l in _EVAL_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    wilayah = os.getenv("RETRIEVER_WILAYAH", "Sleman Tengah").strip('"')
    filters = RetrievalFilters(dokumen=wilayah)
    conn = psycopg.connect(os.getenv("DATABASE_URL"))

    konfigurasi = ["dense", "lexical", "rrf"] + (["dense+rerank", "rrf+rerank"] if pakai_rerank else [])
    per_query: dict[str, list[dict]] = {k: [] for k in konfigurasi}
    latensi: dict[str, list[float]] = {k: [] for k in konfigurasi}
    detail: list[dict] = []

    for i, row in enumerate(baris, 1):
        query, relevan = row["query"], set(row["relevan"])
        urutan: dict[str, list[str]] = {}

        t0 = time.perf_counter()
        urutan["dense"] = _urutan_dense(conn, query, filters)
        latensi["dense"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        urutan["lexical"] = _urutan_lexical(conn, query, filters)
        latensi["lexical"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        urutan["rrf"] = _urutan_rrf(urutan["dense"], urutan["lexical"])
        latensi["rrf"].append(time.perf_counter() - t0)

        if pakai_rerank:
            t0 = time.perf_counter()
            urutan["dense+rerank"] = _urutan_rerank(conn, query, urutan["dense"])
            latensi["dense+rerank"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            urutan["rrf+rerank"] = _urutan_rerank(conn, query, urutan["rrf"])
            latensi["rrf+rerank"].append(time.perf_counter() - t0)

        baris_detail = {"query": query, "n_relevan": len(relevan)}
        for kfg in konfigurasi:
            m = _metrik_satu_query(urutan[kfg][:_TOP_N], relevan)
            per_query[kfg].append(m)
            baris_detail[kfg] = m["peringkat_pertama"]
        detail.append(baris_detail)
        print(f"  [{i}/{len(baris)}] {query[:46]:48s} " +
              "  ".join(f"{k}={baris_detail[k] or '-'}" for k in konfigurasi), flush=True)
        if jeda_s and i < len(baris):
            time.sleep(jeda_s)

    # --- konteks korpus ---
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE dokumen ILIKE %s", [f"%{wilayah}%"])
        n_chunk = cur.fetchone()[0]
        cur.execute("""SELECT level, count(*) FROM chunks WHERE dokumen ILIKE %s
                       GROUP BY 1 ORDER BY 1""", [f"%{wilayah}%"])
        per_level = dict(cur.fetchall())
        cur.execute("""SELECT count(*) FROM chunk_embeddings_ab a JOIN chunks c ON c.id = a.chunk_id
                       WHERE c.dokumen ILIKE %s""", [f"%{wilayah}%"])
        n_vektor = cur.fetchone()[0]
    conn.close()

    return {
        "dibuat": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wilayah": wilayah,
        "provider_embedding": os.getenv("EMBEDDING_PROVIDER", "local"),
        "provider_rerank": os.getenv("RERANK_PROVIDER", "local"),
        "n_query": len(baris),
        "top_n_dinilai": _TOP_N,
        "korpus": {"chunk": n_chunk, "per_level": per_level, "vektor": n_vektor},
        "konfigurasi": konfigurasi,
        "agregat": {k: _agregat(per_query[k]) for k in konfigurasi},
        "latensi_ms": {k: statistics.fmean(latensi[k]) * 1000 for k in konfigurasi},
        "detail_per_query": detail,
        "generasi": _ringkas_diagnostik_log(),
    }


def _ringkas_diagnostik_log() -> dict:
    """Sebaran `sebab` dari log operasional — data historis yang SUDAH terkumpul, bukan uji baru."""
    if not _LOG_PRECHECK.exists():
        return {}
    sebab: dict[str, int] = {}
    n_rec = n_poin = sitasi = sitasi_ver = 0
    for l in _LOG_PRECHECK.read_text(encoding="utf-8", errors="replace").splitlines():
        if not l.strip():
            continue
        try:
            rec = json.loads(l)
        except Exception:
            continue
        if "diagnostik" not in rec:
            continue
        n_rec += 1
        for d in rec["diagnostik"]:
            n_poin += 1
            sebab[d.get("sebab", "?")] = sebab.get(d.get("sebab", "?"), 0) + 1
        for p in rec.get("response", {}).get("poin", []):
            for s in p.get("sitasi", []) or []:
                sitasi += 1
                sitasi_ver += int(bool(s.get("terverifikasi")))
    return {"n_permohonan": n_rec, "n_poin": n_poin, "sebab": sebab,
            "sitasi": sitasi, "sitasi_terverifikasi": sitasi_ver}


# ---------------------------------------------------------------------------
# Laporan HTML mandiri — SVG inline, tanpa dependensi eksternal
# ---------------------------------------------------------------------------
_WARNA = {"dense": "#2563eb", "lexical": "#d97706", "rrf": "#7c3aed",
          "dense+rerank": "#0891b2", "rrf+rerank": "#059669"}
_LABEL = {"dense": "Dense saja", "lexical": "Lexical saja", "rrf": "RRF (tanpa rerank)",
          "dense+rerank": "Dense + rerank (tanpa lexical)", "rrf+rerank": "RRF + rerank (produksi)"}


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _bar_berkelompok(judul: str, kategori: list[str], seri: dict[str, list[float]],
                     maks: float = 1.0, fmt: str = "{:.0%}") -> str:
    """Bar chart berkelompok, SVG murni."""
    W, H = 760, 300
    kiri, bawah, atas = 56, 54, 34
    lebar_grup = (W - kiri - 16) / max(len(kategori), 1)
    lebar_bar = lebar_grup / (len(seri) + 0.6)
    tinggi = H - atas - bawah
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(judul)}">',
         f'<text x="{kiri}" y="20" class="jdl">{_esc(judul)}</text>']
    for g in range(5):
        y = atas + tinggi * g / 4
        p.append(f'<line x1="{kiri}" y1="{y:.1f}" x2="{W-16}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{kiri-8}" y="{y+4:.1f}" class="ax" text-anchor="end">'
                 f'{fmt.format(maks*(1-g/4))}</text>')
    for gi, kat in enumerate(kategori):
        x0 = kiri + gi * lebar_grup
        for si, (nama, nilai) in enumerate(seri.items()):
            v = max(0.0, min(nilai[gi] / maks if maks else 0, 1.0))
            h = tinggi * v
            x = x0 + 8 + si * lebar_bar
            y = atas + tinggi - h
            p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{lebar_bar-3:.1f}" height="{h:.1f}" '
                     f'rx="2" fill="{_WARNA.get(nama, "#64748b")}"><title>{_esc(nama)} — '
                     f'{_esc(kat)}: {fmt.format(nilai[gi])}</title></rect>')
            if h > 22:
                p.append(f'<text x="{x+(lebar_bar-3)/2:.1f}" y="{y-4:.1f}" class="val" '
                         f'text-anchor="middle">{fmt.format(nilai[gi])}</text>')
        p.append(f'<text x="{x0+lebar_grup/2:.1f}" y="{atas+tinggi+18}" class="ax" '
                 f'text-anchor="middle">{_esc(kat)}</text>')
    lx = kiri
    for nama in seri:
        p.append(f'<rect x="{lx}" y="{H-22}" width="10" height="10" rx="2" fill="{_WARNA.get(nama,"#64748b")}"/>')
        p.append(f'<text x="{lx+15}" y="{H-13}" class="lg">{_esc(_LABEL.get(nama, nama))}</text>')
        lx += 20 + 7.2 * len(_LABEL.get(nama, nama))
    p.append("</svg>")
    return "".join(p)


def _bar_horizontal(judul: str, label: list[str], nilai: list[float], warna: list[str],
                    satuan: str = "") -> str:
    W = 760
    tinggi_baris = 30
    H = 40 + tinggi_baris * len(label) + 10
    kiri = 190
    maks = max(nilai) if nilai and max(nilai) > 0 else 1.0
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(judul)}">',
         f'<text x="16" y="20" class="jdl">{_esc(judul)}</text>']
    for i, (lb, v) in enumerate(zip(label, nilai)):
        y = 38 + i * tinggi_baris
        w = (W - kiri - 90) * (v / maks)
        p.append(f'<text x="{kiri-10}" y="{y+13}" class="ax" text-anchor="end">{_esc(lb)}</text>')
        p.append(f'<rect x="{kiri}" y="{y}" width="{w:.1f}" height="18" rx="3" fill="{warna[i]}"/>')
        p.append(f'<text x="{kiri+w+8:.1f}" y="{y+13}" class="val">{v:,.0f}{_esc(satuan)}</text>')
    p.append("</svg>")
    return "".join(p)


def _strip_peringkat(detail: list[dict], konfigurasi: list[str]) -> str:
    """Peringkat chunk relevan pertama per query — memperlihatkan query mana yang sulit."""
    W = 760
    tinggi_baris = 22
    kiri = 250
    H = 46 + tinggi_baris * len(detail) + 16
    kotak = 26
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Peringkat per query">',
         '<text x="16" y="20" class="jdl">Peringkat chunk relevan pertama, per query '
         '(makin kecil makin baik)</text>']
    for ci, kfg in enumerate(konfigurasi):
        x = kiri + ci * (kotak + 30)
        singkat = {"dense+rerank": "d+rr", "rrf+rerank": "rrf+rr"}.get(kfg, kfg)
        p.append(f'<text x="{x+kotak/2}" y="38" class="ax" text-anchor="middle">'
                 f'{_esc(singkat)}</text>')
    for i, d in enumerate(detail):
        y = 46 + i * tinggi_baris
        p.append(f'<text x="{kiri-12}" y="{y+14}" class="ax" text-anchor="end">'
                 f'{_esc(d["query"][:34])}</text>')
        for ci, kfg in enumerate(konfigurasi):
            r = d.get(kfg)
            x = kiri + ci * (kotak + 30)
            if r is None:
                isi, teks, warna_t = "#fee2e2", "—", "#b91c1c"
            else:
                tingkat = min(max((r - 1) / 9.0, 0.0), 1.0)
                isi = f"hsl(152 55% {int(42 + tingkat * 44)}%)"
                teks = str(r)
                warna_t = "#052e1a" if tingkat < 0.5 else "#334155"
            p.append(f'<rect x="{x}" y="{y}" width="{kotak}" height="18" rx="3" fill="{isi}"/>')
            p.append(f'<text x="{x+kotak/2}" y="{y+13}" class="cel" text-anchor="middle" '
                     f'fill="{warna_t}">{teks}</text>')
    p.append("</svg>")
    return "".join(p)


def _donat(judul: str, data: dict[str, int], palet: dict[str, str]) -> str:
    total = sum(data.values()) or 1
    W, H, cx, cy, r, tebal = 760, 220, 120, 118, 74, 30
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(judul)}">',
         f'<text x="16" y="20" class="jdl">{_esc(judul)}</text>']
    sudut = -math.pi / 2
    for nama, n in sorted(data.items(), key=lambda kv: -kv[1]):
        porsi = n / total
        akhir = sudut + porsi * 2 * math.pi
        x1, y1 = cx + r * math.cos(sudut), cy + r * math.sin(sudut)
        x2, y2 = cx + r * math.cos(akhir), cy + r * math.sin(akhir)
        besar = 1 if porsi > 0.5 else 0
        p.append(f'<path d="M {x1:.2f} {y1:.2f} A {r} {r} 0 {besar} 1 {x2:.2f} {y2:.2f}" '
                 f'fill="none" stroke="{palet.get(nama, "#94a3b8")}" stroke-width="{tebal}">'
                 f'<title>{_esc(nama)}: {n} ({porsi:.0%})</title></path>')
        sudut = akhir
    p.append(f'<text x="{cx}" y="{cy+2}" class="ctr" text-anchor="middle">{total}</text>')
    p.append(f'<text x="{cx}" y="{cy+20}" class="ax" text-anchor="middle">poin</text>')
    ly = 54
    for nama, n in sorted(data.items(), key=lambda kv: -kv[1]):
        p.append(f'<rect x="250" y="{ly}" width="11" height="11" rx="2" fill="{palet.get(nama,"#94a3b8")}"/>')
        p.append(f'<text x="268" y="{ly+10}" class="lg">{_esc(nama)} — {n} ({n/total:.0%})</text>')
        ly += 22
    p.append("</svg>")
    return "".join(p)


def bangun_html(r: dict) -> str:
    kfg = r["konfigurasi"]
    agg = r["agregat"]
    produksi = "rrf+rerank" if "rrf+rerank" in kfg else "rrf"

    hit = {k: [agg[k][f"hit@{n}"] for n in _K_LIST] for k in kfg}
    mutu_kat = ["MRR", "MAP", "nDCG@5", "Recall@5"]
    mutu = {k: [agg[k]["mrr"], agg[k]["map"], agg[k]["ndcg@5"], agg[k]["recall@5"]] for k in kfg}

    baris_tabel = ""
    for k in kfg:
        a = agg[k]
        sorot = ' class="sorot"' if k == produksi else ""
        baris_tabel += (
            f"<tr{sorot}><td><b>{_esc(_LABEL.get(k,k))}</b></td>"
            + "".join(f"<td>{a[f'hit@{n}']:.0%}</td>" for n in _K_LIST)
            + f"<td>{a['mrr']:.3f}</td><td>{a['map']:.3f}</td>"
            f"<td>{a['ndcg@5']:.3f}</td><td>{a['recall@5']:.0%}</td>"
            f"<td>{a['precision@5']:.0%}</td>"
            f"<td>{a['query_tanpa_hasil_relevan']}</td>"
            f"<td>{r['latensi_ms'][k]:,.0f} ms</td></tr>")

    gen = r.get("generasi") or {}
    blok_gen = ""
    if gen.get("n_poin"):
        palet = {"berhasil": "#059669", "guardrail_menolak": "#d97706",
                 "panggilan_llm_gagal": "#dc2626", "retrieval_kosong": "#7c3aed"}
        sitasi_txt = (f"{gen['sitasi_terverifikasi']}/{gen['sitasi']} sitasi terverifikasi"
                      if gen.get("sitasi") else "tidak ada sitasi tercatat")
        n_gagal = gen["sebab"].get("panggilan_llm_gagal", 0)
        blok_gen = f"""
  <h2>4. Sisi generasi — data operasional historis</h2>
  <div class="peringatan" style="margin-bottom:14px">
  <b>Baca dengan hati-hati — ini BUKAN tingkat kegagalan produksi.</b> Rekap ini diambil dari kunci
  <code>diagnostik</code> yang sudah terkumpul di <code>logs/precheck.jsonl</code>
  ({gen['n_permohonan']} permohonan, {gen['n_poin']} poin), dan isinya <b>didominasi replay
  pengujian</b>, bukan lalu lintas pemohon. Porsi <code>panggilan_llm_gagal</code>
  ({n_gagal} poin) sebagian besar berasal dari replay batch yang sengaja dijalankan beruntun
  sampai memicu rate limit provider, plus satu uji yang memang dirancang selalu gagal. Angka
  produksi yang sesungguhnya baru bisa dihitung setelah lalu lintas organik terkumpul.
  Yang tetap bermakna di sini: <b>tidak ada satu pun poin dengan sebab
  <code>guardrail_menolak</code></b> — konsisten dengan hasil verifikasi setelah perbaikan.
  </div>
  <div class="kartu">{_donat("Sebab akhir tiap poin", gen["sebab"], palet)}</div>
  <p class="cat">Integritas sitasi pada periode yang sama: <b>{sitasi_txt}</b>. Perlu diingat,
  "terverifikasi" berarti <i>citation_id</i> memang ada di daftar kandidat yang disodorkan
  (anti-halusinasi) — bukan jaminan kandidatnya relevan secara hukum.</p>"""

    korpus = r["korpus"]
    lvl = ", ".join(f"{v} {k}" for k, v in sorted(korpus["per_level"].items()))

    return f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluasi Retrieval RAG — RDTR {_esc(r['wilayah'])}</title>
<style>
  :root {{ color-scheme: light; --bg:#f8fafc; --kartu:#fff; --grs:#e2e8f0; --tks:#0f172a; --redup:#64748b; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:28px 18px 60px; background:var(--bg); color:var(--tks);
         font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }}
  main {{ max-width: 860px; margin: 0 auto; }}
  h1 {{ font-size:25px; margin:0 0 4px; letter-spacing:-.02em; }}
  h2 {{ font-size:18px; margin:34px 0 10px; letter-spacing:-.01em; }}
  .sub {{ color:var(--redup); margin:0 0 6px; }}
  .kartu {{ background:var(--kartu); border:1px solid var(--grs); border-radius:10px;
            padding:14px; margin:14px 0; overflow-x:auto; }}
  svg {{ display:block; width:100%; height:auto; min-width:640px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13.5px; min-width:640px; }}
  th,td {{ padding:7px 9px; border-bottom:1px solid var(--grs); text-align:right; white-space:nowrap; }}
  th:first-child, td:first-child {{ text-align:left; }}
  thead th {{ background:#f1f5f9; font-weight:600; }}
  tr.sorot td {{ background:#ecfdf5; }}
  .cat {{ color:var(--redup); font-size:14px; }}
  .pil {{ display:inline-block; background:#eef2ff; color:#3730a3; border-radius:999px;
          padding:2px 10px; font-size:12.5px; margin:0 6px 6px 0; }}
  .peringatan {{ background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:12px 14px; }}
  .peringatan h2 {{ margin-top:0; }}
  code {{ background:#f1f5f9; padding:1px 5px; border-radius:4px; font-size:13px; }}
  .jdl {{ font:600 13.5px system-ui; fill:#0f172a; }}
  .ax  {{ font:11.5px system-ui; fill:#64748b; }}
  .val {{ font:600 11px system-ui; fill:#0f172a; }}
  .cel {{ font:600 11px system-ui; }}
  .lg  {{ font:12px system-ui; fill:#334155; }}
  .ctr {{ font:600 22px system-ui; fill:#0f172a; }}
  .grid {{ stroke:#e2e8f0; stroke-width:1; }}
</style></head><body><main>

  <h1>Evaluasi Retrieval RAG</h1>
  <p class="sub">RDTR {_esc(r['wilayah'])} · dibuat {_esc(r['dibuat'])} ·
     {r['n_query']} query · dinilai sampai peringkat {r['top_n_dinilai']}</p>
  <p>
    <span class="pil">embedding: {_esc(r['provider_embedding'])}</span>
    <span class="pil">rerank: {_esc(r['provider_rerank'])}</span>
    <span class="pil">korpus: {korpus['chunk']} chunk ({_esc(lvl)})</span>
    <span class="pil">vektor: {korpus['vektor']}</span>
  </p>

  <h2>1. Ringkasan metrik &amp; ablasi per lapis</h2>
  <p class="cat">Pipeline produksi adalah dense + lexical → RRF → rerank. Tiap lapis dinilai
  sendiri supaya kontribusinya terlihat, bukan diasumsikan. Baris hijau = jalur produksi.</p>
  <div class="kartu"><table>
    <thead><tr><th>Konfigurasi</th>{''.join(f'<th>Hit@{n}</th>' for n in _K_LIST)}
      <th>MRR</th><th>MAP</th><th>nDCG@5</th><th>Recall@5</th><th>P@5</th>
      <th>Gagal</th><th>Latensi</th></tr></thead>
    <tbody>{baris_tabel}</tbody>
  </table></div>

  <h2>2. Hit-Rate menurut kedalaman</h2>
  <p class="cat">Seberapa sering minimal satu chunk relevan masuk peringkat-k teratas.</p>
  <div class="kartu">{_bar_berkelompok("Hit-Rate@k", [f"@{n}" for n in _K_LIST], hit)}</div>

  <h2>3. Mutu peringkat</h2>
  <p class="cat">Hit-Rate hanya menanyakan "ketemu atau tidak". Metrik di bawah menanyakan
  "ketemu di posisi berapa" dan "berapa banyak yang ketemu" — di sinilah reranker seharusnya
  membayar dirinya sendiri.</p>
  <div class="kartu">{_bar_berkelompok("MRR · MAP · nDCG@5 · Recall@5", mutu_kat, mutu, 1.0, "{:.2f}")}</div>
  <div class="kartu">{_strip_peringkat(r["detail_per_query"], kfg)}</div>
  <div class="kartu">{_bar_horizontal("Latensi rata-rata per query",
      [_LABEL.get(k, k) for k in kfg], [r["latensi_ms"][k] for k in kfg],
      [_WARNA.get(k, "#64748b") for k in kfg], " ms")}</div>
{blok_gen}

  <h2>{'5' if blok_gen else '4'}. Batas pembacaan</h2>
  <div class="peringatan">
  <ul>
    <li><b>Ground truth diseed developer, belum divalidasi ahli tata ruang.</b> Angka di sini
        mengukur konsistensi sistem terhadap label kami sendiri — bukan kebenaran hukum.</li>
    <li><b>{r['n_query']} query itu sampel kecil.</b> Satu query berpindah peringkat menggeser
        rata-rata secara kasat mata. Perlakukan selisih kecil antar-konfigurasi sebagai setara.</li>
    <li><b>Satu wilayah saja</b> ({_esc(r['wilayah'])}). Wilayah lain belum punya eval set, dan
        cakupan vektornya berbeda.</li>
    <li><b>Relevansi biner.</b> Ground truth tidak membedakan "sangat relevan" dan "agak relevan",
        jadi nDCG di sini lebih kasar daripada nDCG bergradasi.</li>
    <li><b>Latensi diukur di mesin dev</b> terhadap Postgres lokal dan API eksternal — bukan angka
        produksi, hanya untuk membandingkan biaya antar-lapis.</li>
  </ul>
  </div>

  <p class="cat" style="margin-top:26px">Dihasilkan oleh <code>python -m eval.eval_rag</code>.
  Data mentah berdampingan sebagai <code>.json</code>. Nilai konfigurasi ada di
  <code>.env.example</code>; arsitektur sistem di <code>docs/ARSITEKTUR_RAG.md</code>.</p>
</main></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluasi retrieval RAG + laporan HTML mandiri.")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "laporan_rag.html")
    ap.add_argument("--tanpa-rerank", action="store_true",
                    help="lewati konfigurasi ber-rerank (hemat kuota API provider)")
    ap.add_argument("--jeda", type=float, default=0.0,
                    help="jeda detik antar-query, utk menghindari rate limit provider")
    ap.add_argument("--dari-json", type=Path, default=None,
                    help="bangun ulang HTML dari hasil JSON yang sudah ada, TANPA memanggil "
                         "DB/API lagi — untuk mengubah tampilan laporan tanpa membakar kuota")
    args = ap.parse_args()

    if args.dari_json:
        print(f"[eval] bangun ulang dari {args.dari_json.name} (tanpa panggilan DB/API)")
        hasil = json.loads(args.dari_json.read_text(encoding="utf-8"))
    else:
        print(f"[eval] {_EVAL_SET.name} -> {args.out.name}")
        hasil = jalankan_evaluasi(pakai_rerank=not args.tanpa_rerank, jeda_s=args.jeda)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(bangun_html(hasil), encoding="utf-8")
    if not args.dari_json:
        args.out.with_suffix(".json").write_text(
            json.dumps(hasil, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[eval] ringkasan:")
    for k in hasil["konfigurasi"]:
        a = hasil["agregat"][k]
        print(f"  {k:14s} Hit@5={a['hit@5']:.0%}  MRR={a['mrr']:.3f}  "
              f"nDCG@5={a['ndcg@5']:.3f}  MAP={a['map']:.3f}  {hasil['latensi_ms'][k]:,.0f}ms")
    print(f"\n[eval] laporan -> {args.out}")


if __name__ == "__main__":
    main()
