"""eval/laporan_bukti.py — laporan evaluasi LENGKAP dengan bukti per-query yang bisa diperiksa.

Kenapa modul ini ada, terpisah dari `eval/eval_rag.py`. Laporan teknis menyajikan angka agregat,
ablasi, dan uji statistik — tapi pembacanya tak punya cara memeriksa klaim itu selain menjalankan
ulang seluruh evaluasi. Modul ini menutup celah tersebut: untuk SETIAP topik uji ia menampilkan
query yang benar-benar diterbitkan, filter yang menyertainya, chunk yang dilabeli benar, chunk yang
BENAR-BENAR terambil di tiap konfigurasi, dan metrik yang dihasilkan. Dengan begitu tiap angka
agregat bisa ditelusuri sampai ke kasusnya satu per satu.

Bukti mentahnya (`detail_per_query[*].terambil`) direkam oleh `eval_rag.py` saat evaluasi berjalan.
Modul ini TIDAK menghitung ulang apa pun dan tidak memanggil API retrieval; satu-satunya sentuhan
DB adalah mengambil keterangan singkat tiap chunk, supaya id seperti `rdtr-sleman-tengah-vi-r-2`
terbaca sebagai "Lampiran VI · Zona R-2 · tabel" dan pembaca tak perlu membuka basis data sendiri.

CLI:
  python -m eval.laporan_bukti
  python -m eval.laporan_bukti --out eval/laporan_bukti.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_DIR = Path(__file__).parent
_OUT = _DIR / "laporan_bukti.html"

# Definisi metrik: rumus singkat + arti + perilaku langit-langitnya. Ditaruh di laporan supaya
# pembaca tak perlu menebak apa yang diukur, dan supaya batas tiap metrik ikut terbaca.
_DEFINISI = (
    ("Hit@k", "1 bila ada chunk relevan di k teratas, selain itu 0",
     "Hanya menjawab \"apakah retrieval gagal total\". Naik dengan sendirinya bila k diperbesar, "
     "jadi Hit@10 tinggi bukan bukti mutu.", "Selalu bisa 1.0"),
    ("Recall@k", "jumlah relevan di k teratas ÷ seluruh relevan",
     "Berapa bagian chunk otoritatif yang sampai ke LLM.",
     "Dibatasi min(k,R)/R — turun di bawah 1.0 bila satu topik punya lebih dari k chunk relevan"),
    ("Precision@k", "jumlah relevan di k teratas ÷ k",
     "Berapa bagian slot yang terpakai berguna.",
     "Dibatasi min(k,R)/k — jauh di bawah 1.0 bila topik punya lebih sedikit relevan daripada k"),
    ("nDCG@k", "DCG ÷ DCG ideal, dgn diskon logaritmik menurut posisi",
     "Metrik primer: menggabungkan BERAPA BANYAK relevan yang ketemu dan DI POSISI MANA.",
     "Selalu bisa 1.0 — idealnya ikut dibatasi min(R,k)"),
    ("MRR", "rata-rata 1 ÷ peringkat chunk relevan pertama",
     "Seberapa cepat jawaban benar muncul. Buta terhadap relevan selain yang pertama.",
     "Selalu bisa 1.0"),
    ("MAP", "rata-rata precision pada tiap posisi relevan",
     "Menghargai sistem yang menaruh SELURUH relevan di atas, bukan hanya satu.",
     "Selalu bisa 1.0"),
)


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _muat(nama: str) -> dict | None:
    f = _DIR / nama
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def _glosarium(ids: set[str]) -> dict[str, dict]:
    """id chunk -> keterangan singkat. Tanpa DB, laporan tetap jadi — hanya id mentah yang tampil."""
    if not ids:
        return {}
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("[bukti] DATABASE_URL tak ada; id chunk ditampilkan tanpa keterangan")
        return {}
    try:
        import psycopg

        with psycopg.connect(dsn) as conn:
            baris = conn.execute(
                "SELECT id, level, zona, pasal, ayat, left(replace(teks, chr(10), ' '), 150) "
                "FROM chunks WHERE id = ANY(%s)", (sorted(ids),)).fetchall()
        return {r[0]: {"level": r[1], "zona": r[2], "pasal": r[3], "ayat": r[4], "teks": r[5]}
                for r in baris}
    except Exception as exc:
        print(f"[bukti] DB tak terjangkau ({exc}); id chunk ditampilkan tanpa keterangan")
        return {}


def _label_chunk(cid: str, glo: dict) -> str:
    g = glo.get(cid)
    if not g:
        return ""
    bagian = []
    if g.get("pasal"):
        bagian.append(f"Pasal {g['pasal']}" + (f" ayat {g['ayat']}" if g.get("ayat") else ""))
    if g.get("zona"):
        bagian.append(f"Zona {g['zona']}")
    if g.get("level"):
        bagian.append(g["level"])
    return " · ".join(bagian)


def _sel_terambil(ids: list[str], relevan: set[str], glo: dict) -> str:
    """Daftar chunk terambil, ditandai mana yang cocok label. Inilah bukti yang bisa diperiksa."""
    if not ids:
        return '<span class="redup">(kosong)</span>'
    keluar = []
    for n, cid in enumerate(ids, 1):
        kena = cid in relevan
        ket = _label_chunk(cid, glo)
        keluar.append(
            f'<div class="item {"kena" if kena else ""}">'
            f'<span class="rank">{n}</span>'
            f'<code title="{_e(ket)}">{_e(cid)}</code>'
            + (f'<span class="ket">{_e(ket)}</span>' if ket else "")
            + ('<span class="cek">cocok label</span>' if kena else "")
            + "</div>")
    return "".join(keluar)


def _kartu_topik(d: dict, pq: dict, konfigurasi: list[str], glo: dict, k_op: int,
                 label_kfg: dict) -> str:
    relevan = set(d.get("relevan") or [])
    terambil = d.get("terambil") or {}
    baris = ""
    for i, kfg in enumerate(konfigurasi):
        skor = (pq.get(kfg) or [])
        m = skor[d["_idx"]] if d["_idx"] < len(skor) else {}
        peringkat = d.get(kfg)
        baris += (
            f'<tr><td class="kfg">{_e(label_kfg.get(kfg, kfg))}</td>'
            f'<td class="hasil">{_sel_terambil(terambil.get(kfg) or [], relevan, glo)}</td>'
            f'<td class="num">{m.get(f"ndcg@{k_op}", 0):.3f}</td>'
            f'<td class="num">{m.get(f"recall@{k_op}", 0):.3f}</td>'
            f'<td class="num">{m.get(f"hit@{k_op}", 0):.0f}</td>'
            f'<td class="num">{m.get("mrr", 0):.3f}</td>'
            f'<td class="num">{peringkat if peringkat else "&mdash;"}</td></tr>')

    filt = d.get("filter") or {}
    filt_txt = ", ".join(f"{k}={v}" for k, v in filt.items()) or "(tanpa filter zona)"
    lbl = "".join(f'<div class="item kena"><code title="{_e(_label_chunk(c, glo))}">{_e(c)}</code>'
                  f'<span class="ket">{_e(_label_chunk(c, glo))}</span></div>'
                  for c in sorted(relevan))
    gagal = not d.get(konfigurasi[-1]) if konfigurasi else False
    return f"""
<article class="topik" data-poin="{_e(d.get('_poin', ''))}" data-jalur="{_e(d.get('_jalur', ''))}"
         data-label="{_e(d.get('sumber_label', ''))}" data-gagal="{'1' if gagal else '0'}"
         data-cari="{_e((d['query'] + ' ' + (d.get('teks_query') or '')).lower())}">
  <header>
    <code class="tid">{_e(d['query'])}</code>
    <span class="pil">{_e(d.get('_poin', ''))}</span>
    <span class="pil">{_e(d.get('_jalur', ''))}</span>
    <span class="pil {'seeded' if d.get('sumber_label') == 'seeded' else ''}">
      label: {_e(d.get('sumber_label', '?'))}</span>
  </header>
  <div class="masukan">
    <div><span class="k">Query diterbitkan</span><code class="q">{_e(d.get('teks_query') or '?')}</code></div>
    <div><span class="k">Filter</span><code>{_e(filt_txt)}</code></div>
    <div><span class="k">Label benar ({len(relevan)})</span><div class="daftar">{lbl}</div></div>
  </div>
  <table class="hasil-tabel">
    <thead><tr><th>Konfigurasi</th><th>Chunk terambil (urut peringkat)</th>
      <th>nDCG@{k_op}</th><th>Recall@{k_op}</th><th>Hit@{k_op}</th><th>MRR</th>
      <th>Peringkat relevan pertama</th></tr></thead>
    <tbody>{baris}</tbody>
  </table>
</article>"""


def bangun(h: dict, rp: dict | None, bl: dict | None, atr: dict | None) -> str:
    k_op = (h.get("metadata") or {}).get("k_operasi", 3)
    kfg = h["konfigurasi"]
    label_kfg = {"dense": "Dense saja", "lexical": "Lexical saja", "rrf": "RRF (tanpa rerank)",
                 "dense+rerank": "Dense + rerank", "rrf+rerank": "RRF + rerank"}
    detail = h["detail_per_query"]
    pq = h["skor_per_query"]
    punya_bukti = any(d.get("terambil") for d in detail)

    # Kelompokkan topik menurut poin, diturunkan dari awalan id (konvensi bangun_eval_set_operasi).
    for i, d in enumerate(detail):
        d["_idx"] = i
        d["_poin"] = d["query"].split("-")[0]
        d["_jalur"] = "search"

    anc = h.get("anchor") or {}
    detail_anc = anc.get("detail_per_query") or []
    for i, d in enumerate(detail_anc):
        d["_idx"] = i
        d["_poin"] = "itbx"
        d["_jalur"] = "reference"
        d.setdefault("sumber_label", "aturan")

    ids = {c for d in detail + detail_anc for c in (d.get("relevan") or [])}
    ids |= {c for d in detail + detail_anc for lst in (d.get("terambil") or {}).values() for c in lst}
    glo = _glosarium(ids)

    kartu_search = "".join(_kartu_topik(d, pq, kfg, glo, k_op, label_kfg) for d in detail)
    kartu_anchor = "".join(
        _kartu_topik(d, anc.get("skor_per_query") or {}, list(anc.get("konfigurasi") or []),
                     glo, k_op, {**label_kfg, **{
                         "anchor-urutan-db": "Urutan DB (sebelum perbaikan)",
                         "anchor-sadar-zona": "Sadar-zona (produksi)"}})
        for d in detail_anc)

    # --- tabel agregat penuh ------------------------------------------------------------------
    agg, st = h["agregat"], h.get("statistik") or {}
    ci, uji = st.get("ci", {}), st.get("uji_vs_produksi", {})
    metrik_kolom = [f"ndcg@{k_op}", f"recall@{k_op}", f"hit@{k_op}", "mrr", "map"]
    baris_agg = ""
    for k in kfg:
        a, c = agg[k], ci.get(k, {})
        sel = ""
        for m in metrik_kolom:
            e = c.get(m)
            sel += (f'<td class="num"><b>{a.get(m, 0):.3f}</b>'
                    + (f'<span class="ci">[{e["bawah"]:.3f}–{e["atas"]:.3f}]</span>' if e else "")
                    + "</td>")
        u = uji.get(k, {}).get(f"ndcg@{k_op}")
        vonis = ("<span class='redup'>pembanding</span>" if k == h.get("produksi") else
                 (f'{u["selisih_rata"]:+.3f} <span class="ci">p={u["p"]:.2g} '
                  f'{"signifikan" if u["signifikan"] else "setara"}</span>' if u else "&mdash;"))
        baris_agg += (f'<tr{" class=sorot" if k == h.get("produksi") else ""}>'
                      f'<td>{_e(label_kfg.get(k, k))}</td>{sel}<td>{vonis}</td>'
                      f'<td class="num">{h["latensi_ms"][k]:,.0f} ms</td></tr>')

    # --- lantai & langit ----------------------------------------------------------------------
    blok_baseline = ""
    if bl:
        la, lg = bl["lantai_acak"]["metrik"], bl["langit_langit"]["maks"]
        bk = bl["lantai_acak"]["besar_kandidat"]
        rows = ""
        for m in [f"hit@{k_op}", f"recall@{k_op}", f"precision@{k_op}", f"ndcg@{k_op}", "mrr", "map"]:
            base = {"mrr": "rr", "map": "ap"}.get(m, m)
            l1, l2 = la.get(base), lg.get(base)
            ukur = agg.get(h.get("produksi"), {}).get(m)
            rows += (f"<tr><td><b>{_e(m)}</b></td><td class='num'>{l1:.4f}</td>"
                     f"<td class='num'><b>{ukur:.3f}</b></td><td class='num'>{l2:.3f}</td>"
                     f"<td class='num'>{ukur / l1:,.0f}&times;</td>"
                     f"<td class='num'><b>{ukur / l2:.0%}</b></td></tr>")
        blok_baseline = f"""
  <p class="cat">Lantai = harapan metrik bila {bl['lantai_acak']['n_topik']} topik dinilai atas
  urutan ACAK dari kandidat yang lolos filter (median {bk['median']:.0f} chunk per topik,
  {bl['n_simulasi']} simulasi/topik, seed {bl['seed']}). Langit-langit = nilai tertinggi yang
  MUNGKIN mengingat struktur label.</p>
  <table class="tabel">
    <thead><tr><th>Metrik</th><th>Lantai acak</th><th>Produksi</th><th>Langit-langit</th>
      <th>&divide; lantai</th><th>% dari langit</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>"""

    md = h.get("metadata") or {}
    peringatan_bukti = "" if punya_bukti else """
  <div class="catat"><b>Hasil retrieval per-query belum terekam di berkas hasil ini.</b>
  Jalankan ulang <code>python -m eval.eval_rag</code> dengan versi harness terbaru; sampai itu,
  bagian bukti hanya menampilkan label dan skor, tanpa chunk yang terambil.</div>"""

    n_search, n_anc = len(detail), len(detail_anc)
    return f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bukti Evaluasi RAG</title>
<style>
  :root {{ --bg:#f7f8fa; --kartu:#fff; --tinta:#16202b; --redup:#5d6874; --grs:#e4e8ed;
           --ok:#0f9d58; --ok-bg:#e9f6ef; --no:#c9302c; --sorot:#eef7f1; --kode:#f2f4f7; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--tinta);
    font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:28px 16px 80px; }}
  h1 {{ font-size:26px; margin:0 0 4px; letter-spacing:-.02em; }}
  h2 {{ font-size:20px; margin:40px 0 8px; padding-top:10px; border-top:3px solid var(--tinta); }}
  h3 {{ font-size:16px; margin:24px 0 6px; color:var(--redup); }}
  .sub, .cat {{ color:var(--redup); font-size:14px; }}
  code {{ font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace; background:var(--kode);
          padding:1px 5px; border-radius:4px; }}
  .blok {{ background:var(--kartu); border:1px solid var(--grs); border-radius:10px;
           padding:16px 18px; margin:14px 0; overflow-x:auto; }}
  table.tabel {{ width:100%; border-collapse:collapse; font-size:14px; }}
  table.tabel th, table.tabel td {{ padding:9px 10px; border-bottom:1px solid var(--grs);
                                    text-align:left; vertical-align:top; }}
  table.tabel thead th {{ font-size:12.5px; color:var(--redup); border-bottom:2px solid var(--grs); }}
  .num {{ text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }}
  .ci {{ display:block; color:var(--redup); font-size:11.5px; font-weight:400; }}
  tr.sorot {{ background:var(--sorot); }}
  .redup {{ color:var(--redup); }}
  nav.toc {{ background:var(--kartu); border:1px solid var(--grs); border-radius:10px;
             padding:14px 18px; margin:16px 0 8px; }}
  nav.toc a {{ color:var(--tinta); text-decoration:none; margin-right:16px; font-size:14px;
               white-space:nowrap; line-height:2; }}
  nav.toc a:hover {{ text-decoration:underline; }}
  .alat {{ position:sticky; top:0; z-index:5; background:var(--bg); padding:12px 0;
           border-bottom:1px solid var(--grs); display:flex; gap:8px; flex-wrap:wrap;
           align-items:center; }}
  .alat input, .alat select {{ font:14px inherit; padding:7px 10px; border:1px solid var(--grs);
                               border-radius:7px; background:#fff; }}
  .alat input {{ min-width:230px; }}
  .hitung {{ color:var(--redup); font-size:13.5px; }}
  article.topik {{ background:var(--kartu); border:1px solid var(--grs); border-radius:10px;
                   padding:14px 16px; margin:12px 0; }}
  article.topik header {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap;
                          margin-bottom:10px; }}
  .tid {{ font-weight:650; background:none; padding:0; font-size:14px; }}
  .pil {{ font-size:11.5px; color:var(--redup); background:var(--kode); border-radius:99px;
          padding:2px 9px; }}
  .pil.seeded {{ background:#fdf3e3; color:#8a6116; }}
  .masukan {{ display:grid; gap:7px; margin-bottom:11px; font-size:13.5px; }}
  .masukan .k {{ display:inline-block; min-width:150px; color:var(--redup); }}
  .masukan .q {{ background:#eef3fb; }}
  .daftar {{ display:inline-block; vertical-align:top; }}
  .item {{ display:flex; gap:7px; align-items:baseline; padding:2px 0; font-size:13px; }}
  .item .rank {{ color:var(--redup); min-width:16px; font-variant-numeric:tabular-nums; }}
  .item .ket {{ color:var(--redup); font-size:12px; }}
  .item.kena code {{ background:var(--ok-bg); color:#0b6b3e; font-weight:650; }}
  .item .cek {{ color:var(--ok); font-size:11.5px; font-weight:650; }}
  table.hasil-tabel {{ width:100%; border-collapse:collapse; font-size:13px; }}
  table.hasil-tabel th, table.hasil-tabel td {{ padding:7px 8px; vertical-align:top;
    border-bottom:1px solid var(--grs); text-align:left; }}
  table.hasil-tabel thead th {{ font-size:11.5px; color:var(--redup); }}
  td.kfg {{ white-space:nowrap; font-weight:600; }}
  td.hasil {{ width:52%; }}
  .catat {{ background:#fffdf5; border:1px solid #f0e3bf; border-left:4px solid #e8a33d;
            border-radius:8px; padding:14px 16px; margin:14px 0; font-size:14px; }}
  .meta {{ font-size:12.5px; color:var(--redup); border-top:1px solid var(--grs);
           margin-top:40px; padding-top:14px; }}
</style></head><body><div class="wrap">

<h1>Bukti Evaluasi Retrieval RAG</h1>
<p class="sub">Wilayah {_e(h['wilayah'])} &middot; {n_search} topik <code>search</code> +
{n_anc} topik <code>reference</code> &middot; titik operasi k={k_op} &middot;
diukur {_e(h.get('dibuat', '')[:10])}</p>

<nav class="toc">
  <a href="#a">A. Metodologi &amp; definisi metrik</a>
  <a href="#b">B. Hasil agregat</a>
  <a href="#c">C. Lantai &amp; langit-langit</a>
  <a href="#d">D. Bukti per-query — jalur search</a>
  <a href="#e">E. Bukti per-query — jalur rujukan</a>
  <a href="#f">F. Ablasi</a>
  <a href="#g">G. Batas keabsahan</a>
</nav>
{peringatan_bukti}

<h2 id="a">A. Metodologi &amp; definisi metrik</h2>
<h3>A1. Apa yang diukur</h3>
<div class="blok"><p class="cat">Yang diuji adalah <b>leg retrieval</b>: untuk query dan filter yang
benar-benar diterbitkan produksi, apakah chunk otoritatif masuk ke {k_op} teratas — karena
{k_op} chunk itulah yang dikirim ke LLM (<code>generator.top_k_dukungan</code>). Setiap topik
dinilai atas <b>label yang sama</b> untuk seluruh konfigurasi, sehingga selisih antar-konfigurasi
tidak bisa dijelaskan oleh perbedaan label. Label: <b>{h['sumber_label'].get('aturan', 0)}</b>
diturunkan dari struktur peraturan, <b>{h['sumber_label'].get('seeded', 0)}</b> seeded.</p></div>

<h3>A2. Definisi tiap metrik</h3>
<div class="blok"><table class="tabel">
  <thead><tr><th>Metrik</th><th>Rumus</th><th>Arti</th><th>Langit-langit</th></tr></thead>
  <tbody>{''.join(f'<tr><td><b>{_e(n)}</b></td><td class="cat">{_e(f)}</td>'
                  f'<td class="cat">{_e(a)}</td><td class="cat">{_e(l)}</td></tr>'
                  for n, f, a, l in _DEFINISI)}</tbody>
</table></div>

<h3>A3. Statistik &amp; reproduksibilitas</h3>
<div class="blok"><p class="cat">Selang kepercayaan 95% lewat <b>bootstrap</b> atas topik;
perbandingan antar-konfigurasi lewat <b>Wilcoxon signed-rank berpasangan</b> (topik identik), dengan
ukuran efek rank-biserial dilaporkan berdampingan. Jalur retrieval deterministik untuk indeks dan
query yang sama, jadi ketidakpastian datang dari pencuplikan topik — bukan dari derau antar-jalan.
<br>Commit saat diukur <code>{_e(md.get('commit') or '?')}</code>
{'<b>(working tree kotor)</b>' if md.get('commit_kotor') else ''} &middot;
embedding <code>{_e(md.get('model_embedding'))}</code> &middot;
rerank <code>{_e(md.get('model_rerank'))}</code> &middot;
kandidat k={md.get('kandidat_k')} &middot; dinilai sampai peringkat {md.get('top_n_dinilai')} &middot;
korpus {h['korpus']['chunk']} chunk / {h['korpus']['vektor']} vektor.</p></div>

<h2 id="b">B. Hasil agregat pada titik operasi (k={k_op})</h2>
<div class="blok"><table class="tabel">
  <thead><tr><th>Konfigurasi</th>
    {''.join(f'<th class="num">{_e(m)}</th>' for m in metrik_kolom)}
    <th>vs pembanding (nDCG@{k_op})</th><th class="num">Latensi</th></tr></thead>
  <tbody>{baris_agg}</tbody>
</table>
<p class="cat">Rata-rata tak tertimbang atas {n_search} topik. <b>Bukan angka produksi</b> — jalur
produksi berbeda per poin, dan sebagian topik di sini adalah lengan kontrafaktual yang sengaja
dibuat untuk ablasi.</p></div>

<h2 id="c">C. Lantai acak &amp; langit-langit</h2>
<div class="blok">{blok_baseline or '<p class="cat">eval/baseline.json belum ada.</p>'}</div>

<h2 id="d">D. Bukti per-query — jalur <code>search</code> ({n_search} topik)</h2>
<p class="cat">Tiap kartu memuat query yang benar-benar diterbitkan, filternya, label yang benar,
dan <b>chunk yang benar-benar terambil</b> di tiap konfigurasi. Chunk yang cocok label ditandai
hijau. Lima teratas ditampilkan; titik operasi ada di tiga teratas.</p>
<div class="alat">
  <input id="cari" type="search" placeholder="Cari id topik atau teks query…">
  <select id="f-poin"><option value="">Semua poin</option>
    <option>intensitas</option><option>itbx</option><option>dampak</option></select>
  <select id="f-label"><option value="">Semua label</option>
    <option value="aturan">label aturan</option><option value="seeded">label seeded</option></select>
  <select id="f-gagal"><option value="">Semua hasil</option>
    <option value="1">hanya yang GAGAL di produksi</option>
    <option value="0">hanya yang berhasil</option></select>
  <span class="hitung" id="hitung"></span>
</div>
<div id="daftar-search">{kartu_search}</div>

<h2 id="e">E. Bukti per-query — jalur <code>reference</code> ({n_anc} topik)</h2>
<p class="cat">Jalur <code>get_by_reference</code>: murni SQL, tanpa embedding maupun rerank.
Menangani seluruh poin <code>itbx</code> di produksi.</p>
<div id="daftar-anchor">{kartu_anchor}</div>

<h2 id="f">F. Ablasi</h2>
<div class="blok"><p class="cat">Rincian ablasi jalur rujukan dan ablasi string query — beserta
selang kepercayaan dan uji berpasangannya — ada di <code>laporan_rag.html</code> bagian 7 dan 8.
Bukti per-topiknya ada di bagian D dan E di atas: bandingkan baris konfigurasi di dalam kartu yang
sama, karena seluruh konfigurasi dinilai atas topik dan label yang identik.</p></div>

<h2 id="g">G. Batas keabsahan</h2>
<div class="catat">
<ul>
  <li><b>Label belum divalidasi ahli tata ruang.</b> Yang terukur adalah konsistensi terhadap
      struktur peraturan, bukan kebenaran hukum. Klaim absolut seperti &ldquo;akurasi N%&rdquo;
      tidak sah; yang sah adalah klaim relatif antar-konfigurasi.</li>
  <li><b>Relevansi biner</b>, sedangkan nDCG dirancang untuk relevansi bergradasi — jadi nDCG di
      sini lebih kasar daripada maksud aslinya.</li>
  <li><b>Satu wilayah</b> ({_e(h['wilayah'])}).</li>
  <li><b>Latensi diukur di mesin pengembang</b> terhadap Postgres lokal dan API eksternal — untuk
      membandingkan biaya antar-lapis, bukan angka produksi.</li>
</ul>
</div>

<div class="meta">
Disusun oleh <code>eval/laporan_bukti.py</code> dari <code>laporan_rag.json</code> (hasil
pengukuran) + keterangan chunk dari basis data. Tidak ada angka yang dihitung ulang di sini.
Dokumen dibuat {_e(datetime.now().strftime('%Y-%m-%d %H:%M'))}.
</div>

<script>
(function () {{
  var cari = document.getElementById('cari'), fp = document.getElementById('f-poin'),
      fl = document.getElementById('f-label'), fg = document.getElementById('f-gagal'),
      hit = document.getElementById('hitung'),
      kartu = document.querySelectorAll('#daftar-search article.topik');
  function saring() {{
    var q = cari.value.trim().toLowerCase(), n = 0;
    kartu.forEach(function (el) {{
      var ok = (!q || el.dataset.cari.indexOf(q) > -1)
            && (!fp.value || el.dataset.poin === fp.value)
            && (!fl.value || el.dataset.label === fl.value)
            && (!fg.value || el.dataset.gagal === fg.value);
      el.style.display = ok ? '' : 'none';
      if (ok) n++;
    }});
    hit.textContent = n + ' dari ' + kartu.length + ' topik ditampilkan';
  }}
  [cari, fp, fl, fg].forEach(function (el) {{ el.addEventListener('input', saring); }});
  saring();
}})();
</script>
</div></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Laporan evaluasi lengkap dgn bukti per-query.")
    ap.add_argument("--hasil", type=Path, default=_DIR / "laporan_rag.json")
    ap.add_argument("--out", type=Path, default=_OUT)
    args = ap.parse_args()

    h = _muat(args.hasil.name)
    if not h:
        raise SystemExit(f"{args.hasil} tidak ada — jalankan eval_rag dulu")
    args.out.write_text(bangun(h, _muat("ringkasan_produksi.json"), _muat("baseline.json"),
                               _muat("atribusi.json")), encoding="utf-8")
    print(f"[bukti] -> {args.out}  ({args.out.stat().st_size:,} byte)")


if __name__ == "__main__":
    main()
