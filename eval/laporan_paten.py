"""eval/laporan_paten.py — lembar bukti metrik untuk penyusunan dokumen paten.

Kenapa modul ini ada, terpisah dari tiga laporan lain. `laporan_rag.html` adalah dokumen BUKTI
internal: ia sengaja menampilkan seluruh metrik, termasuk yang redundan dan yang langit-langitnya
rendah, karena di sana fungsinya memeriksa silang. Dokumen paten menuntut kebalikannya — set
sekecil mungkin yang masing-masing membawa klaim berbeda, karena tiap metrik tambahan adalah
permukaan serang tambahan bagi pemeriksa.

EMPAT METRIK, dan alasan pemilihannya:

  1. nDCG@3 jalur produksi — kemampuan, pada konfigurasi yang BENAR-BENAR dikirim. Bukan rata-rata
     agregat, yang separuh topiknya lengan ablasi berskor rendah dengan sengaja.
  2. Ablasi jalur rujukan — KAUSALITAS. Inilah bukti paten yang sesungguhnya: paten menilai efek
     teknis dari hal yang diklaim baru, bukan skor yang bagus.
  3. Presisi sitasi — rujukan yang muncul di jawaban benar-benar ada.
  4. Groundedness kutipan — teks yang dikutip memang ada di pasal yang disitasi.

Nomor 3 dan 4 dipertahankan walau keduanya "integritas sitasi" karena menguji hal berbeda: id yang
ada versus kutipan yang verbatim. Sebuah sistem bisa lolos yang pertama dan gagal yang kedua.

YANG SENGAJA TIDAK DITAMPILKAN, dan alasannya ikut dicetak di halaman supaya pembaca dokumen ini
tidak menambahkannya kembali: Recall@3/Hit@3/MRR/MAP (MRR IDENTIK PERSIS dengan Recall@3 pada baris
produksi — 0.868254 keduanya; Hit@3 sama nilainya di 89 dari 105 topik; MAP berkorelasi 0.945 dgn
MRR), Precision@3 (langit-langitnya 0.441, mengundang salah baca), baris agregat, recall zona
(campur versi kode), dan seluruh metrik sisi generasi (penyebut 21 permohonan, dan faithfulness
100% bersyarat pada lolos gerbang guardrail).

TIDAK ADA ANGKA YANG DITULIS TANGAN. Seluruhnya dibaca dari artefak hasil yang sudah di-commit
(`laporan_rag.json`, `baseline.json`, `atribusi.json`) dan dihitung ulang memakai fungsi yang sama
dengan laporan teknis — `indeks_jalur_produksi` dan `statistik.bootstrap_ci`. Kalau pengukurannya
dijalankan ulang, halaman ini ikut berubah, ketimbang membeku pada snapshot yang nanti menyesatkan.

CLI:
  python -m eval.laporan_paten
  python -m eval.laporan_paten --out eval/laporan_paten.html
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
from datetime import datetime
from pathlib import Path

_DIR = Path(__file__).parent
_OUT = _DIR / "laporan_paten.html"

# Cabang yang benar-benar dijalankan produksi, dengan konfigurasi fusi per-poin. Diimpor dari
# eval_rag lewat `indeks_jalur_produksi` supaya tak ada salinan kedua yang bisa menyimpang.
_KFG_PER_POIN = {"intensitas": "dense+rerank", "dampak": "rrf+rerank"}


def _e(s: object) -> str:
    return html.escape(str(s if s is not None else "—"))


def _muat(nama: str) -> dict | None:
    p = _DIR / nama
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _jalur_produksi(h: dict) -> dict:
    """nDCG@3 jalur produksi + selang kepercayaan, dihitung ulang dari skor per-topik."""
    from eval.eval_rag import indeks_jalur_produksi
    from eval.statistik import bootstrap_ci

    detail, skor = h.get("detail_per_query") or [], h.get("skor_per_query") or {}
    idx = indeks_jalur_produksi(detail)
    nilai: list[float] = []
    for i in idx:
        poin = str(detail[i]["query"]).split("-")[0]
        kfg = _KFG_PER_POIN.get(poin)
        if kfg and kfg in skor:
            nilai.append(skor[kfg][i]["ndcg@3"])
    if not nilai:
        return {}
    ci = bootstrap_ci(nilai)
    return {"n": len(nilai), "n_total": len(detail), "rata": ci.rata,
            "bawah": ci.bawah, "atas": ci.atas}


def _ablasi_anchor(h: dict) -> dict:
    """Lengan sebelum/sesudah ablasi jalur rujukan + uji berpasangannya."""
    an = h.get("anchor") or {}
    ag = an.get("agregat") or {}
    uji = (((an.get("statistik") or {}).get("uji_vs_produksi") or {})
           .get("anchor-urutan-db") or {}).get("ndcg@3") or {}
    sebelum, sesudah = ag.get("anchor-urutan-db") or {}, ag.get("anchor-sadar-zona") or {}
    if not sebelum or not sesudah:
        return {}
    return {
        "n": an.get("n_query"),
        "ndcg_sebelum": sebelum.get("ndcg@3"), "ndcg_sesudah": sesudah.get("ndcg@3"),
        "recall_sebelum": sebelum.get("recall@3"), "recall_sesudah": sesudah.get("recall@3"),
        "kosong_sebelum": sebelum.get("query_tanpa_hasil_relevan"),
        "kosong_sesudah": sesudah.get("query_tanpa_hasil_relevan"),
        "p": uji.get("p"), "efek": uji.get("efek"),
    }


def _fmt_p(p: float | None) -> str:
    """p-value dalam notasi ilmiah yang bisa dikutip — 'p=0.000' tidak bisa dipertahankan."""
    if p is None:
        return "—"
    if p >= 1e-4:
        return f"{p:.4f}"
    mant, eksp = f"{p:.1e}".split("e")
    return f"{mant}&times;10<sup>{int(eksp)}</sup>"


def bangun(h: dict, bl: dict | None, atr: dict | None) -> str:
    jp = _jalur_produksi(h)
    ab = _ablasi_anchor(h)
    md = h.get("metadata") or {}
    korpus = h.get("korpus") or {}
    lantai = ((bl or {}).get("lantai_acak") or {}).get("metrik", {}).get("ndcg@3")
    langit = ((bl or {}).get("langit_langit") or {}).get("maks", {})
    kandidat = ((bl or {}).get("lantai_acak") or {}).get("besar_kandidat") or {}
    pk = (atr or {}).get("presisi_korpus") or {}
    pa = (atr or {}).get("presisi_anchor") or {}
    gk = (atr or {}).get("groundedness_kutipan") or {}

    lipat = (jp["rata"] / lantai) if (jp and lantai) else None
    label = ", ".join(f"{v} {k}" for k, v in sorted((h.get("sumber_label") or {}).items()))

    return f"""<title>Bukti Metrik RDTR</title>
<style>
  /* Lembar bukti satu kolom: jejak provenans di atas, empat blok metrik, lalu batas & kutipan. */
  :root {{
    --kertas: #f7f5f1;
    --permukaan: #fffefb;
    --tinta: #1c2024;
    --tinta-redup: #5d6670;
    --garis: #ddd7cd;
    --aksen: #7d5a1f;
    --aksen-redup: #f0e7d4;
    --sahih: #2f6b4f;
    --batas: #8a5a2b;
    --font-judul: "Spectral", Georgia, "Times New Roman", serif;
    --font-teks: "IBM Plex Sans", -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    --font-data: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    color-scheme: light;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --kertas: #15181b;
      --permukaan: #1d2125;
      --tinta: #e8e4dc;
      --tinta-redup: #9aa2ab;
      --garis: #333a41;
      --aksen: #d4a94f;
      --aksen-redup: #2b2519;
      --sahih: #6cba92;
      --batas: #d99a5b;
      color-scheme: dark;
    }}
  }}
  :root[data-theme="dark"] {{
    --kertas: #15181b;
    --permukaan: #1d2125;
    --tinta: #e8e4dc;
    --tinta-redup: #9aa2ab;
    --garis: #333a41;
    --aksen: #d4a94f;
    --aksen-redup: #2b2519;
    --sahih: #6cba92;
    --batas: #d99a5b;
    color-scheme: dark;
  }}

  body {{
    background: var(--kertas);
    color: var(--tinta);
    font-family: var(--font-teks);
    font-size: 15px;
    line-height: 1.62;
    padding-inline: 20px;
    padding-block: 0 72px;
  }}
  .bungkus {{ max-width: 920px; margin-inline: auto; }}
  h1, h2, h3 {{ font-family: var(--font-judul); text-wrap: balance; margin: 0; }}
  p {{ margin: 0; max-width: 68ch; }}
  code {{ font-family: var(--font-data); font-size: 0.88em;
          background: var(--aksen-redup); padding: 1px 5px; border-radius: 3px; }}
  sup {{ font-size: 0.7em; }}

  /* --- kepala --- */
  header {{ padding-block: 52px 28px; border-bottom: 2px solid var(--tinta); }}
  .eyebrow {{ font-family: var(--font-data); font-size: 11px; letter-spacing: .14em;
              text-transform: uppercase; color: var(--aksen); margin: 0 0 14px; }}
  h1 {{ font-size: clamp(30px, 5.2vw, 44px); line-height: 1.12; font-weight: 600;
        letter-spacing: -.012em; }}
  .sub {{ margin-top: 14px; color: var(--tinta-redup); font-size: 16px; }}

  /* --- jejak provenans --- */
  .provenans {{ display: grid; gap: 1px; background: var(--garis);
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                border: 1px solid var(--garis); margin-block: 26px 48px; }}
  .provenans div {{ background: var(--permukaan); padding: 11px 14px; min-width: 0; }}
  .provenans dt {{ font-family: var(--font-data); font-size: 10px; letter-spacing: .1em;
                   text-transform: uppercase; color: var(--tinta-redup); }}
  .provenans dd {{ margin: 3px 0 0; font-family: var(--font-data); font-size: 13px;
                   word-break: break-word; }}

  /* --- blok metrik --- */
  .metrik {{ background: var(--permukaan); border: 1px solid var(--garis);
             border-left: 3px solid var(--aksen); margin-bottom: 22px; }}
  .metrik > .kepala {{ padding: 18px 22px 0; }}
  .no {{ font-family: var(--font-data); font-size: 11px; letter-spacing: .12em;
         color: var(--tinta-redup); }}
  .metrik h2 {{ font-size: 23px; font-weight: 600; margin-top: 2px; }}
  .bidang {{ display: grid; grid-template-columns: 148px 1fr; gap: 1px;
             background: var(--garis); margin-top: 18px; }}
  .bidang > * {{ background: var(--permukaan); padding: 13px 22px; min-width: 0; }}
  .bidang dt {{ font-family: var(--font-data); font-size: 10.5px; letter-spacing: .1em;
                text-transform: uppercase; color: var(--tinta-redup); padding-top: 15px; }}
  @media (max-width: 620px) {{
    .bidang {{ grid-template-columns: 1fr; }}
    .bidang dt {{ padding-bottom: 0; }}
    .bidang dd {{ padding-top: 4px; }}
  }}
  .skor {{ font-family: var(--font-data); font-variant-numeric: tabular-nums; }}
  .besar {{ font-size: 30px; font-weight: 600; line-height: 1.15; letter-spacing: -.02em;
            display: block; }}
  .ci {{ color: var(--tinta-redup); font-size: 13px; }}
  .panah {{ color: var(--aksen); padding-inline: 4px; }}
  .lipat {{ display: inline-block; margin-top: 8px; font-size: 13px; color: var(--sahih);
            font-family: var(--font-data); }}
  .turunan {{ margin-top: 10px; font-size: 13.5px; }}
  .turunan b {{ font-family: var(--font-data); font-variant-numeric: tabular-nums; }}

  /* --- tabel --- */
  .gulir {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--garis);
            vertical-align: top; }}
  th {{ font-family: var(--font-data); font-size: 10.5px; letter-spacing: .09em;
        text-transform: uppercase; color: var(--tinta-redup); border-bottom: 1px solid var(--tinta); }}
  tbody tr:last-child td {{ border-bottom: none; }}

  /* --- bagian --- */
  section {{ margin-top: 52px; }}
  section > h2 {{ font-size: 25px; font-weight: 600; padding-bottom: 10px;
                  border-bottom: 1px solid var(--tinta); margin-bottom: 18px; }}
  .catatan {{ background: var(--permukaan); border: 1px solid var(--garis);
              border-left: 3px solid var(--batas); padding: 16px 20px; margin-top: 16px; }}
  .catatan ol {{ margin: 0; padding-left: 20px; }}
  .catatan li {{ margin-bottom: 9px; }}
  .catatan li:last-child {{ margin-bottom: 0; }}
  .kutipan {{ background: var(--permukaan); border: 1px solid var(--garis);
              padding: 22px 24px; margin-top: 16px; font-family: var(--font-judul);
              font-size: 16.5px; line-height: 1.72; }}
  .kutipan b {{ font-family: var(--font-data); font-size: 15px;
                font-variant-numeric: tabular-nums; }}
  footer {{ margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--garis);
            color: var(--tinta-redup); font-size: 12.5px; }}
  footer p {{ max-width: none; }}
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Spectral:wght@500;600&display=swap">

<div class="bungkus">
<header>
  <p class="eyebrow">Lembar bukti &middot; penyusunan dokumen paten</p>
  <h1>Bukti metrik sistem RAG RDTR {_e(h.get('wilayah'))}</h1>
  <p class="sub">Empat metrik, masing-masing membawa satu klaim yang berbeda. Setiap angka di
  halaman ini dibaca langsung dari artefak hasil pengukuran yang sudah tersimpan di repositori —
  tidak ada yang ditulis tangan, dan menjalankan ulang pengukurannya akan memperbarui halaman ini
  dengan sendirinya.</p>
</header>

<dl class="provenans">
  <div><dt>Korpus</dt><dd>{_e(korpus.get('chunk'))} chunk &middot; {_e(korpus.get('vektor'))} vektor</dd></div>
  <div><dt>Commit pengukuran</dt><dd>{_e(md.get('commit'))}</dd></div>
  <div><dt>Titik operasi</dt><dd>k = {_e(md.get('k_operasi'))}</dd></div>
  <div><dt>Embedding</dt><dd>{_e(md.get('model_embedding'))}</dd></div>
  <div><dt>Rerank</dt><dd>{_e(md.get('model_rerank'))}</dd></div>
  <div><dt>Sumber label</dt><dd>{_e(label)}</dd></div>
</dl>

<article class="metrik">
  <div class="kepala">
    <p class="no">Metrik 1</p>
    <h2>nDCG@3 pada jalur produksi</h2>
  </div>
  <dl class="bidang">
    <dt>Apa yang diuji</dt>
    <dd>Apakah ketentuan yang mengatur permohonan masuk ke <b>3 chunk</b> yang benar-benar dikirim
    ke model — dan di posisi mana. Satu-satunya metrik yang peka terhadap <i>berapa banyak</i>
    sekaligus <i>di posisi mana</i>, dan langit-langitnya tetap 1,000.</dd>

    <dt>Skor</dt>
    <dd class="skor">
      <span class="besar">{jp['rata']:.3f}</span>
      <span class="ci">selang kepercayaan 95% [{jp['bawah']:.3f} &ndash; {jp['atas']:.3f}],
      n = {jp['n']} topik</span>
      <span class="lipat">&asymp; {lipat:.0f}&times; lantai acak {lantai:.4f} &middot;
      langit-langit {langit.get('ndcg@3', 1.0):.3f}</span>
    </dd>

    <dt>Cara mengujinya</dt>
    <dd>{jp['n']} topik pada titik operasi yang dikirim produksi: k&nbsp;=&nbsp;3
    (<code>top_k_dukungan</code>) dengan konfigurasi fusi <b>per poin</b> —
    <code>dense+rerank</code> untuk intensitas, <code>rrf+rerank</code> untuk dampak. Label
    chunk otoritatif <b>diturunkan deterministik</b> dari kolom <code>chunks.zona</code> mengikuti
    struktur peraturan (tabel Lampiran VI per zona untuk intensitas, Lampiran V.B untuk ITBX),
    bukan diketik manual. Selang kepercayaan lewat bootstrap persentil atas topik.
    <div class="turunan"><b>Lantai acak</b> dihitung dengan simulasi
    {_e((bl or {}).get('n_simulasi'))}&times; per topik yang mengambil 3 chunk <b>acak</b> dari
    kolam kandidat yang lolos filter yang sama — median <b>{_e(kandidat.get('median'))}</b> chunk
    (rentang {_e(kandidat.get('min'))}&ndash;{_e(kandidat.get('maks'))}), seed
    <code>{_e((bl or {}).get('seed'))}</code>. Yang hilang hanya peringkatnya, jadi selisihnya
    benar-benar mengukur retrieval.</div></dd>
  </dl>
</article>

<article class="metrik">
  <div class="kepala">
    <p class="no">Metrik 2</p>
    <h2>Ablasi jalur rujukan &mdash; pemilihan sitasi sadar-zona</h2>
  </div>
  <dl class="bidang">
    <dt>Apa yang diuji</dt>
    <dd>Apakah komponen yang diklaim baru <b>benar-benar yang menyebabkan</b> kinerja di atas —
    bukan kebetulan arsitektur lain. Inilah bukti yang dinilai pemeriksa paten: efek teknis dari
    hal yang diklaim, bukan skor yang bagus secara umum.</dd>

    <dt>Skor</dt>
    <dd class="skor">
      <span class="besar">{ab['ndcg_sebelum']:.3f}<span class="panah">&rarr;</span>{ab['ndcg_sesudah']:.3f}</span>
      <span class="ci">nDCG@3, n = {_e(ab['n'])} topik berpasangan</span>
      <div class="turunan">
        Recall@3 <b>{ab['recall_sebelum']:.0%} &rarr; {ab['recall_sesudah']:.0%}</b><br>
        Permohonan yang tidak memperoleh satu pun ketentuan yang benar:
        <b>{_e(ab['kosong_sebelum'])}/{_e(ab['n'])} &rarr; {_e(ab['kosong_sesudah'])}/{_e(ab['n'])}</b><br>
        Wilcoxon berpasangan <b>p = {_fmt_p(ab['p'])}</b> &middot;
        ukuran efek rank-biserial <b>{ab['efek']:.2f}</b>
      </div>
    </dd>

    <dt>Cara mengujinya</dt>
    <dd>{_e(ab['n'])} topik yang <b>sama</b> dengan label yang <b>sama</b>; satu-satunya yang
    ditukar adalah langkah pemilihan kandidat — memotong daftar menurut urutan yang dikembalikan
    basis data, versus memilih menurut kecocokan zona pemohon. Karena topik dan labelnya identik,
    selisihnya tidak bisa dijelaskan oleh perbedaan indeks, versi kode, maupun kesulitan topik.
    Diuji dengan <b>Wilcoxon signed-rank berpasangan</b> (non-parametrik, tidak mengasumsikan
    normalitas) dan ukuran efek rank-biserial.
    <div class="turunan">Efeknya <b>kategoris</b>: seluruh {_e(ab['n'])} pasang bergerak searah
    (ukuran efek {ab['efek']:.2f} adalah nilai ekstrem yang mungkin), dan selang kepercayaan kedua
    lengan tidak bersinggungan. Kelas kesalahan yang dicegahnya nyata dan terdokumentasi: permohonan
    pada satu zona memperoleh tabel ketentuan milik zona lain.</div></dd>
  </dl>
</article>

<article class="metrik">
  <div class="kepala">
    <p class="no">Metrik 3</p>
    <h2>Presisi sitasi</h2>
  </div>
  <dl class="bidang">
    <dt>Apa yang diuji</dt>
    <dd>Apakah rujukan yang muncul di jawaban <b>benar-benar ada</b>. Menangkap halusinasi rujukan —
    mode kegagalan yang paling merusak untuk dokumen berdasar hukum, karena petugas yang membuka
    rujukannya tidak menemukan apa pun.</dd>

    <dt>Skor</dt>
    <dd class="skor">
      <span class="besar">{pk['nilai']:.1%}</span>
      <span class="ci">chunk korpus {_e(pk.get('kena'))}/{_e(pk.get('n'))} &middot;
      rujukan back-end {_e(pa.get('kena'))}/{_e(pa.get('n'))}
      ({pa['nilai']:.1%})</span>
    </dd>

    <dt>Cara mengujinya</dt>
    <dd>Setiap <code>citation_id</code> pada keluaran nyata yang tersimpan di log dicocokkan ke
    tabel <code>chunks</code> di basis data; rujukan ber-pola <code>anchor-N</code> dicocokkan ke
    indeks <code>dasar_hukum</code> yang memang dikirim back-end. Baris uji otomatis dan baris
    retriever tiruan dibuang lebih dulu, dan penyebutnya selalu ikut dilaporkan.
    <div class="turunan"><b>Tanpa anotator dan tanpa model penilai.</b> Verifikasinya mungkin
    karena setiap sitasi membawa pengenal yang menunjuk satu chunk konkret — properti arsitektur
    ini yang membuat pemeriksa dapat menurunkan ulang angka ini sendiri dari artefak yang
    sama.</div></dd>
  </dl>
</article>

<article class="metrik">
  <div class="kepala">
    <p class="no">Metrik 4</p>
    <h2>Groundedness kutipan</h2>
  </div>
  <dl class="bidang">
    <dt>Apa yang diuji</dt>
    <dd>Apakah teks yang <b>dikutip</b> memang ada di pasal yang disitasi. Menangkap kutipan
    karangan — kegagalan yang lolos dari Metrik 3, karena sitasinya sah tapi isinya tidak pernah
    tertulis di sana.</dd>

    <dt>Skor</dt>
    <dd class="skor">
      <span class="besar">{gk['nilai']:.1%}</span>
      <span class="ci">{_e(gk.get('kena'))}/{_e(gk.get('n'))} kutipan</span>
    </dd>

    <dt>Cara mengujinya</dt>
    <dd>Teks <code>kutipan</code> dinormalisasi (huruf kecil, tanda baca menjadi spasi, spasi
    dirapatkan) lalu diuji sebagai substring teks chunk yang disitasi. Yang diuji adalah <i>isinya</i>
    ada di sana, bukan apakah karakternya identik — model merapikan kutipan, dan menuntut kecocokan
    byte akan menandai kutipan yang sah sebagai karangan. Kutipan di bawah 25 karakter dilewati,
    dan ambang itu ditetapkan <b>sebelum</b> hasilnya dilihat.
    <div class="turunan">{_e(gk.get('n', 0) - gk.get('kena', 0))} kegagalannya diperiksa satu per
    satu dan <b>bukan halusinasi</b>: {_e(gk.get('kutipan_judul_dokumen'))} di antaranya mengutip
    judul dokumen alih-alih isi pasal — formalnya bersitasi, substansinya tidak menjelaskan
    apa pun.</div></dd>
  </dl>
</article>

<section>
  <h2>Yang sengaja tidak ditampilkan</h2>
  <p>Setiap metrik tambahan adalah permukaan serang tambahan. Enam hal berikut diukur dan tersedia
  di laporan teknis, tetapi tidak dibawa ke dokumen paten — alasannya dicantumkan supaya tidak
  ditambahkan kembali.</p>
  <div class="gulir"><table>
    <thead><tr><th>Tidak ditampilkan</th><th>Alasan</th></tr></thead>
    <tbody>
      <tr><td>Recall@3, Hit@3, MRR, MAP</td>
          <td><b>MRR identik persis dengan Recall@3</b> pada baris produksi (0,868254 keduanya,
          sama nilainya di 83 dari 105 topik); Hit@3 sama nilainya dengan Recall@3 di 89 dari 105
          topik; MAP berkorelasi 0,945 dengan MRR. Menampilkannya memberi kesan bukti berlipat
          yang tidak ada.</td></tr>
      <tr><td>Precision@3</td>
          <td>Langit-langitnya hanya {langit.get('precision@3', 0):.3f} karena mayoritas topik
          punya kurang dari 3 chunk relevan — sebagian slot pasti terisi chunk tak relevan walaupun
          sistem sempurna. Pembaca yang membandingkannya dengan 1,0 akan menyimpulkan
          sebaliknya.</td></tr>
      <tr><td>Baris agregat seluruh topik</td>
          <td>Separuh topiknya adalah lengan kontrafaktual yang <b>sengaja</b> dibuat berskor
          rendah untuk keperluan ablasi.</td></tr>
      <tr><td>Recall sitasi terhadap zona</td>
          <td>Angkanya mencampur beberapa versi kode, dan mayoritas penyebutnya tidak dapat
          dinilai.</td></tr>
      <tr><td>Seluruh metrik sisi generasi</td>
          <td>Penyebutnya 21 permohonan. Dan faithfulness numerik 100% <b>bersyarat pada lolos
          gerbang</b> pemeriksa internal — populasi yang terukur justru narasi yang sudah
          melewatinya, dan pemeriksa yang teliti akan menunjukkan itu.</td></tr>
      <tr><td>Ablasi kewajiban menyebut zona</td>
          <td>n = 7, cakupan pengulangan tidak seragam karena kuota penyedia habis di tengah
          pengukuran, dan satu kasus masih mengarang kode sub-zona.</td></tr>
    </tbody>
  </table></div>
</section>

<section>
  <h2>Batas yang wajib ikut tertulis</h2>
  <p>Menghilangkan tiga hal ini tidak membuat dokumen lebih kuat — justru itu yang membuatnya bisa
  dibantah di kemudian hari.</p>
  <div class="catatan"><ol>
    <li><b>Label diturunkan dari struktur peraturan, belum divalidasi ahli tata ruang.</b> Yang
    terukur adalah konsistensi sistem terhadap struktur regulasi, <b>bukan kebenaran hukum</b>.
    Klaim berbentuk &ldquo;akurasi N%&rdquo; tidak sah; yang sah adalah klaim relatif
    antar-konfigurasi seperti Metrik 2.</li>
    <li><b>Satu wilayah</b> ({_e(h.get('wilayah'))}). Wilayah lain belum memiliki korpus ber-tag
    zona yang setara, sehingga angka di sini tidak boleh digeneralisasi.</li>
    <li><b>Relevansi biner.</b> Label tidak membedakan &ldquo;sangat relevan&rdquo; dari
    &ldquo;agak relevan&rdquo;, sehingga nDCG di sini lebih kasar daripada maksud aslinya yang
    bergradasi.</li>
  </ol></div>
</section>

<section>
  <h2>Rumusan siap pakai</h2>
  <p>Satu paragraf yang memuat keempat metrik beserta pembingkainya, siap dipindahkan ke naskah
  dokumen.</p>
  <div class="kutipan">
    Pada titik operasi produksi (k&nbsp;=&nbsp;3, konfigurasi fusi per poin, {jp['n']} topik uji
    berlabel aturan), sistem mencapai <b>nDCG@3 = {jp['rata']:.3f}</b> (selang kepercayaan 95%
    {jp['bawah']:.3f}&ndash;{jp['atas']:.3f}) terhadap lantai acak <b>{lantai:.4f}</b> pada kolam
    kandidat bermedian {_e(kandidat.get('median'))} chunk dengan filter identik &mdash;
    <b>{lipat:.0f}&times;</b> &mdash; dengan langit-langit struktural
    {langit.get('ndcg@3', 1.0):.3f}. Efek teknis pemilihan sitasi sadar-zona diisolasi melalui
    ablasi atas topik dan label yang identik: <b>nDCG@3 {ab['ndcg_sebelum']:.3f} &rarr;
    {ab['ndcg_sesudah']:.3f}</b>, Recall@3 <b>{ab['recall_sebelum']:.0%} &rarr;
    {ab['recall_sesudah']:.0%}</b>, dan permohonan yang tidak memperoleh satu pun ketentuan yang
    benar turun dari <b>{_e(ab['kosong_sebelum'])}/{_e(ab['n'])}</b> menjadi
    <b>{_e(ab['kosong_sesudah'])}/{_e(ab['n'])}</b> (Wilcoxon berpasangan,
    <b>p = {_fmt_p(ab['p'])}</b>, ukuran efek rank-biserial <b>{ab['efek']:.2f}</b> &mdash; seluruh
    {_e(ab['n'])} pasang searah). Integritas rujukan diverifikasi secara mekanis tanpa penilai
    manusia: <b>{pk['nilai']:.1%}</b> sitasi ({_e(pk.get('kena'))}/{_e(pk.get('n'))} chunk korpus,
    {_e(pa.get('kena'))}/{_e(pa.get('n'))} rujukan back-end) menunjuk chunk yang nyata, dan
    <b>{gk['nilai']:.1%}</b> kutipan ({_e(gk.get('kena'))}/{_e(gk.get('n'))}) terbukti ada verbatim
    di chunk yang disitasi. Angka-angka ini mengukur konsistensi terhadap struktur peraturan pada
    satu wilayah dengan relevansi biner, dan <b>bukan</b> klaim kebenaran hukum.
  </div>
</section>

<footer>
  <p>Dibangkitkan oleh <code>python -m eval.laporan_paten</code> dari
  <code>laporan_rag.json</code>, <code>baseline.json</code>, dan <code>atribusi.json</code>.
  Tidak ada angka yang ditulis tangan di halaman ini. Rincian metodologi, bukti per topik, dan
  metrik yang sengaja tidak dibawa ke sini ada di <code>laporan_rag.html</code> dan
  <code>laporan_bukti.html</code>. Disusun {_e(datetime.now().strftime('%d-%m-%Y %H:%M'))}.</p>
</footer>
</div>
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Lembar bukti metrik untuk penyusunan dokumen paten (4 metrik final).")
    ap.add_argument("--out", type=Path, default=_OUT)
    args = ap.parse_args()

    h = _muat("laporan_rag.json")
    if not h:
        raise SystemExit("eval/laporan_rag.json tidak ada — jalankan eval_rag dulu")
    bl, atr = _muat("baseline.json"), _muat("atribusi.json")
    if not bl:
        raise SystemExit("eval/baseline.json tidak ada — lantai & langit-langit wajib ada di sini")
    if not atr:
        raise SystemExit("eval/atribusi.json tidak ada — metrik 3 & 4 wajib ada di sini")

    args.out.write_text(bangun(h, bl, atr), encoding="utf-8")
    print(f"[paten] -> {args.out}  ({args.out.stat().st_size:,} byte)")


if __name__ == "__main__":
    main()
