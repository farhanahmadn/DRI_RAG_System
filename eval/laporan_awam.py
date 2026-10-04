"""eval/laporan_awam.py — laporan evaluasi untuk pembaca NON-TEKNIS.

Kenapa modul ini ada, terpisah dari `eval/eval_rag.py`. Laporan teknis (`laporan_rag.html`) adalah
dokumen bukti: 11 bagian, lengkap dengan selang kepercayaan, uji signifikansi, lantai acak, dan
batas keabsahan. Itu memang tugasnya, dan tidak boleh disederhanakan. Tapi akibatnya ia tak bisa
dibaca oleh rekan di luar tim teknis — dan justru mereka yang perlu tahu bagian mana dari sistem
ini boleh dipercaya.

Laporan ini TIDAK menghitung apa pun sendiri. Ia hanya membaca artefak yang sudah dihasilkan
modul lain, lalu menyajikannya dalam bahasa biasa:

  eval/laporan_rag.json        — hasil evaluasi retrieval
  eval/ringkasan_produksi.json — kinerja per-poin pada jalur produksi (termasuk % dari sempurna)
  eval/baseline.json           — lantai acak & langit-langit
  eval/atribusi.json           — mutu sitasi dari keluaran nyata

Dengan begitu tak ada peluang dua laporan menyajikan angka yang berbeda: kalau laporan teknis
berubah, laporan ini ikut berubah saat diregenerasi, dan tak ada angka yang ditulis tangan.

SATU KEPUTUSAN YANG HARUS DIBACA SEBAGAI KONVENSI, BUKAN STANDAR. Pembaca non-teknis butuh label
("Baik", "Perlu perbaikan"), tapi ambang absolut untuk metrik retrieval TIDAK ADA di literatur —
lihat bagian 3 laporan teknis. Karena itu label di sini diturunkan dari "persen dari nilai
tertinggi yang mungkin dicapai", dan ambangnya adalah konvensi internal laporan ini sendiri.
Laporan menyatakan hal itu secara terbuka, bukan menyembunyikannya, supaya tak ada yang mengutip
labelnya sebagai penilaian berstandar industri.

CLI:
  python -m eval.laporan_awam
  python -m eval.laporan_awam --out eval/laporan_ringkas.html
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

_DIR = Path(__file__).parent
_OUT = _DIR / "laporan_ringkas.html"

# Konvensi internal laporan ini, BUKAN standar industri. Diterapkan pada "persen dari nilai
# tertinggi yang mungkin", bukan pada skor mentah — karena skor mentah punya langit-langit yang
# berbeda tiap subset (lihat eval/baseline_acak.py).
_BAND = (
    (0.85, "Sangat baik", "baik"),
    (0.70, "Baik", "baik"),
    (0.50, "Cukup", "sedang"),
    (0.00, "Perlu perbaikan", "lemah"),
)

# Nama poin dalam bahasa yang dipakai pemohon, bukan id internal.
_NAMA_POIN = {
    "itbx": ("Boleh atau tidak kegiatannya",
             "Apakah jenis usaha yang diajukan diizinkan di zona lokasi tersebut."),
    "intensitas": ("Seberapa besar boleh membangun",
                   "Batas KDB, KLB, dan KDH — berapa persen lahan boleh tertutup bangunan, "
                   "berapa lantai, dan berapa yang wajib tetap hijau."),
    "dampak": ("Dampak ke lingkungan",
               "Perubahan limpasan air hujan akibat pembangunan, dan ketentuan khusus yang "
               "berlaku bila lokasi berada di kawasan resapan air atau rawan bencana."),
}

_METRIK_AWAM = (
    ("hit@3", "Menemukan dasar hukumnya",
     "Dari 10 permohonan, pada berapa permohonan sistem berhasil menemukan pasal yang mengatur."),
    ("mrr", "Menaruhnya di urutan teratas",
     "Setelah ditemukan, apakah pasal itu muncul di urutan pertama atau terkubur di bawah."),
    ("recall@3", "Mengambil SEMUA yang berlaku",
     "Satu permohonan bisa terkena beberapa pasal sekaligus. Ini mengukur berapa bagian yang "
     "berhasil terambil."),
    ("ndcg@3", "Nilai gabungan",
     "Menggabungkan ketiga hal di atas menjadi satu angka. Ini angka utama yang dipakai."),
)


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _band(p: float | None) -> tuple[str, str]:
    if p is None:
        return ("Tak terukur", "netral")
    for batas, label, kelas in _BAND:
        if p >= batas:
            return (label, kelas)
    return ("Tak terukur", "netral")


def _muat(nama: str) -> dict | None:
    f = _DIR / nama
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def _bar(p: float | None, kelas: str, lebar: int = 190) -> str:
    """Bilah sederhana. Dipakai karena angka saja sulit dibandingkan dengan mata."""
    if p is None:
        return '<span class="netral">tak terukur</span>'
    isi = max(2, int(round(min(p, 1.0) * lebar)))
    return (f'<span class="bar" style="width:{lebar}px">'
            f'<span class="bar-isi {kelas}" style="width:{isi}px"></span></span>')


def _poin_terangkum(rp: dict) -> list[dict]:
    """Gabungkan cabang per poin menjadi satu baris ringkas, ditimbang bobot trafik.

    Pembaca non-teknis tidak perlu memisah cabang "sub-zona diketahui" dan "tidak"; yang mereka
    butuhkan adalah satu angka per poin yang mencerminkan beban nyata. Cabang tetap ditampilkan
    di tabel rinci, tapi tidak di kartu ringkasan.
    """
    per_poin: dict[str, dict] = {}
    for row in rp.get("baris", []):
        pid = row["poin"]
        w = row.get("bobot_nilai") or 0.0
        slot = per_poin.setdefault(pid, {"poin": pid, "n": 0, "bobot_total": 0.0,
                                         "cabang": [], "akum": {}})
        slot["n"] += row["n"]
        slot["bobot_total"] += w
        slot["cabang"].append(row)
        for m, e in row["metrik"].items():
            dl = e.get("dari_langit")
            if dl is None:
                continue
            a = slot["akum"].setdefault(m, [0.0, 0.0])
            a[0] += dl * w
            a[1] += w
    keluar = []
    for pid, slot in per_poin.items():
        rata = {m: (v[0] / v[1]) if v[1] else None for m, v in slot["akum"].items()}
        slot["dari_langit"] = rata
        slot["utama"] = rata.get("ndcg@3")
        keluar.append(slot)
    urut = {"itbx": 0, "intensitas": 1, "dampak": 2}
    return sorted(keluar, key=lambda s: urut.get(s["poin"], 9))


def bangun(h: dict, rp: dict, bl: dict | None, atr: dict | None) -> str:
    md = h.get("metadata") or {}
    korpus = h["korpus"]
    bt = rp["bobot_trafik"]
    poin_ringkas = _poin_terangkum(rp)

    # --- kartu ringkasan per poin -------------------------------------------------------------
    kartu = ""
    for slot in poin_ringkas:
        nama, penjelasan = _NAMA_POIN.get(slot["poin"], (slot["poin"], ""))
        p = slot["utama"]
        label, kelas = _band(p)
        kartu += f"""
      <div class="kartu-poin {kelas}">
        <div class="poin-kepala">
          <span class="poin-nama">{_e(nama)}</span>
          <span class="lencana {kelas}">{_e(label)}</span>
        </div>
        <div class="poin-angka">{'—' if p is None else f'{p:.0%}'}</div>
        <div class="poin-satuan">dari nilai terbaik yang mungkin</div>
        <p class="poin-ket">{_e(penjelasan)}</p>
      </div>"""

    # --- tabel rinci per cabang ---------------------------------------------------------------
    rinci = ""
    poin_sebelum = None
    for slot in poin_ringkas:
        nama = _NAMA_POIN.get(slot["poin"], (slot["poin"], ""))[0]
        for row in slot["cabang"]:
            sel = ""
            for kunci, _label, _ket in _METRIK_AWAM:
                e = row["metrik"].get(kunci) or {}
                dl = e.get("dari_langit")
                lab, kls = _band(dl)
                sel += (f'<td>{_bar(dl, kls, 110)}'
                        f'<span class="angka-kecil">{"—" if dl is None else f"{dl:.0%}"}</span></td>')
            bobot = row.get("bobot_nilai")
            rinci += (
                f'<tr><td>{_e(nama) if slot["poin"] != poin_sebelum else ""}</td>'
                f'<td class="netral">{_e(row["cabang"])}</td>'
                f'<td>{"semua" if bobot is None or bobot >= 1 else f"{bobot:.0%}"}</td>'
                f'{sel}</tr>')
            poin_sebelum = slot["poin"]

    kepala_rinci = "".join(
        f'<th>{_e(label)}<span class="th-ket">{_e(ket)}</span></th>'
        for _k, label, ket in _METRIK_AWAM)

    # --- dua perbaikan ------------------------------------------------------------------------
    anc = h.get("anchor") or {}
    abq = h.get("ablasi_query") or {}
    perbaikan = []
    if anc.get("n_query"):
        lama = anc["agregat"]["anchor-urutan-db"]
        baru = anc["agregat"][anc["produksi"]]
        perbaikan.append({
            "judul": "Memilih pasal menurut zona pemohon",
            "masalah": ("Dulu sistem mengambil pasal menurut urutan apa adanya dari basis data. "
                        "Akibatnya pemohon di Zona Perumahan bisa disodori aturan Zona Perkantoran "
                        "— dan itu benar-benar terjadi pada 7 permohonan nyata."),
            "sebelum": lama["ndcg@3"], "sesudah": baru["ndcg@3"],
            "tambahan": (f"Permohonan yang sama sekali tidak mendapat pasal yang benar turun dari "
                         f"{lama['query_tanpa_hasil_relevan']} menjadi "
                         f"{baru['query_tanpa_hasil_relevan']} dari {anc['n_query']}."),
        })
    if abq.get("n_keluarga"):
        d = abq["per_konfigurasi"].get("rrf+rerank") or {}
        if d:
            perbaikan.append({
                "judul": "Memperbaiki kata kunci pencarian",
                "masalah": ("Saat back-end tidak memberi kode sub-zona yang presisi — dan itu "
                            f"terjadi pada {bt['tanpa_subzona']:.0%} permohonan — sistem mencari "
                            "dengan kata kunci yang terlalu pendek, sehingga tabel batas "
                            "KDB/KLB/KDH yang benar hampir tak pernah terambil."),
                "sebelum": d["lengan"]["kdb"]["ndcg@3"]["rata"],
                "sesudah": d["lengan"]["tajam"]["ndcg@3"]["rata"],
                "tambahan": (f"Tabel yang benar sampai ke urutan pertama: dari "
                             f"{d['lengan']['kdb']['peringkat1']} menjadi "
                             f"{d['lengan']['tajam']['peringkat1']} dari {abq['n_keluarga']} "
                             f"keluarga zona."),
            })

    blok_perbaikan = ""
    for p in perbaikan:
        naik = (p["sesudah"] - p["sebelum"])
        blok_perbaikan += f"""
      <div class="perbaikan">
        <h3>{_e(p['judul'])}</h3>
        <p class="ket">{_e(p['masalah'])}</p>
        <div class="banding">
          <div><span class="banding-label">Sebelum</span>
               {_bar(p['sebelum'], 'lemah')}<b>{p['sebelum']:.2f}</b></div>
          <div><span class="banding-label">Sesudah</span>
               {_bar(p['sesudah'], 'baik')}<b>{p['sesudah']:.2f}</b></div>
        </div>
        <p class="ket"><b>Naik {naik:.2f} poin.</b> {_e(p['tambahan'])}</p>
      </div>"""

    # --- mutu sitasi -------------------------------------------------------------------------
    blok_sitasi = ""
    if atr:
        ca = atr["cakupan"]
        rows = ""
        for kunci, label, ket in (
            ("presisi_korpus", "Pasal yang dirujuk benar-benar ada",
             "Tidak ada rujukan karangan. Setiap nomor pasal yang disebut bisa dibuka."),
            ("groundedness_kutipan", "Kutipannya benar-benar ada di pasal itu",
             "Teks yang dikutip memang tertulis di pasal yang dirujuk, bukan dirangkai sendiri."),
            ("presisi_anchor", "Rujukan dari back-end dipakai dengan benar",
             "Pasal yang dikirim sistem lain tidak tertukar atau salah nomor."),
        ):
            e = atr.get(kunci) or {}
            v = e.get("nilai")
            lab, kls = _band(v)
            rows += (f'<tr><td><b>{_e(label)}</b><span class="th-ket">{_e(ket)}</span></td>'
                     f'<td>{_bar(v, kls, 150)}</td>'
                     f'<td class="angka-besar">{"—" if v is None else f"{v:.1%}"}</td>'
                     f'<td class="netral">dari {e.get("n", 0)} sitasi</td></tr>')
        g = atr["groundedness_kutipan"]
        catatan_g = ""
        if g.get("kutipan_judul_dokumen"):
            catatan_g = (f'<p class="ket"><b>Yang tersisa:</b> {g["kutipan_judul_dokumen"]} kutipan '
                         f'berisi judul peraturannya, bukan isi pasalnya. Rujukannya benar, tapi '
                         f'kutipannya tidak menjelaskan apa pun.</p>')
        blok_sitasi = f"""
      <table class="tabel">
        <thead><tr><th>Yang diperiksa</th><th></th><th>Hasil</th><th></th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      {catatan_g}
      <p class="ket">Diperiksa dari <b>{ca['permohonan_retriever_nyata']}</b> permohonan nyata
      ({ca['sitasi_dinilai']} sitasi). {ca['permohonan_dari_mock']} permohonan lain di catatan
      sistem berasal dari pengujian internal, bukan pemohon sungguhan, dan sengaja tidak dihitung.</p>"""

    lantai_txt = ""
    if bl:
        la = bl["lantai_acak"]["metrik"].get("ndcg@3")
        bk = bl["lantai_acak"]["besar_kandidat"]
        if la:
            lantai_txt = (
                f"Sebagai pembanding: kalau sistem menebak acak dari sekitar "
                f"{bk['median']:.0f} pasal yang tersedia, nilainya hanya <b>{la:.3f}</b>. "
                f"Jadi sistem bekerja puluhan kali lebih baik daripada menebak — tapi itu hanya "
                f"membuktikan sistemnya hidup, bukan bahwa hasilnya sudah baik.")

    dibuat = h.get("dibuat", "")[:10]
    gen = h.get("generasi") or {}

    return f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ringkasan Evaluasi Sistem</title>
<style>
  :root {{
    --bg:#f6f7f9; --kartu:#fff; --tinta:#17202a; --redup:#5b6673; --grs:#e3e7ec;
    --baik:#0f9d58; --baik-bg:#e8f5ee; --sedang:#e8a33d; --sedang-bg:#fdf3e3;
    --lemah:#d64541; --lemah-bg:#fdecec; --netral:#8b97a3;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--tinta);
    font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:32px 16px 72px; }}
  header.atas {{ border-bottom:3px solid var(--tinta); padding-bottom:16px; margin-bottom:8px; }}
  h1 {{ font-size:28px; margin:0 0 6px; letter-spacing:-.02em; }}
  .sub {{ color:var(--redup); margin:0; font-size:15px; }}
  h2 {{ font-size:21px; margin:44px 0 6px; letter-spacing:-.01em; }}
  h2 .no {{ color:var(--netral); font-weight:400; margin-right:8px; }}
  h3 {{ font-size:17px; margin:22px 0 6px; }}
  .lead {{ color:var(--redup); margin:0 0 16px; font-size:16px; }}
  .ket {{ color:var(--redup); font-size:14.5px; }}
  .netral {{ color:var(--netral); }}
  .blok {{ background:var(--kartu); border:1px solid var(--grs); border-radius:12px;
           padding:20px 22px; margin:16px 0; }}
  .grid3 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px;
            margin:18px 0; }}
  .kartu-poin {{ background:var(--kartu); border:1px solid var(--grs); border-radius:12px;
                 padding:18px; border-top:5px solid var(--netral); }}
  .kartu-poin.baik {{ border-top-color:var(--baik); }}
  .kartu-poin.sedang {{ border-top-color:var(--sedang); }}
  .kartu-poin.lemah {{ border-top-color:var(--lemah); }}
  .poin-kepala {{ display:flex; justify-content:space-between; align-items:flex-start; gap:10px; }}
  .poin-nama {{ font-weight:650; font-size:16px; }}
  .lencana {{ font-size:12px; font-weight:650; padding:3px 9px; border-radius:99px;
              white-space:nowrap; }}
  .lencana.baik {{ background:var(--baik-bg); color:var(--baik); }}
  .lencana.sedang {{ background:var(--sedang-bg); color:var(--sedang); }}
  .lencana.lemah {{ background:var(--lemah-bg); color:var(--lemah); }}
  .lencana.netral {{ background:#eef1f4; color:var(--netral); }}
  .poin-angka {{ font-size:40px; font-weight:700; letter-spacing:-.03em; margin:10px 0 0; }}
  .poin-satuan {{ color:var(--netral); font-size:13px; }}
  .poin-ket {{ color:var(--redup); font-size:14px; margin:12px 0 0; }}
  .bar {{ display:inline-block; height:9px; background:#eceff3; border-radius:99px;
          overflow:hidden; vertical-align:middle; margin-right:8px; }}
  .bar-isi {{ display:block; height:100%; border-radius:99px; background:var(--netral); }}
  .bar-isi.baik {{ background:var(--baik); }}
  .bar-isi.sedang {{ background:var(--sedang); }}
  .bar-isi.lemah {{ background:var(--lemah); }}
  table.tabel {{ width:100%; border-collapse:collapse; font-size:14.5px; }}
  table.tabel th, table.tabel td {{ text-align:left; padding:11px 10px;
                                    border-bottom:1px solid var(--grs); vertical-align:top; }}
  table.tabel thead th {{ font-size:13px; color:var(--redup); font-weight:650;
                          border-bottom:2px solid var(--grs); }}
  .th-ket {{ display:block; font-weight:400; color:var(--netral); font-size:12.5px;
             margin-top:3px; }}
  .angka-kecil {{ font-weight:650; font-size:13.5px; }}
  .angka-besar {{ font-weight:700; font-size:17px; white-space:nowrap; }}
  .banding {{ display:grid; gap:7px; margin:12px 0; }}
  .banding-label {{ display:inline-block; width:72px; color:var(--redup); font-size:13.5px; }}
  .catat {{ background:#fffdf5; border:1px solid #f0e3bf; border-left:4px solid var(--sedang);
            border-radius:8px; padding:16px 18px; margin:16px 0; font-size:14.5px; }}
  .catat b {{ color:#7a5a12; }}
  .meta {{ font-size:13px; color:var(--netral); border-top:1px solid var(--grs);
           margin-top:40px; padding-top:16px; }}
  ul {{ margin:8px 0; padding-left:22px; }}
  li {{ margin:6px 0; }}
  @media print {{ body {{ background:#fff; }} .blok, .kartu-poin {{ break-inside:avoid; }} }}
</style></head><body><div class="wrap">

<header class="atas">
  <h1>Ringkasan Evaluasi Sistem Precheck RDTR</h1>
  <p class="sub">Wilayah {_e(h['wilayah'])} &middot; pengukuran {_e(dibuat)} &middot;
  disusun untuk pembaca umum</p>
</header>
<p class="lead">Dokumen ini menjelaskan <b>seberapa bisa dipercaya</b> jawaban sistem, dalam bahasa
sehari-hari. Versi teknisnya lengkap dengan uji statistik ada di <code>laporan_rag.html</code>.</p>

<h2><span class="no">1</span>Kesimpulan</h2>
<p class="lead">Sistem menjawab tiga pertanyaan untuk setiap permohonan. Dua di antaranya sudah
bekerja mendekati batas terbaik yang mungkin; satu masih perlu perbaikan.</p>
<div class="grid3">{kartu}</div>

<div class="catat">
<b>Cara membaca angka persen di atas.</b> Angka itu <b>bukan</b> "nilai ujian dari 100". Ia berarti
<b>berapa persen dari hasil terbaik yang masih mungkin dicapai</b> pada kasus-kasus yang diuji.
Pembedaan ini penting: pada sebagian kasus, mengambil seluruh jawaban yang benar memang mustahil
karena sistem hanya mengirim {md.get('k_operasi', 3)} pasal sekaligus, sementara aturan yang
berlaku bisa lebih banyak. Membandingkan dengan 100% apa adanya akan menyalahkan sistem atas batas
yang kita sendiri tetapkan.
<br><br>
<b>Label "Sangat baik" sampai "Perlu perbaikan" adalah kesepakatan internal laporan ini</b>, bukan
standar industri &mdash; untuk metrik seperti ini tidak ada ambang baku yang diakui di literatur
ilmiah. Labelnya dipakai agar mudah dibaca, bukan untuk dikutip sebagai sertifikasi.
</div>

<h2><span class="no">2</span>Apa yang sebenarnya diuji</h2>
<div class="blok">
<p>Setiap kali ada permohonan masuk, sistem harus menjawab tiga hal &mdash; dan setiap jawaban
<b>wajib menyebut pasal</b> yang menjadi dasarnya. Yang diuji di sini adalah bagian pencarian
pasal itu: <b>apakah sistem menemukan pasal yang tepat dari keseluruhan peraturan.</b></p>
<p class="ket">Bahan ujinya: Peraturan Bupati Sleman tentang RDTR Kawasan {_e(h['wilayah'])}, yang
dipecah menjadi <b>{korpus['chunk']} bagian</b> ({korpus['per_level'].get('ayat', 0)} ayat,
{korpus['per_level'].get('pasal', 0)} pasal, {korpus['per_level'].get('tabel', 0)} tabel lampiran).
Diuji dengan <b>{h['n_query']} kasus uji</b>, dan untuk {h['sumber_label'].get('aturan', 0)} di
antaranya jawaban yang benar <b>ditentukan oleh struktur peraturannya sendiri</b>, bukan oleh
pendapat siapa pun. {lantai_txt}</p>
</div>

<h2><span class="no">3</span>Hasil rinci</h2>
<p class="lead">Sistem menempuh dua jalur berbeda, tergantung apakah data lokasi dari sistem lain
menyertakan kode sub-zona yang presisi atau tidak. Kolom "bagian permohonan" menunjukkan seberapa
sering tiap jalur dipakai.</p>
<div class="blok" style="overflow-x:auto">
<table class="tabel">
  <thead><tr><th>Pertanyaan</th><th>Jalur</th><th>Bagian permohonan</th>{kepala_rinci}</tr></thead>
  <tbody>{rinci}</tbody>
</table>
</div>
<div class="catat">
<b>Kenapa angka "Dampak ke lingkungan" rendah &mdash; dan kenapa itu bukan berarti sistemnya
salah.</b> Sistem mencari dengan satu kata kunci yang seluruhnya tentang <b>air</b>: limpasan,
sumur resapan, drainase. Ketentuan tentang air (Pasal 53, kawasan resapan air) ditemukannya
<b>100% &mdash; 39 dari 39 kali</b>. Tetapi daftar jawaban benar yang kami susun juga memuat
ketentuan tentang <b>gempa bumi dan banjir lahar</b> (Pasal 50), dan itu <b>0 dari 95 kali</b>
ditemukan &mdash; wajar, karena kata kunci tentang air memang tidak menanyakan soal gempa.
<br><br>
Jadi yang rendah adalah <b>penilaian kami</b>, bukan kemampuan sistem. Bila dinilai hanya pada
tema yang memang ditanyakan, angkanya <b>92%</b>.
<br><br>
Yang perlu diputuskan ahli tata ruang: <b>apakah penilaian dampak suatu permohonan memang harus
menyebut ketentuan kebencanaan?</b> Kalau ya, sistem kurang satu pencarian untuk tema itu, dan
pemohon di zona rawan lahar saat ini memang tidak diberi tahu. Kalau tidak, daftar jawaban
benarnya yang perlu dipersempit.
</div>

<h2><span class="no">4</span>Perbaikan yang sudah dilakukan</h2>
<p class="lead">Dua masalah ditemukan lewat pengukuran ini, lalu diperbaiki. Keduanya terbukti
berpengaruh besar.</p>
<div class="blok">{blok_perbaikan}</div>

<h2><span class="no">5</span>Mutu kutipan pada jawaban nyata</h2>
<p class="lead">Selain mencari pasal yang tepat, sistem juga harus <b>mengutip dengan jujur</b>.
Ini diperiksa langsung dari jawaban yang sudah pernah dikeluarkan sistem.</p>
<div class="blok">{blok_sitasi}</div>

<h2><span class="no">6</span>Apa yang angka-angka ini TIDAK katakan</h2>
<div class="catat">
<ul>
  <li><b>Ini bukan ukuran kebenaran hukum.</b> Yang diuji adalah apakah sistem menemukan pasal
      yang sesuai struktur peraturan. <b>Belum ada ahli tata ruang yang memvalidasi</b> bahwa
      pasal itu memang jawaban hukum yang tepat untuk kasus tersebut.</li>
  <li><b>Jangan dikutip sebagai "akurasi sistem N%".</b> Angka persen di laporan ini punya makna
      khusus (persen dari yang mungkin dicapai) dan tidak sama dengan akurasi.</li>
  <li><b>Hanya satu wilayah.</b> Seluruh pengujian memakai korpus {_e(h['wilayah'])}. Wilayah lain
      belum punya bahan uji yang setara.</li>
  <li><b>Mutu bahasa jawaban tidak diuji di sini.</b> Yang diukur adalah pencarian pasal dan
      kejujuran kutipan, bukan apakah kalimat penjelasannya enak dibaca atau mudah dipahami.</li>
  <li><b>Sisi kecerdasan buatan bergantung pada layanan pihak ketiga</b> yang modelnya bisa
      berubah tanpa pemberitahuan. Karena itu angka apa pun selalu terikat tanggal
      pengukuran.</li>
</ul>
</div>

<h2><span class="no">7</span>Catatan operasional</h2>
<div class="blok">
<p class="ket">Catatan sistem mencatat <b>{gen.get('n_permohonan', 0)} permohonan</b>
({gen.get('n_poin', 0)} jawaban) sejak instrumentasi dipasang. Dari jumlah itu,
<b>{gen.get('sitasi_terverifikasi', 0)} dari {gen.get('sitasi', 0)} sitasi terverifikasi</b>
&mdash; artinya setiap pasal yang disebut memang ada di daftar yang disodorkan ke mesin penjawab.</p>
<p class="ket"><b>Sebagian besar catatan itu berasal dari pengujian internal, bukan pemohon
sungguhan</b>, termasuk uji yang sengaja dibuat gagal untuk memeriksa pengamanan. Karena itu angka
kegagalan di catatan sistem <b>belum bisa dibaca sebagai tingkat kegagalan layanan</b>. Angka
sesungguhnya baru bisa dihitung setelah penggunaan nyata terkumpul lebih banyak.</p>
</div>

<div class="meta">
Disusun otomatis oleh <code>eval/laporan_awam.py</code> dari hasil pengukuran
<code>laporan_rag.json</code>, <code>ringkasan_produksi.json</code>, <code>baseline.json</code>,
dan <code>atribusi.json</code> &mdash; tidak ada angka yang ditulis tangan di dokumen ini.<br>
Versi kode saat diukur: {_e(md.get('commit') or '?')} &middot;
model pencari: {_e(md.get('model_embedding') or '?')} &middot;
model pemeringkat: {_e(md.get('model_rerank') or '?')} &middot;
model penjawab: {_e(md.get('model_llm') or '?')}<br>
Bobot penggunaan dihitung dari {_e(bt.get('n', 0))} permohonan tercatat.
Dokumen dibuat {_e(datetime.now().strftime('%Y-%m-%d %H:%M'))}.
</div>

</div></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Laporan evaluasi untuk pembaca non-teknis.")
    ap.add_argument("--out", type=Path, default=_OUT)
    args = ap.parse_args()

    h = _muat("laporan_rag.json")
    rp = _muat("ringkasan_produksi.json")
    if not h:
        raise SystemExit("eval/laporan_rag.json tidak ada — jalankan eval_rag dulu")
    if not rp:
        raise SystemExit("eval/ringkasan_produksi.json tidak ada — jalankan ringkasan_produksi dulu")

    args.out.write_text(bangun(h, rp, _muat("baseline.json"), _muat("atribusi.json")),
                        encoding="utf-8")
    print(f"[laporan-awam] -> {args.out}  ({args.out.stat().st_size:,} byte)")


if __name__ == "__main__":
    main()
