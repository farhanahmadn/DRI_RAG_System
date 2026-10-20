# Kontrak keluaran L3 (disesuaikan dengan spesifikasi BE v1.0.0, 2026-10-20)

Sumber: "Spesifikasi Kontrak Variabel Output AI Reasoning (L3)" dari BE. Dokumen ini menjelaskan bagaimana sistem
kita memenuhinya, apa yang sengaja berbeda, dan apa yang masih perlu dikonfirmasi. Dokumen ini menggantikan
`KONTRAK_L3_MULTI_ZONA.md`.

Endpoint `POST /reasoning` kini mengembalikan bentuk kontrak ini (`AdvisoryL3`, `app/kontrak_be.py`). Pipeline
internal tetap memakai daftar poin datar (`OutputL3`) dan dipetakan di akhir.

## 1. Kamus data: status pemenuhan

| Variabel spesifikasi | Status | Catatan |
|---|---|---|
| `rekomendasi_sistem` | Dipenuhi | Hanya `Setuju`, `Setuju Bersyarat`, `Lolos Terbatas`, `Tolak` (sebelumnya `Tidak Setuju`). |
| `kesimpulan.ringkasan` | Dipenuhi | 1-2 kalimat, ditulis LLM bersama langkah, dengan cadangan deterministik dari status poin. |
| `kesimpulan.langkah_berdampak` | Dipenuhi | Dirangkum dari saran per poin. |
| `kesimpulan.ringkasan_tindak_lanjut` | Dipenuhi | Salinan persis `langkah_berdampak` (spesifikasi: harus konsisten). |
| `kesimpulan.catatan_lokasi` | Dipenuhi | Dari data spasial (blok, SWP, resapan, koridor jalan, ketentuan khusus aktif), bukan tulisan LLM. |
| `ringkasan_gate.kalimat` | Dipenuhi | Kalimat gate + status ITBX dan intensitas (per zona bila multi-zona). Tambahan: `final_gate_status`, `decisive_stage`. |
| `ringkasan_dampak.impact_category`, `.kalimat` | Dipenuhi | Tambahan: `impact_score`. |
| `ringkasan_dampak.kawasan_label`, `.is_kawasan_resapan` | Dipenuhi | Label dari L2 (`kawasan_label`) atau nama kawasan hidrologi. |
| `ringkasan_dampak.poin` | Dipenuhi | Butir dari angka L2 apa adanya: kategori dan indeks, C sebelum dan sesudah, KDB dan KDH usulan, luas persil. |
| `narasi_rekomendasi.*` | Dipenuhi | Tidak berubah. |
| `rekomendasi_per_zona[]` | Dipenuhi | **Selalu terisi**: satu entri untuk lahan satu zona, satu per zona untuk multi-zona. |
| `poin[]` (pohon) | Dipenuhi | Akar `gate_hukum` dan `gate_hidrologis`, anak lewat `sub_poin` (lihat bagian 3). |

## 2. Logika rekomendasi (bagian 4 spesifikasi)

Urutan prioritas, kondisi yang lebih tinggi mengalahkan yang di bawahnya (`app/reasoning/rekomendasi.py`):

1. **Tolak**: gate `Tidak Lolos`, atau ITBX X.
2. **Lolos Terbatas**: gate `Lolos Terbatas`, atau ITBX T (juga TB). Mengalahkan Setuju Bersyarat walau BE melabeli gate `Lolos Bersyarat`.
3. **Setuju Bersyarat**: gate `Lolos Bersyarat`, ITBX B, atau dampak **Sedang/Tinggi/Sangat Tinggi**.
4. **Setuju**: semua legal I, intensitas patuh, dampak Rendah.

Pada lahan multi-zona, ITBX paling ketat di antara zona ikut menentukan `rekomendasi_sistem` keseluruhan. Rekomendasi per
zona hanya memakai ITBX dan intensitas zona itu (hidrologi satu untuk seluruh lahan, jadi tidak dipecah).

**Perubahan perilaku yang perlu diketahui**: dampak **Sedang** sebelumnya dianggap aman (tanpa mitigasi korektif).
Sesuai spesifikasi, kini Sedang memerlukan mitigasi: rekomendasi menjadi Setuju Bersyarat dan saran memuat langkah
mitigasi (target indeks turun ke kategori Rendah).

## 3. Pohon `poin`

Lahan satu zona:
```
poin[0] gate_hukum        (Penilaian Legal Tata Ruang)
          sub_poin: itbx_evaluasi, intensitas_evaluasi
poin[1] gate_hidrologis   (Penilaian Lingkungan (Hidrologi))
          sub_poin: dampak
```
Lahan multi-zona (>= 2 entri `gate_hukum.itbx_zones` dari L2):
```
poin[0] gate_hukum
          sub_poin: zona_P  -> sub_poin: itbx_P, intensitas_P
                    zona_R  -> sub_poin: itbx_R, intensitas_R
poin[1] gate_hidrologis -> sub_poin: dampak     (satu untuk seluruh lahan)
```
Status akar: `gate_hukum` = `LOLOS`, `LOLOS_BERSYARAT`, `LOLOS_TERBATAS`, `TIDAK_LOLOS`. `gate_hidrologis` =
`PERLU_MITIGASI` (dampak Sedang ke atas), `MEMENUHI_SYARAT` (Rendah), `TIDAK_DINILAI`. Status simpul daun tetap memakai
nilai asal (`I`/`B`/`T`/`X`, `MEMENUHI_SYARAT`/`MELAMPAUI_BATAS`, `Rendah`/`Sedang`/`Tinggi`/`Sangat Tinggi`).

Setiap simpul memuat `reasoning_pendek`, `reasoning_panjang`, `dasar_regulasi[]` {judul, deskripsi, kategori},
`rekomendasi` {saran, poin[], langkah_konkret[], alasan} dan `sub_poin[]`. Akar mengagregasi anak-anaknya
(dasar_regulasi digabung tanpa duplikat, butir dan langkah digabung berurutan).

## 4. Tambahan di luar spesifikasi (sengaja dipertahankan)

| Field | Alasan |
|---|---|
| `poin[].sitasi[]` | Sitasi terverifikasi lengkap (citation_id, pasal, halaman, kutipan). `dasar_regulasi` hanya memuat judul dan deskripsi pendek (300 karakter), plus `citation_id` dan `terverifikasi`. |
| `rekomendasi.langkah_terstruktur[]` | Langkah dengan angka (parameter, nilai_saat_ini, nilai_target, satuan). **`rekomendasi.langkah_konkret` kini array string** sesuai spesifikasi; objek lama dipindah ke field ini. |
| `rekomendasi.disclaimer`, `.tipe`, `.target` | Catatan batas lebih ketat, tipe rekomendasi, dan target numerik. |
| `poin[].zona`, `poin[].low_confidence` | Identitas zona poin, dan penanda penjelasan LLM yang tidak lolos validasi. |
| `rekomendasi_per_zona[].poin_ids`, `.low_confidence` | Tautan ke simpul daun zona itu. |
| `catatan_global`, `low_confidence_keseluruhan` | Catatan level permohonan (caveat BE, pemberitahuan multi-zona, poin yang perlu ditinjau manual). |

## 5. Perbedaan isi dengan contoh JSON ideal BE (disengaja)

Contoh BE memuat hal yang **tidak ada di payload L2**, sedangkan spesifikasi sendiri melarang mengarangnya ("jangan
mengarang data spasial, hasil perhitungan, kapasitas mitigasi, atau rujukan regulasi"). Sistem kita tidak menirunya:

- Dokumen SPPL, izin teknis sanitasi, PBG/SLF melalui SIMBG: tidak ada di fakta L2. Kita hanya menuliskan syarat yang
  ada di `keterangan_ketentuan` (dikutip persis).
- "Sumur resapan diameter 1 m kedalaman 3 m sebanyak 2 unit", paving berpori: tidak dihitung L2. Kita memakai
  `dimensi_minimum` dari BE bila ada.
- "Area peresapan alami turun 45%": tidak ada di L2. Butir dampak kita hanya memakai angka yang dikirim.

## 6. Pertanyaan terbuka untuk BE / PM

1. **Sumur di dampak non-resapan.** Contoh BE (dampak Sedang, non-resapan) merekomendasikan sumur resapan 6,50 m³,
   sedangkan aturan PM untuk Tinggi non-resapan hanya KDB turun dan RTH naik, tanpa sumur. Sistem mengikuti PM.
2. **Format `status`.** Contoh BE memakai `SEDANG` (kapital) dan `Bersyarat (B)`. Kita mengirim `Sedang` dan `B`.
   Bila BE/PDF membutuhkan format contoh, cukup satu pemetaan di `app/kontrak_be.py`.
3. **Dampak Rendah.** `rekomendasi_mitigasi` L2 untuk dampak Rendah berisi angka tak bermakna (lantai dasar maks
   10.960,79 m², dimensi 0 m³). Kita mengabaikannya; sebaiknya BE mengirim `null`.
4. **Zona berstatus ITBX X**: mohon `intensitas_zones` zona itu tetap dikirim dengan `status` yang jelas.
5. **Sumber kata "wajib" dan ambang indeks 2,5** untuk mitigasi dampak di luar RDTR (kebijakan atau peraturan apa).
6. **KLB P-1**: 1 (BE) atau 0,1 (Lampiran VI RDTR, resapan lokal).
