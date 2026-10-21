# Permintaan dan konfirmasi untuk BE (2026-10-21)

Konteks: keluaran L3 sudah mengikuti "Spesifikasi Kontrak Variabel Output AI Reasoning (L3)" v1.0.0
(`docs/KONTRAK_L3_BE_V1.md`, contoh JSON di `docs/contoh_output/`). Butir di bawah adalah hal yang perlu BE
konfirmasi atau perbaiki di sisi L2. Tidak ada yang menghalangi sistem berjalan; semuanya menyangkut kebenaran data.

## 1. Perlu konfirmasi FE

Bentuk keluaran `POST /reasoning` berubah dan sudah aktif di produksi:

| Sebelumnya | Sekarang |
|---|---|
| `poin` daftar datar (itbx, intensitas, dampak) | `poin` pohon: akar `gate_hukum` dan `gate_hidrologis`, anak lewat `sub_poin` |
| `rekomendasi_sistem` = `Tidak Setuju` | `Tolak`, dan nilai baru `Lolos Terbatas` |
| `rekomendasi.langkah_konkret` = array objek | array **string**; objek berangka pindah ke `langkah_terstruktur` |
| `rekomendasi_per_zona` tidak ada | selalu ada (satu zona = satu entri) |

Id simpul daun: lahan satu zona `itbx_evaluasi`, `intensitas_evaluasi`, `dampak`. Lahan multi-zona: simpul
`zona_P` dengan anak `itbx_P`, `intensitas_P` (kode zona di belakang). Mohon FE memastikan render sudah sesuai.

## 2. Data L2 yang perlu diperbaiki atau dikonfirmasi

**a. `impact_assessment.rekomendasi_mitigasi` untuk dampak Rendah berisi angka tak bermakna.**
Contoh APP-2026-0022 (dampak RENDAH, indeks 1,013): `target_kategori: SEDANG`, `luas_bangunan_maks_m2: 10960.79`
(sama dengan seluruh luas persil), `luas_rth_min_m2: 0`, `kdb_maks_persen: 100`, `dimensi_minimum: 0 m³`, dan `saran`
yang berbunyi "Wajib memperluas RTH menjadi 9.864,71 m²". Dampak Rendah tidak butuh mitigasi korektif, jadi blok ini
sebaiknya `null` (seperti pada APP-2026-0014). L3 kini mengabaikan blok ini untuk dampak Rendah, tetapi payload
tetap membingungkan pihak lain yang membacanya.

**b. Zona berstatus ITBX X pada lahan multi-zona.** Mohon `intensitas_zones` untuk zona itu tetap dikirim dengan
`status` yang jelas (mis. `TIDAK_DIEVALUASI`) dan `parameter` kosong. Bila entrinya tidak ada, L3 menandai intensitas zona
itu "Tidak Dinilai" (aman), tetapi status jelas dari BE lebih baik daripada ditebak.

**c. KLB maksimum zona P-1: 1 atau 0,1?** Payload APP-2026-0014 dan APP-2026-0022 mengirim `klb.ambang_maks: 1` untuk
P-1 (Tanaman Pangan, resapan, jalan lokal). Tabel Lampiran VI yang kami ingest memuat P-1: "KDB maksimum 10%, KLB maksimum 0.1,
KDH minimum 85-88%". L3 memakai angka BE apa adanya; mohon dicek mana yang benar.

**d. `dasar_regulasi` menyebut "Pasal 53 & Pasal 20" untuk ketentuan resapan air.** Di Perbup 80/2023, Pasal 20 mengatur
Zona Ruang Terbuka Hijau (sub-zona RTH-2, RTH-3, RTH-4, RTH-7), bukan resapan air. Ketentuan khusus resapan ada di Pasal 53
(dan daftar kategori ketentuan khusus di Pasal 47). Mohon dikoreksi. L3 tidak memakai teks ini; sitasi diambil dari korpus RDTR.

**e. Arti `rekomendasi_mitigasi.dimensi_minimum`.** Contoh: `nilai: 0.62, satuan: m³` dengan rumus "modifikasi Sunjoto / SNI 8445:2017".
Mohon dikonfirmasi: apakah itu volume tampungan minimum satu sumur/kolam untuk seluruh persil, atau per unit? Dan kenapa bernilai 0 pada
dampak Rendah (lihat butir a)? L3 menuliskannya sebagai "kapasitas minimum X m³".

**f. Kegiatan dan tipe bangunan di dalam `impact_assessment`.** Kegiatan dan KBLI saat ini kami ambil dari `gate_hukum.tahapan.itbx`,
dan kelas atap dari `detailed_surface_breakdown`. Usul: kirim eksplisit di `impact_assessment` field `kegiatan_diusulkan`,
`kbli_diusulkan`, `jumlah_lantai`, dan `kelas_atap`, agar dampak tidak bergantung pada ITBX (terutama untuk lahan multi-zona,
yang ITBX-nya berbeda per zona).

**g. Syarat ITBX bernilai angka tidak dievaluasi.** APP-2026-0014: ITBX B dengan syarat "luas maksimal persil yang diperbolehkan
300 m²", sedangkan luas tanah pemohon 1.024 m², namun gate berstatus Lolos Bersyarat tanpa catatan pelanggaran. L3 hanya menuliskan syarat itu
sebagai "syarat yang harus dipenuhi pemohon" dan tidak menyatakannya terpenuhi. Mohon diputuskan: apakah BE akan mengevaluasi syarat
numerik seperti ini (lalu menandai tidak terpenuhi), atau dibiarkan sebagai syarat yang diverifikasi petugas?

## 2A. Perbedaan dengan contoh keluaran ideal BE (KBLI 56303, zona K-1)

Contoh ini dibandingkan dengan keluaran kita. Struktur, status, dan logika rekomendasi sudah cocok. Kita menambahkan field
`ringkasan_tindak_lanjut` di tingkat atas (sama isinya dengan `kesimpulan.ringkasan_tindak_lanjut`). Butir di bawah perlu diputuskan BE.

**a. `langkah_konkret` bertentangan dengan spesifikasi v1.0.0.** Pada simpul `dampak`, contoh memakai array objek
(`parameter`, `nilai_saat_ini`, `nilai_target`, `satuan`), sedangkan spesifikasi dan simpul `gate_hukum` memakai array string.
Kita mengirim string di `langkah_konkret` dan objeknya di `langkah_terstruktur`. Mohon pilih satu bentuk.

**b. Format status dampak tidak konsisten.** Contoh ini `"Tinggi"`, contoh BE sebelumnya `SEDANG` huruf kapital. Kita memakai huruf kapital.

**c. Angka limpasan tidak cocok.** Teks menulis "2.65x lipat (C 0.32 menjadi 0.74)", padahal 0,74 / 0,32 = 2,31. Bila 2,65 adalah indeks
dengan rumus lain, jangan dinarasikan sebagai kenaikan C. Mohon rumus indeks dijelaskan.

**d. Isi yang tidak ada di data L2.** Spesifikasi melarang mengarang, jadi kita tidak menirunya:
- "area peresapan alami turun 45%".
- Dokumen SPPL/AMDAL/UKL-UPL, IPAL, PBG/SLF lewat SIMBG, patok batas tanah, akses pengawasan Pemkab.
- "Dampak Sempadan Bangunan: pelanggaran batas sempadan" (tidak ada data pelanggaran sempadan).
- "berpotensi memicu genangan atau banjir lokal" dan "batas aman (< 2.5)" tanpa dasar di fakta.

Bila butir-butir ini memang diinginkan, mohon dikirim sebagai data di L2 (mis. syarat ITBX dari `keterangan_ketentuan`) atau diputuskan PM
sebagai teks standar.

**e. Format `saran` tidak seragam.** Sebagian butir `- ...`, sebagian kalimat biasa (intensitas). Kita selalu memakai butir.

**f. `dasar_regulasi` sangat tipis.** Contoh hanya memuat judul umum tanpa pasal. Kita mengirim sitasi terverifikasi lengkap (pasal, halaman,
kutipan) di `sitasi[]`. FE sebaiknya tidak bergantung pada `dasar_regulasi` yang lebih sederhana.

## 3. Dasar aturan mitigasi dampak (bukan masalah, hanya permintaan rujukan)

Kata "wajib" pada mitigasi dampak Sedang/Tinggi/Sangat Tinggi ("Wajib menurunkan KDB dan meningkatkan KDH/RTH") dan ambang indeks limpasan
(Rendah < 1,5, Sedang 1,5-2,5, Tinggi 2,5-4,0, Sangat Tinggi > 4,0; target mitigasi ke 2,5) berasal dari metodologi hidrologi BE dan keputusan
PM, **bukan** dari teks RDTR. RDTR hanya memuat batas KDB/KDH (Lampiran VI) dan ketentuan resapan (Pasal 53). Karena setiap keluaran
harus bersitasi, mohon BE/PM menyebut peraturan atau dokumen kebijakan yang menjadi dasar ambang dan kewajiban itu (mis. perda,
SNI 8445:2017, atau juknis), supaya bisa dicantumkan sebagai dasar regulasi mitigasi.

## 4. Perubahan perilaku yang perlu diketahui

- Dampak **Sedang** kini memerlukan mitigasi dan menjadikan rekomendasi `Setuju Bersyarat` (sesuai bagian 4 spesifikasi).
- ITBX **T** menghasilkan `Lolos Terbatas`, walau gate BE melabelinya `Lolos` atau `Lolos Bersyarat`. ITBX **X** menghasilkan `Tolak`.
- Mitigasi dampak di lokasi non-resapan tidak menyebut sumur/kolam resapan (aturan PM); di lokasi resapan, sumur/kolam disebut sebagai sarana
  memenuhi zero delta Q (Pasal 53 ayat 3).
- Status simpul daun mengikuti contoh BE: ITBX `Bersyarat (B)`, dampak `SEDANG`, intensitas `MELAMPAUI_BATAS`.
