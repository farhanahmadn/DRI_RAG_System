# NOTA KESEPAKATAN & KONFIRMASI TEKNIS
**Dari:** Tim Backend (BE) Urban Digital Triplet  
**Kepada:** Tim AI / L3 (Advisory Engine)  
**Perihal:** Tanggapan Resmi, Klarifikasi Teknis, dan Kesepakatan atas Dokumen `PERMINTAAN_BE_L3_2026-10-21.md`  
**Tanggal:** 21 Oktober 2026  
**Status Dokumen:** FINAL & DISETUJUI  

---

## 1. Ringkasan Eksekutif
Dokumen ini merupakan tanggapan resmi dan kesepakatan teknis dari tim Backend (BE) terhadap poin-poin masukan, temuan perbedaan, dan permintaan penyesuaian payload L2 dari tim AI (L3) sebagaimana tercantum dalam dokumen kerja `PERMINTAAN_BE_L3_2026-10-21.md`. 

Secara prinsip, **BE menyepakati hampir seluruh usulan penyempurnaan dari tim AI/L3** demi meningkatkan akurasi data hidrologi, konsistensi hukum tata ruang, dan keandalan sistem perizinan mandiri (*self-assessment*).

---

## 2. Tanggapan Data L2 yang Perlu Diperbaiki atau Dikonfirmasi

### 2.a. `impact_assessment.rekomendasi_mitigasi` untuk Dampak Rendah
- **Status:** **SEPAKAT (Disetujui)**
- **Penjelasan & Best Practice:**
  - Pemohon dengan kategori dampak **RENDAH** (indeks limpasan $< 1,5$) secara hidrologis telah memenuhi batas aman dan tidak memerlukan tindakan korektif fisik.
  - Nilai `kdb_maks: 100%`, `dimensi_minimum: 0 m³`, dan saran korektif yang muncul sebelumnya berasal dari eksekusi *fallback template* lama.
  - **Keputusan:** Untuk permohonan berstatus dampak **RENDAH**, blok `impact_assessment.rekomendasi_mitigasi` akan dikirim bernilai **`null`** secara konsisten.

---

### 2.b. Zona Berstatus ITBX 'X' pada Lahan Multi-Zona
- **Status:** **SEPAKAT (Disetujui)**
- **Penjelasan & Best Practice:**
  - Menghilangkan entri zona 'X' dari `intensitas_zones` menyebabkan inkonsistensi struktur array per zona dan memaksa L3 menebak statusnya.
  - **Keputusan:** BE akan tetap menyertakan sub-zona berstatus 'X' di dalam array `intensitas_zones` dengan format status eksplisit:
    - `status`: `"TIDAK_DIEVALUASI"`
    - `alasan`: `"Kegiatan berstatus X (Dilarang) pada zona ini, sehingga intensitas ruang tidak dievaluasi."`
    - `parameter`: `{}` (objek kosong / `null`)

---

### 2.c. Ambang Maksimum KLB Zona P-1: 1 atau 0,1?
- **Status:** **SEPAKAT dengan Tim L3 (Angka yang benar adalah `0,1`)**
- **Penjelasan & Best Practice:**
  - Berdasarkan **Perbup Sleman No. 80 Tahun 2023 Lampiran VI**, Sub-Zona Pertanian Tanaman Pangan (**P-1**) memiliki batas:
    - KDB maksimum = **$10\%$ ($0,10$)**
    - Ketinggian bangunan maksimum = **$1$ lantai**
    - KDH minimum = **$85\% - 88\%$**
  - Berdasarkan rumus baku tata ruang:
    $$\text{KLB} = \text{KDB} \times \text{Jumlah Lantai} = 0,10 \times 1 = \mathbf{0,10}$$
  - Angka `1` sebelumnya muncul akibat pembulatan *integer* pada atribut tabel spasial GIS RDTR. BE telah mengoreksi nilai ambang batas KLB untuk sub-zona P-1 menjadi **`0.1`**.

---

### 2.d. Koreksi Rujukan Regulasi Resapan Air (`dasar_regulasi`)
- **Status:** **SEPAKAT (Disetujui)**
- **Penjelasan & Best Practice:**
  - **Pasal 20** Perbup Sleman 80/2023 mengatur peruntukan Zona Ruang Terbuka Hijau (RTH-2 s/d RTH-7), bukan resapan air.
  - Pengaturan Kawasan Resapan Air diatur dalam:
    - **Pasal 47:** Daftar dan kategori ketentuan khusus.
    - **Pasal 53:** Ketentuan Khusus Kawasan Resapan Air (*Zero Delta Q Policy*, sumur/kolam resapan, dan perlindungan air tanah).
  - **Keputusan:** Rujukan regulasi di BE diperbaiki dari `Pasal 53 & Pasal 20` menjadi **`Pasal 47 & Pasal 53`** Perbup Sleman No. 80 Tahun 2023.

---

### 2.e. Arti Parameter `rekomendasi_mitigasi.dimensi_minimum`
- **Status:** **KONFIRMASI L3 SUDAH SANGAT TEPAT (Volume Total Kumulatif Persil)**
- **Penjelasan & Best Practice:**
  - Nilai kapasitas ($X\text{ m}³$) adalah **volume tampungan/retensi minimum kumulatif untuk SELURUH PERSIL** yang diwajibkan guna menahan debit limpasan lebih ($\Delta Q$) akibat pembangunan atap/lahan terbangun.
  - Pemohon memiliki fleksibilitas teknis di lapangan untuk memenuhinya, baik dengan membuat **1 kolam retensi/tampungan** berkapasitas $X\text{ m}³$, maupun membaginya ke dalam **beberapa unit sumur resapan** (misalnya total 3 m³ dibagi menjadi 3 unit @ 1 m³).
  - Narasi L3 yang menuliskan *"kapasitas minimum X m³"* sudah **sangat tepat**.
  - Bernilai `0` pada dampak Rendah karena tidak ada selisih kelebihan debit limpasan ($\Delta Q \le 0$) yang perlu dipotong. Sejalan dengan poin (2.a), blok ini di-`null`-kan untuk dampak Rendah.

---

### 2.f. Field Eksplisit Kegiatan & Bangunan di dalam `impact_assessment`
- **Status:** **SEPAKAT (Sangat Disetujui)**
- **Penjelasan & Best Practice:**
  - Pemisahan tanggung jawab (*decoupling*) antara aspek fisik hidrologi dengan aspek legalitas zonasi sangat penting, khususnya pada lahan multi-zona di mana status ITBX terpecah per zona sedangkan bangunan pemohon adalah satu kesatuan fisik.
  - **Keputusan:** BE menambahkan 4 atribut berikut langsung ke dalam objek `impact_assessment`:
    - `kegiatan_diusulkan`: Nama kegiatan pemohon (string).
    - `kbli_diusulkan`: Kode KBLI 5 digit (string).
    - `jumlah_lantai`: Ketinggian lantai usulan (number).
    - `kelas_atap`: Jenis material/kelas penutup atap yang menentukan nilai $C_{\text{atap}}$ (string).

---

### 2.g. Evaluasi Numerik pada Syarat ITBX Bersyarat (B)
- **Status:** **DIJALANKAN SECARA HIBRIDA (Petugas dengan Asistensi Flag Sistem)**
- **Penjelasan & Best Practice:**
  - Keterangan syarat pada ITBX matriks RDTR (seperti *"luas maksimal persil yang diperbolehkan 300 m²"*) berupa teks bebas dalam ribuan variasi redaksional regulasi, sehingga berisiko jika diputuskan mutlak secara otomatis tanpa verifikasi lapangan.
  - **Keputusan:**
    1. Status legalitas utama tetap **`Lolos Bersyarat`** (kegiatan bersyarat 'B' sah diajukan namun memerlukan pembuktian syarat teknis).
    2. BE meneruskan teks syarat tersebut secara utuh pada `keterangan_ketentuan`.
    3. L3 menarasikannya sebagai: **"Syarat teknis khusus yang wajib dipenuhi pemohon dan diverifikasi dokumen pembuktiannya oleh petugas PTSP/DPUPKP pada saat pemeriksaan berkas lanjutan."**

---

## 2A. Perbedaan dengan Contoh Keluaran Ideal BE (KBLI 56303, Zona K-1)

### 2A.a. Format `langkah_konkret` vs `langkah_terstruktur`
- **Status:** **SEPAKAT**
- **Keputusan:**
  - Pertahankan **`langkah_konkret`** sebagai array string (`string[]`) untuk menjaga konsistensi dengan *node* hukum `gate_hukum`.
  - Sediakan array objek pada **`langkah_terstruktur`** (`object[]` dengan field `parameter`, `nilai_saat_ini`, `nilai_target`, `satuan`) agar Frontend dapat menampilkan tabel komparasi parameter terstruktur.

---

### 2A.b. Format Penulisan Status Dampak
- **Status:** **SEPAKAT STANDARISASI UPPERCASE**
- **Keputusan:** Seluruh status dan kategori dampak distandarkan menjadi huruf kapital:
  - `"RENDAH"`
  - `"SEDANG"`
  - `"TINGGI"`
  - `"SANGAT_TINGGI"`

---

### 2A.c. Sinkronisasi Narasi Angka Limpasan & Nilai C
- **Status:** **SEPAKAT (Harus Presisi Matematis)**
- **Keputusan:**
  - Rumus rasio limpasan adalah perbandingan langsung koefisien komposit:
    $$\text{Indeks Limpasan} = \frac{C_{\text{sesudah}}}{C_{\text{sebelum}}}$$
  - Narasi L3 harus selalu mencerminkan hasil bagi eksak tersebut (misal $0,74 / 0,32 = 2,31\times$ lipat). Tidak diperkenankan menampilkan angka fiktif yang tidak bersumber dari data kalkulasi L2.

---

### 2A.d. Larangan Halusinasi Fakta Non-L2
- **Status:** **SEPAKAT PENUH DENGAN PRINSIP ANTI-HALUSINASI L3**
- **Keputusan:**
  - AI L3 dilarang mengarang klaim teknis yang tidak ada pada data L2 (seperti *"peresapan turun 45%"* atau pelanggaran sempadan tanpa dasar spasial).
  - Untuk kewajiban administratif umum perizinan berusaha (AMDAL/UKL-UPL/SPPL, PBG/SLF via SIMBG, IPAL standar):
    - Diberikan label jelas sebagai **"Ketentuan Normatif Administratif Perizinan"** (kewajiban standar OSS/SIMBG), bukan hasil temuan simulasi teknis Triplet.

---

### 2A.e. Format Poin `saran`
- **Status:** **SEPAKAT STANDARISASI BULLET**
- **Keputusan:** Seluruh entri pada array atau daftar `saran` wajib diseragamkan menggunakan format *bullet point* (`- ...`).

---

### 2A.f. Tingkat Kedalaman `dasar_regulasi` dan `sitasi`
- **Status:** **SEPAKAT MEMAKAI SITASI LENGKAP L3**
- **Keputusan:**
  - Frontend diarahkan untuk memprioritaskan rendering array **`sitasi[]`** dari L3 (yang memuat pasal, ayat, dan kutipan teks hukum hasil RAG RDTR), sementara `dasar_regulasi` dari BE difungsikan sebagai referensi ringkas metadata.

---

## 3. Dasar Regulasi Mitigasi Dampak & Ambang Batas Hidrologi

Berikut adalah rujukan yuridis dan metodologis resmi untuk dicantumkan oleh tim L3 pada sitasi mitigasi:

1. **Kewajiban Zero Delta Q Policy & Peresapan Air Hujan:**
   - **Peraturan Bupati Sleman No. 80 Tahun 2023 tentang RDTR Kawasan Sleman Tengah**, **Pasal 53 ayat (3) huruf b**:
     > *"Penerapan prinsip zero delta Q policy bagi setiap kegiatan pembangunan..."*
   - **Standar Teknis Desain:**
     - **SNI 8445:2017**: *Tata cara perencanaan sumur resapan air hujan untuk lahan pekarangan*.
     - **SNI 03-2453-2002**: *Tata cara perencanaan sumur resapan air hujan untuk lahan permukiman*.

2. **Ambang Batas Indeks Limpasan (1.5, 2.5, 4.0) & Target Mitigasi 2,5:**
   - **Metodologi:** Mengacu pada pedoman standar internasional drainase perkotaan **Federal Highway Administration (FHWA), HEC-22 Urban Drainage Design Manual, 4th Edition (2013)**, Section 3.2.2.1 (*Runoff Coefficient*).
   - **Kebijakan Teknis Daerah:** Merupakan turunan teknis kajian tata kelola drainase berkelanjutan (*Sustainable Urban Drainage Systems / SUDS*) DPUPKP Kabupaten Sleman untuk mencegah beban berlebih (*overcapacity*) pada saluran drainase makro akibat alih fungsi lahan.

---

## 4. Konfirmasi Perubahan Perilaku Sistem

BE mengonfirmasi dan mencatat perubahan perilaku sistem sebagai berikut:
1. **Dampak SEDANG:** Menghasilkan rekomendasi **`Setuju Bersyarat`** (karena memerlukan penyediaan sarana tampungan/drainase lingkungan).
2. **Status Keputusan ITBX:**
   - ITBX **T (Terbatas)** $\rightarrow$ **`Lolos Terbatas`**
   - ITBX **X (Dilarang)** $\rightarrow$ **`Tolak`**
3. **Pembedaan Kawasan Resapan vs Non-Resapan:**
   - Lokasi **Resapan Air:** Wajib fasilitas sumur/kolam resapan (*Zero Delta Q Policy* - Pasal 53).
   - Lokasi **Non-Resapan:** Menitikberatkan pada saluran drainase lingkungan dan retensi tampungan tanpa kewajiban sumur resapan air tanah.
4. **Format Simpul Daun:** Seragam huruf kapital / kode baku (`SEDANG`, `MELAMPAUI_BATAS`, `Bersyarat (B)`).

---
*Dokumen ini diterbitkan oleh Tim Backend Urban Digital Triplet sebagai panduan acuan bersama demi kelancaran integrasi L2, L3, dan Frontend.*
