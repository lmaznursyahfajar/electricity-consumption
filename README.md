# ⚡ SIGAP Listrik — Sistem Analisis Pola Konsumsi Listrik

Aplikasi web (Streamlit) untuk **segmentasi pelanggan** dan **deteksi anomali**
pola konsumsi listrik menggunakan **Self-Organizing Map (SOM)**, dengan
dataset sintetis berskala besar (328.200 baris, skema data ternormalisasi)
dan tampilan dashboard yang ringkas serta profesional.

---

## Daftar Isi

1. [Fitur Utama](#fitur-utama)
2. [Struktur Proyek](#struktur-proyek)
3. [Instalasi & Menjalankan Aplikasi](#instalasi--menjalankan-aplikasi)
4. [Dataset](#dataset)
5. [Alur Analisis (4 Menu)](#alur-analisis-4-menu)
6. [Menggunakan Data Anda Sendiri](#menggunakan-data-anda-sendiri)
7. [Konfigurasi](#konfigurasi)
8. [Pengujian](#pengujian)
9. [Batasan & Catatan Etika Data](#batasan--catatan-etika-data)
10. [Lisensi & Kredit](#lisensi--kredit)

---

## Fitur Utama

- 🎯 **Navigasi ringkas — 4 menu**, bukan berbelas-belas halaman. Tahapan
  yang berkaitan erat (eksplorasi ↔ praproses, pelatihan ↔ visualisasi)
  digabung dalam tab agar tetap satu alur tanpa kehilangan fitur.
- 🎨 **Tampilan didesain khusus** — logo & branding, header bergradasi di
  tiap halaman, kartu KPI, badge status, dan tipografi yang konsisten.
  Bukan tampilan Streamlit default.
- 📊 **Dataset sintetis realistis** — 600 pelanggan × 18 bulan data harian,
  dengan pola musiman per wilayah, efek akhir pekan/hari libur nasional,
  *missing value*, dan anomali yang disimulasikan.
- 🗂️ **Skema data ternormalisasi** (fact + dimension table), tersedia
  dalam **CSV** (dataset lengkap) & **Excel** (sampel terformat + kamus
  data + ringkasan berformula).
- 🧩 **Kode modular** — logika inti (`src/`) terpisah dari tampilan
  (`views/`), dapat diuji & dipakai ulang di luar aplikasi web.
- 🧠 **Pelatihan SOM interaktif** dengan metrik evaluasi (Quantization
  Error, Topographic Error) dan **visualisasi lengkap**: Peta Segmen Akhir,
  U-Matrix, Component Planes, Hit Histogram, proyeksi PCA 2D/3D.
- 🧩 **Klasterisasi dua-tahap (SOM + K-Means)** — jumlah segmen akhir bisa
  diatur (default 10), jauh lebih ringkas dibanding memakai tiap neuron
  SOM sebagai kluster tersendiri (bisa ratusan). Tiap pelanggan diberi
  **satu** segmen (voting mayoritas dari hari-harinya) agar persentase
  antar segmen selalu menjumlah 100%.
- 📋 **Rekomendasi berbasis data nyata** — dihitung langsung dari statistik
  tiap segmen dibanding populasi keseluruhan (bukan teks generik): potensi
  penghematan kWh, kontribusi % terhadap total konsumsi & peringkatnya,
  tingkat anomali, pola weekday/weekend, load factor — lengkap dengan
  persona segmen otomatis (mis. "Industri Besar — Konsumsi Sangat Tinggi").
  Ekspor hasil ke CSV.
- 📚 **Dokumentasi terintegrasi** di dalam aplikasi (metodologi, kamus
  data, arsitektur, batasan model) — tidak perlu keluar dari aplikasi.
- ✅ **Diuji ujung-ke-ujung** memakai `streamlit.testing.v1.AppTest`
  (lihat bagian [Pengujian](#pengujian)).

---

## Struktur Proyek

```
electricity-som-analytics/
├── app.py                         # Satu-satunya entry point (routing 4 menu)
├── views/                         # Satu file = satu menu di sidebar
│   ├── beranda.py                 # Dashboard ringkas & alur kerja
│   ├── data_praproses.py          # Tab: Eksplorasi & Filter | Praproses & Fitur
│   ├── model_visualisasi.py       # Tab: Pelatihan Model | Visualisasi Hasil
│   └── kluster_dokumentasi.py     # Tab: Analisis Kluster | Dokumentasi
├── src/                           # Logika inti (bebas dari Streamlit)
│   ├── config.py                  # Semua konstanta & parameter terpusat
│   ├── data_generator.py          # Pembangkit dataset sintetis (vektorized)
│   ├── data_loader.py             # Muat data bawaan/unggahan + validasi skema
│   ├── preprocessing.py           # Merge, feature engineering, missing, scaling
│   ├── som_model.py               # Wrapper pelatihan SOM + klasterisasi dua-tahap (K-Means)
│   ├── cluster_insights.py        # Profil, persona & rekomendasi berbasis data per segmen
│   ├── visualizations.py          # Semua fungsi pembuat grafik
│   └── styling.py                 # Sistem desain (CSS, kartu, badge, header)
├── scripts/
│   └── build_excel_workbook.py    # Membuat workbook Excel dari data CSV
├── assets/                        # Logo aplikasi (SVG)
│   ├── logo.svg
│   └── logo_icon.svg
├── data/                          # Dataset (dibuat otomatis jika belum ada)
│   ├── fact_konsumsi_harian.csv   # Tabel fakta (328.200 baris)
│   ├── dim_pelanggan.csv          # Tabel dimensi pelanggan (600 baris)
│   ├── dim_wilayah.csv            # Tabel dimensi wilayah (8 baris)
│   └── dataset_konsumsi_listrik.xlsx  # Workbook Excel (sampel + dokumentasi)
├── .streamlit/
│   └── config.toml                # Tema warna aplikasi
├── requirements.txt
└── README.md
```

**Mengapa `app.py` + `views/`, bukan folder `pages/` klasik?**
Aplikasi ini memakai API multipage modern Streamlit (`st.navigation` +
`st.Page`, tersedia sejak Streamlit 1.36). Dengan API ini, nama file di
`views/` bisa bersih & deskriptif (`data_praproses.py`, bukan
`1_📊_Data_Praproses.py`), sementara judul dan ikon yang tampil di sidebar
diatur terpisah lewat `st.Page(..., title=..., icon=...)` di `app.py`.
`app.py` juga bertindak sebagai "router" yang memuat data awal & tema
sekali di awal, sebelum salah satu dari 4 menu dirender.

---

## Instalasi & Menjalankan Aplikasi

**Prasyarat:** Python 3.10 atau lebih baru.

```bash
# 1. Masuk ke folder proyek
cd electricity-som-analytics

# 2. (Opsional tapi disarankan) buat virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependensi
pip install -r requirements.txt

# 4. Jalankan aplikasi (app.py, BUKAN salah satu file di views/)
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.
Dataset bawaan akan otomatis dibangkitkan pada run pertama jika folder
`data/` belum berisi file CSV (dapat memakan waktu 1–2 detik).

---

## Dataset

Dataset dibagi menjadi tiga tabel bergaya *data warehouse*:

| Tabel | Baris | Deskripsi |
|---|---|---|
| `fact_konsumsi_harian.csv` | 328.200 | Data transaksional harian: konsumsi (kWh), suhu, beban puncak, status akhir pekan/libur/anomali |
| `dim_pelanggan.csv` | 600 | Data induk pelanggan: golongan tarif, daya terpasang, wilayah, tanggal pasang |
| `dim_wilayah.csv` | 8 | Data induk wilayah: provinsi, karakteristik suhu & musiman |
| `dataset_konsumsi_listrik.xlsx` | — | Workbook Excel: sampel 5.000 baris + tabel dimensi lengkap + kamus data + ringkasan berformula |

**Membangkitkan ulang dataset dengan parameter berbeda:**

```bash
# Ubah N_CUSTOMERS, START_DATE, END_DATE, dsb. di src/config.py, lalu:
python -m src.data_generator

# Setelah itu, perbarui juga workbook Excel:
python scripts/build_excel_workbook.py
```

> ⚠️ Seluruh data bersifat **sintetis** untuk keperluan latihan/edukasi.
> Angka tarif (Rp/kWh) bersifat **ilustratif**, bukan tarif resmi PLN.

---

## Alur Analisis (4 Menu)

| # | Menu | Isi (tab) |
|---|---|---|
| 1 | **🏠 Beranda** | Dashboard ringkas: KPI, cuplikan tren, panduan alur kerja |
| 2 | **📊 Data & Praproses** | *Eksplorasi & Filter* — jelajahi, filter, unduh, unggah data · *Praproses & Fitur* — sampling, pilih fitur, tangani missing value, normalisasi |
| 3 | **🧠 Model & Visualisasi** | *Pelatihan Model* — atur parameter & latih SOM · *Visualisasi Hasil* — U-Matrix, component planes, PCA |
| 4 | **📋 Kluster & Dokumentasi** | *Analisis Kluster* — profil & rekomendasi tiap kluster · *Dokumentasi* — metodologi, kamus data, arsitektur, batasan |

---

## Menggunakan Data Anda Sendiri

Di menu **Data & Praproses** (tab *Eksplorasi & Filter*), unggah file
CSV/XLSX Anda sendiri:

- **Wajib** — tabel data konsumsi dengan kolom minimal:
  `customer_id`, `date`, `consumption_kwh`
- **Opsional** — tabel data pelanggan dengan kolom minimal:
  `customer_id`, `customer_type`, `region`

Aplikasi akan memvalidasi skema sebelum data dipakai, dan menampilkan
pesan kesalahan yang jelas jika ada kolom wajib yang hilang.

---

## Konfigurasi

Seluruh parameter (jumlah pelanggan, rentang tanggal, daftar wilayah,
golongan tarif, hari libur, fitur default SOM, warna tema, dsb.)
terpusat di `src/config.py` — tidak ada nilai konstan yang tersebar di
file lain.

---

## Pengujian

Logika inti (`src/`) sengaja dipisah dari Streamlit sehingga dapat diuji
tanpa menjalankan server web:

```bash
# Uji sintaks seluruh file
python -m py_compile app.py src/*.py views/*.py scripts/*.py

# Uji headless aplikasi dengan streamlit.testing.v1.AppTest
python -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('app.py')
at.run()
assert not at.exception
print('OK')
"
```

Seluruh menu (termasuk pelatihan model, semua mode visualisasi, semua
kombinasi strategi *missing value*/normalisasi, dan unggah data kustom)
telah diuji dengan pendekatan di atas selama pengembangan.

---

## Batasan & Catatan Etika Data

- Dataset bawaan **sintetis**, bukan data pelanggan PLN yang sebenarnya.
- Angka tarif listrik **ilustratif**, bukan tarif resmi yang berlaku.
- SOM adalah metode eksplorasi/klasterisasi, **bukan model prediktif** —
  validasi hasil dengan pengetahuan domain sebelum dipakai mengambil
  keputusan operasional.
- Jika mengganti dengan data pelanggan nyata, pastikan proses tersebut
  mematuhi **regulasi perlindungan data pribadi** yang berlaku (mis.
  anonimisasi ID pelanggan) sebelum diunggah ke lingkungan bersama.

---

## Lisensi & Kredit

Dibangun dengan [Streamlit](https://streamlit.io),
[MiniSom](https://github.com/JustGlowing/minisom), pandas, NumPy,
scikit-learn, Plotly, dan Matplotlib. Dataset sepenuhnya sintetis dan
bebas dipakai untuk keperluan belajar, tugas kuliah, maupun portofolio.
