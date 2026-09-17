"""Halaman: Dokumentasi — metodologi, kamus data, arsitektur, dan batasan model."""

import pandas as pd
import streamlit as st

from src import config, styling

st.set_page_config(page_title="Dokumentasi", page_icon="📚", layout="wide")
styling.inject_global_css()
styling.page_header("📚 Dokumentasi Aplikasi", "Metodologi, kamus data, arsitektur, dan batasan model.")

tab1, tab2, tab3, tab4 = st.tabs(["Metodologi", "Kamus Data", "Arsitektur Proyek", "Batasan & Etika Data"])

with tab1:
    st.markdown(
        """
### Apa itu Self-Organizing Map (SOM)?

SOM (Kohonen, 1982) adalah metode pembelajaran tanpa pengawasan (*unsupervised learning*)
yang memetakan data berdimensi tinggi ke grid neuron berdimensi rendah (biasanya 2D),
sambil mempertahankan hubungan ketetanggaan (*topology-preserving*). Data yang mirip akan
dipetakan ke neuron yang berdekatan.

**Tahapan dalam aplikasi ini:**

1. **Ringkasan Data** — memuat & memfilter data konsumsi listrik.
2. **Praproses Data** — feature engineering (statistik agregat per pelanggan), penanganan
   *missing value*, dan normalisasi (MinMax/Standard Scaling) agar semua fitur numerik
   berkontribusi setara terhadap jarak Euclidean yang dipakai SOM.
3. **Model SOM** — inisialisasi bobot neuron secara acak dari data, lalu diperbarui secara
   iteratif: pada tiap iterasi, satu sampel data dipilih acak, neuron "pemenang" (BMU —
   *Best Matching Unit*) dicari, lalu bobot BMU dan tetangganya digeser mendekati sampel
   tersebut. Radius tetangga (*sigma*) dan *learning rate* mengecil seiring waktu.
4. **Evaluasi model:**
   - **Quantization Error** — rata-rata jarak antara setiap data dan BMU-nya (semakin kecil,
     semakin baik representasi data oleh peta).
   - **Topographic Error** — proporsi data yang BMU pertama & keduanya tidak bertetangga
     (mengindikasikan seberapa baik topologi data asli dipertahankan).
5. **Visualisasi & Interpretasi** — U-Matrix (batas kluster), component planes (kontribusi
   tiap fitur), dan proyeksi PCA sebagai pembanding independen dari SOM.
6. **Analisis Kluster** — neuron-neuron pada grid SOM dikelompokkan menjadi kluster
   (di aplikasi ini, tiap neuron = satu ID kluster), lalu diprofilkan berdasarkan
   karakteristik pelanggan yang dominan di dalamnya.

### Mengapa hasilnya bisa berbeda-beda?

SOM bersifat stokastik (inisialisasi & urutan sampel bersifat acak) dan sensitif terhadap
parameter (ukuran grid, sigma, learning rate, jumlah iterasi). Bandingkan beberapa kombinasi
parameter dan gunakan quantization/topographic error sebagai acuan, bukan satu-satunya patokan.
        """
    )

with tab2:
    st.markdown("### Kamus Data")
    kamus = pd.DataFrame([
        ("fact_konsumsi_harian", "customer_id", "Teks", "ID unik pelanggan (kunci ke Dim_Pelanggan)"),
        ("fact_konsumsi_harian", "date", "Tanggal", "Tanggal pencatatan konsumsi harian"),
        ("fact_konsumsi_harian", "consumption_kwh", "Numerik", "Konsumsi listrik harian (kWh); dapat kosong"),
        ("fact_konsumsi_harian", "temperature_c", "Numerik", "Suhu rata-rata harian sintetis (°C)"),
        ("fact_konsumsi_harian", "peak_load_kwh", "Numerik", "Estimasi beban puncak harian (kWh)"),
        ("fact_konsumsi_harian", "load_factor", "Numerik (0–1)", "Rasio beban rata-rata terhadap beban puncak"),
        ("fact_konsumsi_harian", "is_weekend", "0/1", "1 jika akhir pekan"),
        ("fact_konsumsi_harian", "is_holiday", "0/1", "1 jika hari libur nasional (simulasi)"),
        ("fact_konsumsi_harian", "is_anomaly", "0/1", "1 jika baris disuntik sebagai anomali"),
        ("dim_pelanggan", "customer_type", "Kategori", "Segmen pelanggan"),
        ("dim_pelanggan", "tariff_class", "Kategori", "Golongan tarif yang disederhanakan"),
        ("dim_pelanggan", "daya_terpasang_va", "Numerik", "Daya listrik terpasang (VA)"),
        ("dim_pelanggan", "tarif_rp_per_kwh", "Numerik", "Tarif ilustratif — BUKAN tarif resmi PLN"),
        ("dim_pelanggan", "region", "Kategori", "Kota/wilayah (kunci ke Dim_Wilayah)"),
        ("dim_wilayah", "suhu_baseline_c", "Numerik", "Suhu rata-rata tahunan baseline wilayah"),
        ("dim_wilayah", "amplitudo_musiman", "Numerik", "Faktor pengali variasi musiman"),
        ("hasil analisis", "avg/std/min/max/total_consumption", "Numerik", "Statistik historis per pelanggan (dihitung saat praproses)"),
        ("hasil analisis", "z_score", "Numerik", "Standar skor konsumsi harian relatif terhadap riwayat pelanggan"),
        ("hasil analisis", "cluster", "Kategori", "ID kluster hasil pemetaan SOM (indeks neuron pemenang)"),
    ], columns=["Tabel/Sumber", "Kolom", "Tipe", "Deskripsi"])
    st.dataframe(kamus, width='stretch', height=460)

with tab3:
    st.markdown(
        """
### Struktur Proyek

```
electricity-som-analytics/
├── Beranda.py                    # Entry point aplikasi (halaman utama)
├── pages/                        # Setiap file = satu halaman di sidebar
│   ├── 1_📊_Ringkasan_Data.py
│   ├── 2_🔧_Praproses_Data.py
│   ├── 3_🧠_Model_SOM.py
│   ├── 4_📈_Visualisasi_Hasil.py
│   ├── 5_📋_Analisis_Kluster.py
│   └── 6_📚_Dokumentasi.py
├── src/                          # Logika inti — bebas dari Streamlit, bisa diuji mandiri
│   ├── config.py                 # Semua konstanta & parameter
│   ├── data_generator.py         # Pembangkit dataset sintetis (vektorized)
│   ├── data_loader.py            # Memuat data bawaan / unggahan pengguna + validasi skema
│   ├── preprocessing.py          # Merge, feature engineering, missing value, scaling
│   ├── som_model.py              # Wrapper pelatihan & evaluasi SOM
│   ├── visualizations.py         # Semua fungsi pembuat grafik
│   └── styling.py                # CSS & komponen tampilan
├── scripts/
│   └── build_excel_workbook.py   # Membuat workbook Excel dari data CSV
├── data/                         # Dataset (fact + dimension tables, CSV & XLSX)
├── requirements.txt
└── README.md
```

### Alasan desain

- **Skema fact/dimension** dipilih dibanding tabel datar tunggal agar tidak ada duplikasi
  data pelanggan pada tiap baris transaksi — mengurangi ukuran file signifikan dan
  mencerminkan praktik pemodelan data yang lazim di industri.
- **`src/` bebas dari Streamlit** (kecuali `styling.py`) sehingga fungsi-fungsi inti
  (generator data, preprocessing, model SOM) dapat diuji atau dipakai ulang di luar
  aplikasi web, misalnya dari skrip batch atau notebook.
- **Multipage native Streamlit** (folder `pages/`) dipilih dibanding satu file panjang
  dengan `if/elif` agar navigasi otomatis tersedia di sidebar dan setiap halaman lebih
  mudah dirawat secara independen.
        """
    )

with tab4:
    st.markdown(
        """
### Batasan Dataset & Model

- **Dataset bawaan bersifat sintetis** (dibangkitkan dengan aturan matematis + noise acak),
  dibuat untuk keperluan latihan/edukasi. Pola musiman, hari libur, dan anomali disimulasikan
  secara sederhana dan **tidak merepresentasikan data pelanggan PLN yang sebenarnya**.
- **Angka tarif (Rp/kWh)** bersifat ilustratif untuk kebutuhan simulasi dan **bukan tarif
  resmi** yang berlaku — jangan dijadikan rujukan bisnis atau kebijakan.
- **SOM adalah metode klasterisasi/eksplorasi**, bukan model prediktif. Hasil klaster perlu
  divalidasi dengan pengetahuan domain sebelum dipakai mengambil keputusan operasional.
- **Deteksi anomali** di sini berbasis simulasi label (`is_anomaly`) dan z-score sederhana,
  bukan sistem deteksi kecurangan/pencurian listrik yang sudah divalidasi.
- Jika Anda mengganti dengan data pelanggan nyata, pastikan proses tersebut mematuhi
  **regulasi perlindungan data pribadi** yang berlaku (mis. anonimisasi ID pelanggan)
  sebelum data diunggah ke lingkungan bersama.
        """
    )

st.caption(f"Versi aplikasi: {config.APP_VERSION}")
styling.sidebar_footer()
