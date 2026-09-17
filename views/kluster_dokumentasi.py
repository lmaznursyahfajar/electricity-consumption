"""View: Kluster & Dokumentasi — profil hasil segmentasi + referensi metodologi."""

from datetime import datetime

import pandas as pd
import streamlit as st

from src import config, styling, visualizations as viz
from src.cluster_insights import build_cluster_profiles

styling.hero_header(
    "📋 Kluster & Dokumentasi",
    "Profil tiap segmen hasil SOM beserta rekomendasi tindak lanjut berbasis data, "
    "dilengkapi dokumentasi metodologi dan kamus data.",
    eyebrow="Menu 3 dari 3",
)

tab_kluster, tab_dok = st.tabs(["📋 Analisis Kluster", "📚 Dokumentasi"])


# =============================================================================
# TAB 1 — Analisis Kluster
# =============================================================================
def render_kluster() -> None:
    if st.session_state.get("df_with_clusters") is None:
        st.warning("Latih model terlebih dahulu di menu **Model & Visualisasi** (tab Pelatihan Model).")
        return

    df = st.session_state.df_with_clusters
    cluster_col = "customer_segment" if "customer_segment" in df.columns else "cluster"
    profiles = build_cluster_profiles(df, cluster_col=cluster_col)
    profile_by_id = {p.cluster_id: p for p in profiles}

    styling.section_title("📊", "Ringkasan Tiap Segmen",
                           f"{len(profiles)} segmen pelanggan — tiap pelanggan masuk tepat satu segmen")
    summary_df = pd.DataFrame([{
        "Segmen": p.cluster_id,
        "Persona": p.persona,
        "Pelanggan": p.n_customers,
        "% Pelanggan": round(p.pct_customers, 1),
        "Konsumsi Rata-rata (kWh)": round(p.avg_consumption, 1),
        "vs Rata-rata Populasi": f"{p.consumption_vs_overall_pct:+.0f}%",
        "Kontribusi Total (%)": round(p.pct_of_total_consumption, 1),
        "Peringkat Kontribusi": f"#{p.consumption_rank}/{p.n_total_clusters}",
        "Tipe Dominan": p.dominant_type or "—",
        "Wilayah Dominan": p.dominant_region or "—",
    } for p in profiles]).sort_values("Peringkat Kontribusi")
    st.dataframe(summary_df, width="stretch", height=min(38 * (len(summary_df) + 1), 420))
    st.caption(f"Total: {summary_df['% Pelanggan'].sum():.0f}% pelanggan (setiap pelanggan dihitung "
                "di tepat satu segmen berdasarkan pola konsumsi yang paling sering muncul).")

    styling.section_title("🔍", "Analisis Detail per Segmen")
    clusters = sorted(df[cluster_col].unique())
    selected_cluster = st.selectbox("Pilih segmen:", clusters,
                                     format_func=lambda c: f"Segmen {c} — {profile_by_id[c].persona}")
    profile = profile_by_id[selected_cluster]
    cluster_data = df[df[cluster_col] == selected_cluster]

    st.markdown(
        f"### Segmen {selected_cluster} {styling.badge(profile.persona, 'success' if profile.consumption_vs_overall_pct > 25 else 'neutral')}",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jumlah Pelanggan", f"{profile.n_customers:,}", f"{profile.pct_customers:.1f}% dari total")
    k2.metric("Konsumsi Rata-rata", f"{profile.avg_consumption:,.0f} kWh",
              f"{profile.consumption_vs_overall_pct:+.0f}% vs populasi")
    k3.metric("Total Konsumsi", f"{profile.total_consumption:,.0f} kWh",
              f"{profile.pct_of_total_consumption:.1f}% dari total")
    k4.metric("Peringkat Kontribusi", f"#{profile.consumption_rank} / {profile.n_total_clusters}")

    if "customer_type" in cluster_data.columns and "region" in cluster_data.columns:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(viz.pie_distribution(cluster_data, "customer_type", None,
                                                  f"Distribusi Tipe Pelanggan — Segmen {selected_cluster}"),
                             width="stretch")
        with c2:
            st.plotly_chart(viz.pie_distribution(cluster_data, "region", None,
                                                  f"Distribusi Wilayah — Segmen {selected_cluster}"),
                             width="stretch")

    if "day_of_week" in cluster_data.columns:
        weekly = cluster_data.groupby("day_of_week")["consumption_kwh"].mean().reset_index()
        day_names = {0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis", 4: "Jumat", 5: "Sabtu", 6: "Minggu"}
        weekly["Hari"] = weekly["day_of_week"].map(day_names)
        st.plotly_chart(viz.bar_by_category(weekly, "Hari", "consumption_kwh",
                                             f"Rata-rata Konsumsi per Hari — Segmen {selected_cluster}"),
                         width="stretch")

    if "is_anomaly" in cluster_data.columns:
        anomaly_counts = cluster_data["is_anomaly"].value_counts().reindex([0, 1], fill_value=0)
        anomaly_df = pd.DataFrame({"Status": ["Normal", "Anomali"], "Jumlah": anomaly_counts.values})
        st.plotly_chart(viz.pie_distribution(anomaly_df, "Status", "Jumlah",
                                              f"Data Normal vs Anomali — Segmen {selected_cluster}"),
                         width="stretch")
        if anomaly_counts.get(1, 0) > 0:
            anomalies = cluster_data[cluster_data["is_anomaly"] == 1]
            st.caption(f"Ditemukan **{len(anomalies):,}** baris anomali pada segmen ini (10 pertama):")
            show_cols = [c for c in ["customer_id", "date", "consumption_kwh", "z_score"] if c in anomalies.columns]
            st.dataframe(anomalies[show_cols].head(10), width="stretch")

    styling.section_title("💡", f"Rekomendasi untuk Segmen {selected_cluster}",
                           "Dihitung dari statistik nyata segmen ini vs seluruh populasi pelanggan")
    for rec in profile.recommendations:
        st.markdown(f"✅ {rec}")

    styling.section_title("⬇️", "Ekspor Hasil")
    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Unduh Data + Segmen (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"data_dengan_segmen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", width="stretch",
        )
    with e2:
        st.download_button(
            "Unduh Ringkasan Segmen (CSV)",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"ringkasan_segmen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", width="stretch",
        )


# =============================================================================
# TAB 2 — Dokumentasi
# =============================================================================
def render_dokumentasi() -> None:
    sub1, sub2, sub3, sub4 = st.tabs(["Metodologi", "Kamus Data", "Arsitektur Proyek", "Batasan & Etika Data"])

    with sub1:
        st.markdown(
            """
### Apa itu Self-Organizing Map (SOM)?

SOM (Kohonen, 1982) adalah metode pembelajaran tanpa pengawasan (*unsupervised
learning*) yang memetakan data berdimensi tinggi ke grid neuron berdimensi
rendah (biasanya 2D), sambil mempertahankan hubungan ketetanggaan
(*topology-preserving*). Data yang mirip akan dipetakan ke neuron yang berdekatan.

**Tahapan dalam aplikasi ini:**

1. **Data & Praproses** — memuat & memfilter data, feature engineering, menangani
   *missing value*, dan normalisasi (MinMax/Standard Scaling).
2. **Pelatihan SOM** — inisialisasi bobot neuron secara acak dari data, lalu
   diperbarui secara iteratif: pada tiap iterasi, satu sampel data dipilih acak,
   neuron "pemenang" (BMU — *Best Matching Unit*) dicari, lalu bobot BMU dan
   tetangganya digeser mendekati sampel tersebut.
3. **Evaluasi model:**
   - **Quantization Error** — rata-rata jarak antara setiap data dan BMU-nya.
   - **Topographic Error** — proporsi data yang BMU pertama & keduanya tidak
     bertetangga (mengindikasikan seberapa baik topologi data dipertahankan).
4. **Klasterisasi dua-tahap (SOM + K-Means)** — grid SOM biasanya berisi puluhan
   hingga ratusan neuron, terlalu banyak untuk dianalisis satu per satu sebagai
   segmen bisnis. Aplikasi ini menjalankan **K-Means** pada *vektor bobot* tiap
   neuron (bukan pada data mentah) untuk mengelompokkan neuron-neuron yang mirip
   menjadi sejumlah kecil **segmen akhir** (dapat diatur, mis. 10) — teknik yang
   diperkenalkan Vesanto & Alhoniemi (2000). Neuron yang bertetangga pada grid
   SOM cenderung masuk segmen akhir yang sama, sehingga topologi yang dipelajari
   SOM tetap termanfaatkan.
5. **Kluster & Dokumentasi** — klasterisasi pada langkah 4 berjalan per **baris
   data harian**, sehingga satu pelanggan bisa punya hari-hari yang jatuh ke
   segmen berbeda (mis. pola weekday vs weekend yang cukup jauh berbeda).
   Untuk analisis bisnis, tiap pelanggan diberi **satu** label segmen akhir —
   diambil dari segmen yang paling sering muncul (modus) di antara hari-hari
   pelanggan tersebut — lalu diprofilkan (persona, statistik dibanding
   populasi keseluruhan) dan diberi rekomendasi yang dihitung langsung dari
   angka-angka tersebut.

### Mengapa hasilnya bisa berbeda-beda?

SOM bersifat stokastik (inisialisasi & urutan sampel acak) dan sensitif
terhadap parameter (ukuran grid, sigma, learning rate, jumlah iterasi).
K-Means pada tahap kedua juga bergantung pada jumlah segmen akhir yang dipilih.
Bandingkan beberapa kombinasi parameter dan gunakan quantization/topographic
error sebagai acuan, bukan satu-satunya patokan.
            """
        )

    with sub2:
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
            ("hasil analisis", "avg/std/min/max/total_consumption", "Numerik", "Statistik historis per pelanggan"),
            ("hasil analisis", "z_score", "Numerik", "Standar skor konsumsi harian relatif terhadap riwayat"),
            ("hasil analisis", "neuron_id", "Kategori", "ID neuron SOM pemenang (BMU) — level paling detail"),
            ("hasil analisis", "neuron_x / neuron_y", "Numerik", "Koordinat neuron pemenang pada grid SOM"),
            ("hasil analisis", "cluster", "Kategori", "ID segmen akhir level-HARIAN (hasil K-Means pada neuron)"),
            ("hasil analisis", "customer_segment", "Kategori", "ID segmen akhir level-PELANGGAN (modus dari `cluster` di seluruh hari pelanggan tsb.) — dipakai di menu Kluster & Dokumentasi"),
        ], columns=["Tabel/Sumber", "Kolom", "Tipe", "Deskripsi"])
        st.dataframe(kamus, width="stretch", height=460)

    with sub3:
        st.markdown(
            """
### Struktur Proyek

```
electricity-som-analytics/
├── app.py                        # Entry point — routing 4 menu via st.navigation
├── views/                        # Satu file = satu menu di sidebar
│   ├── beranda.py
│   ├── data_praproses.py         # Tab: Eksplorasi & Filter | Praproses & Fitur
│   ├── model_visualisasi.py      # Tab: Pelatihan Model | Visualisasi Hasil
│   └── kluster_dokumentasi.py    # Tab: Analisis Kluster | Dokumentasi
├── src/                          # Logika inti — bebas dari Streamlit, bisa diuji mandiri
│   ├── config.py                 # Semua konstanta & parameter
│   ├── data_generator.py         # Pembangkit dataset sintetis (vektorized)
│   ├── data_loader.py            # Memuat data bawaan/unggahan + validasi skema
│   ├── preprocessing.py          # Merge, feature engineering, missing value, scaling
│   ├── som_model.py              # Wrapper pelatihan SOM + klasterisasi dua-tahap (K-Means)
│   ├── cluster_insights.py       # Profil, persona & rekomendasi berbasis data per segmen
│   ├── visualizations.py         # Semua fungsi pembuat grafik
│   └── styling.py                # Sistem desain (CSS, kartu, badge, header)
├── scripts/
│   └── build_excel_workbook.py   # Membuat workbook Excel dari data CSV
├── assets/                       # Logo aplikasi (SVG)
├── data/                         # Dataset (fact + dimension tables, CSV & XLSX)
├── requirements.txt
└── README.md
```

### Alasan desain

- **4 menu, bukan 6** — dua pasang halaman yang berkaitan erat (eksplorasi ↔
  praproses, pelatihan ↔ visualisasi) digabung memakai tab agar navigasi
  lebih ringkas tanpa mengurangi fitur.
- **`st.navigation`/`st.Page`** (API multipage modern Streamlit) dipakai agar
  nama file di folder `views/` bisa bersih & deskriptif, sementara judul dan
  ikon di sidebar diatur terpisah lewat `st.Page(..., title=..., icon=...)`.
- **Skema fact/dimension** menghindari duplikasi data pelanggan pada tiap
  baris transaksi, mencerminkan praktik pemodelan data yang lazim di industri.
- **`src/` bebas dari Streamlit** (kecuali `styling.py`) sehingga fungsi inti
  dapat diuji atau dipakai ulang di luar aplikasi web.
            """
        )

    with sub4:
        st.markdown(
            """
### Batasan Dataset & Model

- **Dataset bawaan bersifat sintetis**, dibuat untuk keperluan latihan/edukasi.
  Pola musiman, hari libur, dan anomali disimulasikan secara sederhana dan
  **tidak merepresentasikan data pelanggan PLN yang sebenarnya**.
- **Angka tarif (Rp/kWh)** bersifat ilustratif untuk simulasi dan **bukan
  tarif resmi** yang berlaku — jangan dijadikan rujukan bisnis atau kebijakan.
- **SOM adalah metode klasterisasi/eksplorasi**, bukan model prediktif. Hasil
  klaster perlu divalidasi dengan pengetahuan domain sebelum dipakai
  mengambil keputusan operasional.
- **Deteksi anomali** di sini berbasis simulasi label & z-score sederhana,
  bukan sistem deteksi kecurangan/pencurian listrik yang sudah divalidasi.
- Jika mengganti dengan data pelanggan nyata, pastikan proses tersebut
  mematuhi **regulasi perlindungan data pribadi** yang berlaku (mis.
  anonimisasi ID pelanggan) sebelum data diunggah ke lingkungan bersama.
            """
        )

    st.caption(f"Versi aplikasi: {config.APP_VERSION}")


with tab_kluster:
    render_kluster()

with tab_dok:
    render_dokumentasi()
