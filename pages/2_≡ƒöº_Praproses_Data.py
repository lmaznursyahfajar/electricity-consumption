"""Halaman: Praproses Data — pemilihan fitur, penanganan missing value, normalisasi, sampling."""

import streamlit as st

from src import config, styling, visualizations as viz
from src.preprocessing import (
    add_customer_stats,
    handle_missing,
    merge_dataset,
    sample_for_training,
    scale_features,
)

st.set_page_config(page_title="Praproses Data", page_icon="🔧", layout="wide")
styling.inject_global_css()
styling.page_header("🔧 Praproses Data", "Menyiapkan data numerik yang bersih dan ternormalisasi untuk SOM.")

if "fact" not in st.session_state:
    st.warning("Data belum dimuat. Silakan buka halaman **Beranda** terlebih dahulu.")
    st.stop()

with st.spinner("Menggabungkan tabel & menghitung fitur agregat pelanggan..."):
    merged = merge_dataset(st.session_state.fact, st.session_state.dim_pelanggan, st.session_state.dim_wilayah)
    merged = add_customer_stats(merged)

st.subheader("1️⃣ Sampling (opsional, disarankan untuk dataset besar)")
st.markdown(
    "SOM dilatih secara iteratif dengan mengambil satu sampel acak per iterasi, jadi kecepatan "
    "*training* tidak terlalu terpengaruh ukuran data. Namun **penetapan kluster untuk setiap baris "
    "setelah training** memproses seluruh baris — pada dataset besar ini bisa memakan waktu. "
    "Gunakan sampling di bawah ini agar proses tetap responsif."
)
use_sampling = st.checkbox("Gunakan sampel data (stratified per pelanggan)", value=len(merged) > config.MAX_ROWS_FOR_TRAINING_DEFAULT)
if use_sampling:
    max_rows = st.slider("Jumlah maksimum baris", 1000, min(100000, len(merged)),
                          min(config.MAX_ROWS_FOR_TRAINING_DEFAULT, len(merged)), step=1000)
    working_df = sample_for_training(merged, max_rows)
    st.caption(f"Menggunakan {len(working_df):,} dari {len(merged):,} baris total.")
else:
    working_df = merged
    st.caption(f"Menggunakan seluruh {len(working_df):,} baris.")

st.subheader("2️⃣ Pilih Fitur untuk Analisis SOM")
numeric_cols = working_df.select_dtypes(include="number").columns.tolist()
exclude_cols = {"is_anomaly", "z_score", "neuron_x", "neuron_y", "cluster"}
available_features = [c for c in numeric_cols if c not in exclude_cols]

method = st.radio("Metode pemilihan fitur:",
                   ["Fitur rekomendasi", "Pilih manual", "Semua fitur numerik"], horizontal=True)

if method == "Fitur rekomendasi":
    selected_features = [f for f in config.DEFAULT_SOM_FEATURES if f in available_features]
elif method == "Semua fitur numerik":
    selected_features = available_features
else:
    selected_features = st.multiselect("Pilih fitur:", available_features,
                                        default=[f for f in config.DEFAULT_SOM_FEATURES if f in available_features])

if not selected_features:
    st.warning("Pilih minimal satu fitur untuk melanjutkan.")
    st.stop()

st.info(f"Fitur terpilih: {', '.join(selected_features)}")

st.subheader("3️⃣ Tangani Missing Value")
missing_strategy = st.selectbox("Strategi penanganan missing value:",
                                 ["mean", "median", "drop_rows"],
                                 format_func=lambda x: {"mean": "Isi dengan rata-rata (mean)",
                                                         "median": "Isi dengan median",
                                                         "drop_rows": "Hapus baris yang kosong"}[x])
clean_df, reports = handle_missing(working_df, selected_features, strategy=missing_strategy)

report_rows = [(r.kolom, r.jumlah_missing, f"{r.persentase:.2f}%") for r in reports]
st.dataframe(
    {"Kolom": [r[0] for r in report_rows], "Jumlah Missing": [r[1] for r in report_rows],
     "Persentase": [r[2] for r in report_rows]},
    width='stretch',
)

st.subheader("4️⃣ Normalisasi Data")
scaling_method = st.selectbox("Metode normalisasi:",
                               ["minmax", "standard"],
                               format_func=lambda x: {"minmax": "MinMax Scaling (0–1)",
                                                       "standard": "Standard Scaling (mean=0, std=1)"}[x])
X_scaled, scaler = scale_features(clean_df, selected_features, method=scaling_method)

st.session_state.X_scaled = X_scaled
st.session_state.selected_features = selected_features
st.session_state.scaler = scaler
st.session_state.preprocessed_df = clean_df

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Sebelum normalisasi (5 baris pertama)**")
    st.dataframe(clean_df[selected_features].head(), width='stretch')
with col2:
    st.markdown("**Setelah normalisasi (5 baris pertama)**")
    import pandas as pd
    st.dataframe(pd.DataFrame(X_scaled, columns=selected_features).head(), width='stretch')

st.subheader("Distribusi Sebelum vs Sesudah Normalisasi")
feature_to_plot = st.selectbox("Pilih fitur untuk dibandingkan:", selected_features)
idx = selected_features.index(feature_to_plot)
st.plotly_chart(viz.histogram_before_after(clean_df[feature_to_plot], X_scaled[:, idx], feature_to_plot),
                 width='stretch')

st.subheader("Matriks Korelasi Antar Fitur")
st.plotly_chart(viz.correlation_heatmap(clean_df[selected_features]), width='stretch')

st.success("✅ Praproses selesai. Data siap dilatih di halaman **🧠 Model SOM**.")
styling.sidebar_footer()
