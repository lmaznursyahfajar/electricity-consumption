"""
Beranda.py
==========
Entry point aplikasi multipage Streamlit "SIGAP Listrik". Halaman ini
berfungsi sebagai landing page: ringkasan alur kerja, status data pada
sesi berjalan, dan tautan cepat ke setiap tahap analisis.

Menjalankan aplikasi:
    streamlit run Beranda.py
"""

import streamlit as st

from src import config, styling
from src.data_loader import load_bundled_dataset

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

styling.inject_global_css()
styling.page_header(
    f"{config.APP_ICON} {config.APP_TITLE}",
    "Segmentasi pelanggan, deteksi anomali, dan eksplorasi pola konsumsi listrik "
    "menggunakan Self-Organizing Map (SOM).",
)


def _ensure_session_data():
    if "fact" not in st.session_state:
        with st.spinner("Memuat dataset bawaan..."):
            fact, dim_pelanggan, dim_wilayah = load_bundled_dataset()
        st.session_state.fact = fact
        st.session_state.dim_pelanggan = dim_pelanggan
        st.session_state.dim_wilayah = dim_wilayah
        st.session_state.data_source = "bawaan"
    st.session_state.setdefault("som_result", None)
    st.session_state.setdefault("X_scaled", None)
    st.session_state.setdefault("selected_features", None)
    st.session_state.setdefault("df_with_clusters", None)


_ensure_session_data()

col1, col2, col3, col4 = st.columns(4)
fact = st.session_state.fact
dim_pelanggan = st.session_state.dim_pelanggan

with col1:
    st.metric("Jumlah Pelanggan", f"{dim_pelanggan['customer_id'].nunique():,}")
with col2:
    st.metric("Jumlah Baris Data", f"{len(fact):,}")
with col3:
    tanggal_min, tanggal_max = fact["date"].min(), fact["date"].max()
    st.metric("Rentang Tanggal", f"{tanggal_min.date()} → {tanggal_max.date()}")
with col4:
    status_model = "✅ Sudah dilatih" if st.session_state.som_result else "— Belum dilatih"
    st.metric("Status Model SOM", status_model)

st.caption(f"Sumber data saat ini: **{st.session_state.data_source}**. "
           "Ganti atau unggah data Anda sendiri di halaman '📊 Ringkasan Data'.")

st.markdown("### 🧭 Alur Kerja Analisis")
st.markdown(
    """
Gunakan menu di sidebar kiri untuk berpindah antar tahap. Alur yang disarankan:

1. **📊 Ringkasan Data** — jelajahi dataset, filter, atau unggah data sendiri.
2. **🔧 Praproses Data** — pilih fitur, tangani missing value, dan normalisasi.
3. **🧠 Model SOM** — atur parameter dan latih Self-Organizing Map.
4. **📈 Visualisasi Hasil** — lihat U-Matrix, component planes, dan proyeksi PCA.
5. **📋 Analisis Kluster** — profil tiap kluster pelanggan dan rekomendasi tindak lanjut.
6. **📚 Dokumentasi** — metodologi, kamus data, dan batasan model.
    """
)

st.info(
    "💡 **Tips:** dataset bawaan berisi ratusan ribu baris. Di halaman Praproses, "
    "aktifkan opsi *sampling* agar pelatihan SOM tetap cepat tanpa perlu memakai "
    "seluruh data sekaligus.",
    icon="💡",
)

styling.sidebar_footer()
