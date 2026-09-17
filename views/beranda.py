"""View: Beranda — dashboard ringkas & panduan alur kerja."""

import streamlit as st

from src import styling, visualizations as viz

fact = st.session_state.fact
dim_pelanggan = st.session_state.dim_pelanggan

styling.hero_header(
    "⚡ Selamat Datang di SIGAP Listrik",
    "Segmentasi pelanggan, deteksi anomali, dan eksplorasi pola konsumsi listrik "
    "menggunakan Self-Organizing Map (SOM) — didukung dataset sintetis berskala besar "
    "dengan skema data ternormalisasi.",
    eyebrow="Dashboard",
)

# ---------------------------------------------------------------------------
# KPI ringkas
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    with st.container(border=True):
        st.metric("Jumlah Pelanggan", f"{dim_pelanggan['customer_id'].nunique():,}")
with c2:
    with st.container(border=True):
        st.metric("Jumlah Baris Data", f"{len(fact):,}")
with c3:
    with st.container(border=True):
        tanggal_min, tanggal_max = fact["date"].min(), fact["date"].max()
        st.metric("Rentang Data", f"{(tanggal_max - tanggal_min).days} hari")
with c4:
    with st.container(border=True):
        status = "Terlatih ✅" if st.session_state.som_result else "Belum Dilatih"
        st.metric("Status Model SOM", status)

st.caption(f"Sumber data saat ini: **{st.session_state.data_source}**. "
           "Ganti atau unggah data Anda sendiri di menu **Data & Praproses**.")

# ---------------------------------------------------------------------------
# Cuplikan tren konsumsi
# ---------------------------------------------------------------------------
styling.section_title("📈", "Cuplikan Tren Konsumsi Harian")
daily = fact.groupby("date", as_index=False)["consumption_kwh"].sum()
st.plotly_chart(viz.kpi_daily_trend(daily), width="stretch")

# ---------------------------------------------------------------------------
# Alur kerja
# ---------------------------------------------------------------------------
styling.section_title("🧭", "Alur Kerja Analisis", "Empat menu di sidebar, mengikuti tahapan berikut")

steps = [
    ("Data & Praproses", "Jelajahi & filter dataset, tangani missing value, pilih fitur, dan normalisasi data."),
    ("Model & Visualisasi", "Latih Self-Organizing Map, lalu telaah U-Matrix, component planes, dan proyeksi PCA."),
    ("Kluster & Dokumentasi", "Lihat profil tiap kluster pelanggan, rekomendasi tindak lanjut, dan metodologi lengkap."),
]
cols = st.columns(3)
for i, (title, desc) in enumerate(steps, start=1):
    with cols[i - 1]:
        st.markdown(styling.flow_card(i, title, desc), unsafe_allow_html=True)

st.info(
    "💡 **Tips:** dataset bawaan berisi ratusan ribu baris. Di menu **Data & Praproses**, "
    "aktifkan opsi *sampling* agar pelatihan model tetap cepat tanpa perlu memakai "
    "seluruh data sekaligus.",
    icon="💡",
)
