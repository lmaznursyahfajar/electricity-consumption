"""Halaman: Ringkasan Data — eksplorasi dataset, filter, dan unggah data sendiri."""

import streamlit as st

from src import config, styling, visualizations as viz
from src.data_generator import generate_full_dataset
from src.data_loader import load_bundled_dataset, load_uploaded_dim_pelanggan, load_uploaded_fact
from src.preprocessing import merge_dataset

st.set_page_config(page_title="Ringkasan Data", page_icon="📊", layout="wide")
styling.inject_global_css()
styling.page_header("📊 Ringkasan Data", "Eksplorasi dataset konsumsi listrik sebelum dianalisis lebih lanjut.")

if "fact" not in st.session_state:
    st.warning("Data belum dimuat. Silakan buka halaman **Beranda** terlebih dahulu.")
    st.stop()

fact = st.session_state.fact
dim_pelanggan = st.session_state.dim_pelanggan
dim_wilayah = st.session_state.dim_wilayah

# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------
with st.expander("🔎 Filter Data", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        regions = st.multiselect("Wilayah", sorted(dim_pelanggan["region"].unique()))
    with c2:
        types = st.multiselect("Tipe Pelanggan", sorted(dim_pelanggan["customer_type"].unique()))
    with c3:
        date_range = st.date_input(
            "Rentang Tanggal",
            value=(fact["date"].min().date(), fact["date"].max().date()),
        )

filtered_customers = dim_pelanggan.copy()
if regions:
    filtered_customers = filtered_customers[filtered_customers["region"].isin(regions)]
if types:
    filtered_customers = filtered_customers[filtered_customers["customer_type"].isin(types)]

filtered_fact = fact[fact["customer_id"].isin(filtered_customers["customer_id"])]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered_fact = filtered_fact[(filtered_fact["date"].dt.date >= start) & (filtered_fact["date"].dt.date <= end)]

merged_preview = merge_dataset(filtered_fact, filtered_customers, dim_wilayah)

# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------
st.subheader("Statistik Dataset (sesuai filter)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Jumlah Pelanggan", f"{filtered_customers['customer_id'].nunique():,}")
k2.metric("Jumlah Baris", f"{len(filtered_fact):,}")
k3.metric("Total Konsumsi (kWh)", f"{filtered_fact['consumption_kwh'].sum():,.0f}")
missing_pct = filtered_fact["consumption_kwh"].isna().mean() * 100
k4.metric("% Data Kosong (consumption_kwh)", f"{missing_pct:.2f}%")

st.subheader("Pratinjau Data (tabel fakta digabung dengan dimensi)")
st.dataframe(merged_preview.head(200), width='stretch', height=280)

# ---------------------------------------------------------------------------
# Visualisasi ringkasan
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    dist = filtered_customers.groupby("customer_type")["customer_id"].nunique().reset_index()
    dist.columns = ["Tipe Pelanggan", "Jumlah"]
    st.plotly_chart(viz.bar_by_category(dist, "Tipe Pelanggan", "Jumlah", "Distribusi Tipe Pelanggan",
                                         color="Tipe Pelanggan"), width='stretch')
with col2:
    region_consumption = merged_preview.groupby("region")["consumption_kwh"].sum().reset_index()
    region_consumption.columns = ["Wilayah", "Total Konsumsi (kWh)"]
    st.plotly_chart(viz.pie_distribution(region_consumption, "Wilayah", "Total Konsumsi (kWh)",
                                          "Distribusi Konsumsi per Wilayah"), width='stretch')

st.subheader("Tren Konsumsi Harian")
daily = filtered_fact.groupby("date", as_index=False)["consumption_kwh"].sum()
st.plotly_chart(viz.kpi_daily_trend(daily), width='stretch')

# ---------------------------------------------------------------------------
# Unduh dataset & unggah data sendiri
# ---------------------------------------------------------------------------
st.subheader("⬇️ Unduh Dataset Bawaan")
d1, d2, d3 = st.columns(3)
with d1:
    st.download_button("Unduh fact_konsumsi_harian.csv", data=fact.to_csv(index=False).encode("utf-8"),
                        file_name="fact_konsumsi_harian.csv", mime="text/csv", width='stretch')
with d2:
    st.download_button("Unduh dim_pelanggan.csv", data=dim_pelanggan.to_csv(index=False).encode("utf-8"),
                        file_name="dim_pelanggan.csv", mime="text/csv", width='stretch')
with d3:
    if config.EXCEL_WORKBOOK_PATH.exists():
        with open(config.EXCEL_WORKBOOK_PATH, "rb") as f:
            st.download_button("Unduh dataset_konsumsi_listrik.xlsx", data=f.read(),
                                file_name="dataset_konsumsi_listrik.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                width='stretch')

st.subheader("📤 Gunakan Data Anda Sendiri")
st.markdown(
    "Unggah tabel fakta minimal berkolom **`customer_id`, `date`, `consumption_kwh`**. "
    "Tabel dimensi pelanggan bersifat opsional (minimal `customer_id`, `customer_type`, `region`)."
)
u1, u2 = st.columns(2)
with u1:
    uploaded_fact = st.file_uploader("File data konsumsi (CSV/XLSX)", type=["csv", "xlsx"], key="upload_fact")
with u2:
    uploaded_dim = st.file_uploader("File data pelanggan (CSV/XLSX, opsional)", type=["csv", "xlsx"], key="upload_dim")

if uploaded_fact is not None:
    new_fact, errors = load_uploaded_fact(uploaded_fact)
    if errors:
        for e in errors:
            st.error(e)
    else:
        st.session_state.fact = new_fact
        if uploaded_dim is not None:
            new_dim, dim_errors = load_uploaded_dim_pelanggan(uploaded_dim)
            if dim_errors:
                for e in dim_errors:
                    st.warning(e)
            else:
                st.session_state.dim_pelanggan = new_dim
        st.session_state.data_source = f"unggahan pengguna ({uploaded_fact.name})"
        st.session_state.som_result = None  # reset model lama karena data berubah
        st.success(f"Data berhasil dimuat: {len(new_fact):,} baris. Lanjut ke halaman Praproses Data.")
        st.rerun()

st.subheader("🔁 Bangkitkan Ulang Dataset Contoh")
n_customers = st.slider("Jumlah pelanggan", 100, 2000, config.N_CUSTOMERS, step=50)
if st.button("Generate Dataset Baru"):
    with st.spinner("Membangkitkan dataset sintetis baru..."):
        new_fact, new_dim, new_wilayah = generate_full_dataset(n_customers=n_customers)
    st.session_state.fact = new_fact
    st.session_state.dim_pelanggan = new_dim
    st.session_state.dim_wilayah = new_wilayah
    st.session_state.data_source = f"sintetis baru ({n_customers} pelanggan)"
    st.session_state.som_result = None
    st.success("Dataset baru berhasil dibuat!")
    st.rerun()

styling.sidebar_footer()
