"""View: Data & Praproses — eksplorasi dataset + persiapan fitur untuk SOM."""

import pandas as pd
import streamlit as st

from src import config, styling, visualizations as viz
from src.data_generator import generate_full_dataset
from src.data_loader import load_uploaded_dim_pelanggan, load_uploaded_fact
from src.preprocessing import (
    add_customer_stats,
    handle_missing,
    merge_dataset,
    sample_for_training,
    scale_features,
)

styling.hero_header(
    "📊 Data & Praproses",
    "Jelajahi dataset konsumsi listrik, lalu siapkan fitur numerik yang bersih dan "
    "ternormalisasi sebelum dimodelkan dengan SOM.",
    eyebrow="Menu 1 dari 3",
)

tab_eksplorasi, tab_praproses = st.tabs(["🔍 Eksplorasi & Filter", "🔧 Praproses & Fitur"])


# =============================================================================
# TAB 1 — Eksplorasi & Filter
# =============================================================================
def render_eksplorasi() -> None:
    fact = st.session_state.fact
    dim_pelanggan = st.session_state.dim_pelanggan
    dim_wilayah = st.session_state.dim_wilayah

    styling.section_title("🎛️", "Filter Data")
    with st.container(border=True):
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
        filtered_fact = filtered_fact[
            (filtered_fact["date"].dt.date >= start) & (filtered_fact["date"].dt.date <= end)
        ]

    merged_preview = merge_dataset(filtered_fact, filtered_customers, dim_wilayah)

    styling.section_title("📌", "Statistik Dataset", "Mengikuti filter di atas")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jumlah Pelanggan", f"{filtered_customers['customer_id'].nunique():,}")
    k2.metric("Jumlah Baris", f"{len(filtered_fact):,}")
    k3.metric("Total Konsumsi (kWh)", f"{filtered_fact['consumption_kwh'].sum():,.0f}")
    missing_pct = filtered_fact["consumption_kwh"].isna().mean() * 100
    k4.metric("% Data Kosong", f"{missing_pct:.2f}%")

    styling.section_title("📋", "Pratinjau Data", "Tabel fakta digabung dengan data pelanggan & wilayah")
    st.dataframe(merged_preview.head(200), width="stretch", height=260)

    styling.section_title("📈", "Visualisasi Ringkas")
    col1, col2 = st.columns(2)
    with col1:
        dist = filtered_customers.groupby("customer_type")["customer_id"].nunique().reset_index()
        dist.columns = ["Tipe Pelanggan", "Jumlah"]
        st.plotly_chart(viz.bar_by_category(dist, "Tipe Pelanggan", "Jumlah", "Distribusi Tipe Pelanggan",
                                             color="Tipe Pelanggan"), width="stretch")
    with col2:
        region_consumption = merged_preview.groupby("region")["consumption_kwh"].sum().reset_index()
        region_consumption.columns = ["Wilayah", "Total Konsumsi (kWh)"]
        st.plotly_chart(viz.pie_distribution(region_consumption, "Wilayah", "Total Konsumsi (kWh)",
                                              "Distribusi Konsumsi per Wilayah"), width="stretch")

    daily = filtered_fact.groupby("date", as_index=False)["consumption_kwh"].sum()
    st.plotly_chart(viz.kpi_daily_trend(daily), width="stretch")

    styling.section_title("⬇️", "Unduh Dataset Bawaan")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("fact_konsumsi_harian.csv", data=fact.to_csv(index=False).encode("utf-8"),
                            file_name="fact_konsumsi_harian.csv", mime="text/csv", width="stretch")
    with d2:
        st.download_button("dim_pelanggan.csv", data=dim_pelanggan.to_csv(index=False).encode("utf-8"),
                            file_name="dim_pelanggan.csv", mime="text/csv", width="stretch")
    with d3:
        if config.EXCEL_WORKBOOK_PATH.exists():
            with open(config.EXCEL_WORKBOOK_PATH, "rb") as f:
                st.download_button("dataset_konsumsi_listrik.xlsx", data=f.read(),
                                    file_name="dataset_konsumsi_listrik.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    width="stretch")

    styling.section_title("📤", "Gunakan Data Anda Sendiri")
    st.caption(
        "Unggah tabel fakta minimal berkolom **`customer_id`, `date`, `consumption_kwh`**. "
        "Tabel dimensi pelanggan bersifat opsional (minimal `customer_id`, `customer_type`, `region`)."
    )
    u1, u2 = st.columns(2)
    with u1:
        uploaded_fact = st.file_uploader("File data konsumsi (CSV/XLSX)", type=["csv", "xlsx"], key="upload_fact")
    with u2:
        uploaded_dim = st.file_uploader("File data pelanggan (CSV/XLSX, opsional)", type=["csv", "xlsx"],
                                         key="upload_dim")

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
            st.session_state.som_result = None
            st.success(f"Data berhasil dimuat: {len(new_fact):,} baris. Lanjut ke tab **Praproses & Fitur**.")
            st.rerun()

    with st.expander("🔁 Bangkitkan Ulang Dataset Contoh"):
        n_customers = st.slider("Jumlah pelanggan", 100, 2000, config.N_CUSTOMERS, step=50)
        if st.button("Generate Dataset Baru", width="stretch"):
            with st.spinner("Membangkitkan dataset sintetis baru..."):
                new_fact, new_dim, new_wilayah = generate_full_dataset(n_customers=n_customers)
            st.session_state.fact = new_fact
            st.session_state.dim_pelanggan = new_dim
            st.session_state.dim_wilayah = new_wilayah
            st.session_state.data_source = f"sintetis baru ({n_customers} pelanggan)"
            st.session_state.som_result = None
            st.success("Dataset baru berhasil dibuat!")
            st.rerun()


# =============================================================================
# TAB 2 — Praproses & Fitur
# =============================================================================
def render_praproses() -> None:
    with st.spinner("Menggabungkan tabel & menghitung fitur agregat pelanggan..."):
        merged = merge_dataset(st.session_state.fact, st.session_state.dim_pelanggan, st.session_state.dim_wilayah)
        merged = add_customer_stats(merged)

    styling.section_title("🧪", "Sampling", "Disarankan untuk dataset besar")
    st.caption(
        "SOM dilatih secara iteratif dengan mengambil satu sampel acak per iterasi, jadi kecepatan "
        "*training* tidak terlalu terpengaruh ukuran data. Namun **penetapan kluster untuk setiap baris "
        "setelah training** memproses seluruh baris — pada dataset besar ini bisa memakan waktu. "
        "Gunakan sampling agar proses tetap responsif."
    )
    use_sampling = st.checkbox("Gunakan sampel data (stratified per pelanggan)",
                                value=len(merged) > config.MAX_ROWS_FOR_TRAINING_DEFAULT)
    if use_sampling:
        max_rows = st.slider("Jumlah maksimum baris", 1000, min(100000, len(merged)),
                              min(config.MAX_ROWS_FOR_TRAINING_DEFAULT, len(merged)), step=1000)
        working_df = sample_for_training(merged, max_rows)
        st.caption(f"Menggunakan {len(working_df):,} dari {len(merged):,} baris total.")
    else:
        working_df = merged
        st.caption(f"Menggunakan seluruh {len(working_df):,} baris.")

    styling.section_title("🧮", "Pilih Fitur untuk Analisis SOM")
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
        return

    st.markdown(f"Fitur terpilih: {' '.join(styling.badge(f, 'neutral') for f in selected_features)}",
                unsafe_allow_html=True)

    styling.section_title("🩹", "Tangani Missing Value")
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
        width="stretch",
    )

    styling.section_title("📐", "Normalisasi Data")
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
        st.caption("**Sebelum normalisasi (5 baris pertama)**")
        st.dataframe(clean_df[selected_features].head(), width="stretch")
    with col2:
        st.caption("**Setelah normalisasi (5 baris pertama)**")
        st.dataframe(pd.DataFrame(X_scaled, columns=selected_features).head(), width="stretch")

    styling.section_title("📊", "Distribusi Sebelum vs Sesudah Normalisasi")
    feature_to_plot = st.selectbox("Pilih fitur untuk dibandingkan:", selected_features)
    idx = selected_features.index(feature_to_plot)
    st.plotly_chart(viz.histogram_before_after(clean_df[feature_to_plot], X_scaled[:, idx], feature_to_plot),
                     width="stretch")

    styling.section_title("🔗", "Matriks Korelasi Antar Fitur")
    st.plotly_chart(viz.correlation_heatmap(clean_df[selected_features]), width="stretch")

    st.success("✅ Praproses selesai. Data siap dilatih di menu **Model & Visualisasi**.")


with tab_eksplorasi:
    render_eksplorasi()

with tab_praproses:
    render_praproses()
