"""Halaman: Model SOM — konfigurasi & pelatihan Self-Organizing Map."""

import streamlit as st

from src import styling, visualizations as viz
from src.som_model import assign_clusters, suggest_grid_size, train_som

st.set_page_config(page_title="Model SOM", page_icon="🧠", layout="wide")
styling.inject_global_css()
styling.page_header("🧠 Model Self-Organizing Map", "Latih SOM untuk mengelompokkan pola konsumsi pelanggan.")

if st.session_state.get("X_scaled") is None:
    st.warning("Lakukan praproses data terlebih dahulu di halaman **🔧 Praproses Data**.")
    st.stop()

X_scaled = st.session_state.X_scaled
features = st.session_state.selected_features
n_samples = X_scaled.shape[0]
suggested = suggest_grid_size(n_samples)

st.caption(f"Data latih: {n_samples:,} baris × {len(features)} fitur.")

col1, col2 = st.columns(2)
with col1:
    grid_mode = st.radio("Ukuran grid SOM:", ["Otomatis", "Manual"], horizontal=True)
    if grid_mode == "Manual":
        grid_x = st.slider("Lebar grid (X)", 3, 30, suggested)
        grid_y = st.slider("Tinggi grid (Y)", 3, 30, suggested)
    else:
        grid_x = grid_y = None
        st.info(f"Ukuran grid otomatis (heuristik 5·√n): **{suggested} × {suggested}**")

    sigma = st.slider("Radius awal (sigma)", 0.1, 5.0, 1.0, 0.1,
                       help="Radius tetangga pada awal pelatihan.")

with col2:
    learning_rate = st.slider("Learning rate awal", 0.01, 1.0, 0.5, 0.01)
    iterations = st.slider("Jumlah iterasi", 200, 20000, 3000, 200,
                            help="Satu iterasi = satu sampel acak diproses. Tidak harus sama dengan jumlah baris data.")

if st.button("🚀 Latih Model SOM", type="primary", width='stretch'):
    progress_bar = st.progress(0.0)
    status = st.empty()

    def _on_progress(frac: float):
        progress_bar.progress(frac)
        status.text(f"Melatih SOM... {frac * 100:.0f}%")

    with st.spinner("Mempersiapkan pelatihan..."):
        result = train_som(
            X_scaled, grid_x=grid_x, grid_y=grid_y, sigma=sigma,
            learning_rate=learning_rate, iterations=iterations, on_progress=_on_progress,
        )
    status.text("Pelatihan selesai!")

    with st.spinner("Menetapkan kluster untuk setiap baris data..."):
        cluster_ids, neuron_x, neuron_y = assign_clusters(result.som, X_scaled)

    df_with_clusters = st.session_state.preprocessed_df.copy()
    df_with_clusters["cluster"] = cluster_ids
    df_with_clusters["neuron_x"] = neuron_x
    df_with_clusters["neuron_y"] = neuron_y

    st.session_state.som_result = result
    st.session_state.df_with_clusters = df_with_clusters

    st.success(f"Model SOM berhasil dilatih dengan grid {result.grid_x}×{result.grid_y}!")

result = st.session_state.get("som_result")
if result is not None:
    st.subheader("📈 Ringkasan Model Terlatih")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ukuran Grid", f"{result.grid_x}×{result.grid_y}")
    m2.metric("Jumlah Neuron", result.grid_x * result.grid_y)
    m3.metric("Quantization Error", f"{result.quantization_error:.4f}",
              help="Rata-rata jarak antara data dan neuron pemenangnya — makin kecil makin baik.")
    m4.metric("Topographic Error", f"{result.topographic_error:.4f}",
              help="Proporsi data yang neuron pertama & kedua terdekatnya tidak bertetangga — makin kecil makin baik.")

    df_with_clusters = st.session_state.df_with_clusters
    cluster_counts = df_with_clusters["cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Kluster", "Jumlah Data"]
    st.plotly_chart(
        viz.bar_by_category(cluster_counts.sort_values("Jumlah Data", ascending=False).head(20),
                             "Kluster", "Jumlah Data", "Distribusi Data per Kluster (Top 20)"),
        width='stretch',
    )
    st.caption("Lanjutkan ke halaman **📈 Visualisasi Hasil** dan **📋 Analisis Kluster** untuk eksplorasi mendalam.")

styling.sidebar_footer()
