"""View: Model & Visualisasi — pelatihan SOM + klasterisasi dua-tahap + interpretasi visual."""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from src import styling, visualizations as viz
from src.som_model import (
    assign_clusters,
    cluster_neurons,
    hit_map as compute_hit_map,
    map_to_final_clusters,
    suggest_grid_size,
    train_som,
    u_matrix as compute_u_matrix,
)

styling.hero_header(
    "🧠 Model & Visualisasi",
    "Latih Self-Organizing Map untuk mempelajari topologi pola konsumsi, lalu ringkas "
    "menjadi beberapa segmen akhir yang mudah diinterpretasi untuk analisis bisnis.",
    eyebrow="Menu 2 dari 3",
)

tab_model, tab_viz = st.tabs(["🧠 Pelatihan Model", "📈 Visualisasi Hasil"])


# =============================================================================
# TAB 1 — Pelatihan Model
# =============================================================================
def render_model() -> None:
    if st.session_state.get("X_scaled") is None:
        st.warning("Lakukan praproses data terlebih dahulu di menu **Data & Praproses** "
                    "(tab Praproses & Fitur).")
        return

    X_scaled = st.session_state.X_scaled
    features = st.session_state.selected_features
    n_samples = X_scaled.shape[0]
    suggested = suggest_grid_size(n_samples)

    st.caption(f"Data latih: {n_samples:,} baris × {len(features)} fitur.")

    styling.section_title("⚙️", "Parameter Pelatihan SOM")
    with st.container(border=True):
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
                                    help="Satu iterasi = satu sampel acak diproses.")

    styling.section_title("🧩", "Ringkas Menjadi Segmen Akhir",
                           "Neuron SOM dikelompokkan lagi memakai K-Means agar jumlah segmen ringkas")
    with st.container(border=True):
        st.caption(
            "Grid SOM bisa berisi puluhan hingga ratusan neuron — terlalu banyak untuk dianalisis "
            "satu per satu. Aplikasi ini otomatis meringkasnya: neuron-neuron yang bobotnya mirip "
            "dikelompokkan lagi memakai **K-Means** menjadi sejumlah kecil **segmen akhir** yang "
            "jauh lebih mudah diinterpretasi (teknik klasterisasi dua-tahap, Vesanto & Alhoniemi, 2000)."
        )
        n_final_clusters = st.slider("Jumlah segmen akhir", 2, 20, 10,
                                      help="Jumlah kelompok pelanggan akhir yang ingin dianalisis "
                                           "di menu Kluster & Dokumentasi.")

    if st.button("🚀 Latih Model SOM", type="primary", width="stretch"):
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

        with st.spinner(f"Meringkas {result.grid_x * result.grid_y} neuron menjadi "
                         f"{n_final_clusters} segmen akhir..."):
            neuron_ids, neuron_x, neuron_y = assign_clusters(result.som, X_scaled)
            neuron_cluster_map, _kmeans = cluster_neurons(result.som, n_final_clusters)
            final_clusters = map_to_final_clusters(neuron_x, neuron_y, neuron_cluster_map)

        df_with_clusters = st.session_state.preprocessed_df.copy()
        df_with_clusters["neuron_id"] = neuron_ids
        df_with_clusters["neuron_x"] = neuron_x
        df_with_clusters["neuron_y"] = neuron_y
        df_with_clusters["cluster"] = final_clusters
        if "customer_id" in df_with_clusters.columns:
            from src.cluster_insights import derive_customer_segment
            df_with_clusters["customer_segment"] = derive_customer_segment(df_with_clusters)
        else:
            df_with_clusters["customer_segment"] = df_with_clusters["cluster"]

        st.session_state.som_result = result
        st.session_state.df_with_clusters = df_with_clusters
        st.session_state.neuron_cluster_map = neuron_cluster_map
        st.session_state.n_final_clusters = int(final_clusters.max()) + 1
        st.success(f"Model SOM berhasil dilatih (grid {result.grid_x}×{result.grid_y}) dan diringkas "
                    f"menjadi {st.session_state.n_final_clusters} segmen akhir!")

    result = st.session_state.get("som_result")
    if result is not None:
        styling.section_title("📈", "Ringkasan Model Terlatih")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1, st.container(border=True):
            st.metric("Ukuran Grid", f"{result.grid_x}×{result.grid_y}")
        with m2, st.container(border=True):
            st.metric("Jumlah Neuron", result.grid_x * result.grid_y)
        with m3, st.container(border=True):
            st.metric("Segmen Akhir", st.session_state.get("n_final_clusters", "—"))
        with m4, st.container(border=True):
            st.metric("Quantization Error", f"{result.quantization_error:.4f}",
                       help="Rata-rata jarak antara data dan neuron pemenangnya — makin kecil makin baik.")
        with m5, st.container(border=True):
            st.metric("Topographic Error", f"{result.topographic_error:.4f}",
                       help="Proporsi data yang neuron pertama & kedua terdekatnya tidak bertetangga.")

        df_with_clusters = st.session_state.df_with_clusters
        cluster_counts = df_with_clusters["cluster"].value_counts().sort_index().reset_index()
        cluster_counts.columns = ["Segmen", "Jumlah Data"]
        cluster_counts["Segmen"] = "Segmen " + cluster_counts["Segmen"].astype(str)
        st.plotly_chart(
            viz.bar_by_category(cluster_counts, "Segmen", "Jumlah Data", "Distribusi Observasi Harian per Segmen"),
            width="stretch",
        )
        st.caption(
            "Distribusi di atas berdasarkan **baris data harian** (satu pelanggan bisa memiliki hari-hari "
            "yang jatuh ke segmen berbeda). Untuk segmentasi **per pelanggan** — tiap pelanggan masuk tepat "
            "satu segmen — lihat menu **Kluster & Dokumentasi**."
        )
        st.caption("Lanjutkan ke tab **Visualisasi Hasil**, atau ke menu **Kluster & Dokumentasi** "
                    "untuk profil & rekomendasi tiap segmen.")


# =============================================================================
# TAB 2 — Visualisasi Hasil
# =============================================================================
def render_visualisasi() -> None:
    if st.session_state.get("som_result") is None:
        st.warning("Latih model terlebih dahulu di tab **Pelatihan Model**.")
        return

    result = st.session_state.som_result
    som = result.som
    X_scaled = st.session_state.X_scaled
    features = st.session_state.selected_features
    df_with_clusters = st.session_state.df_with_clusters
    neuron_cluster_map = st.session_state.get("neuron_cluster_map")
    n_final_clusters = st.session_state.get("n_final_clusters", 10)

    styling.section_title("🗺️", "Peta Self-Organizing Map")
    viz_type = st.selectbox("Pilih tipe visualisasi:",
                             ["Peta Segmen Akhir", "U-Matrix", "Component Planes", "Hit Histogram",
                              "Sebaran Data Mentah pada Grid"])

    if viz_type == "Peta Segmen Akhir":
        st.pyplot(viz.plot_final_segment_map(neuron_cluster_map, n_final_clusters))
        st.caption(
            "Setiap sel adalah satu neuron SOM, diwarnai menurut segmen akhir hasil K-Means. "
            "Neuron bertetangga dengan warna sama berarti masuk segmen pelanggan yang sama."
        )
    elif viz_type == "U-Matrix":
        counts = compute_hit_map(som, X_scaled)
        st.pyplot(viz.plot_umatrix(compute_u_matrix(som), counts))
        st.caption(
            "**Cara membaca:** area gelap = neuron-neuron mirip; area terang = batas antar "
            "karakteristik berbeda; angka pada tiap sel = jumlah data."
        )
    elif viz_type == "Component Planes":
        st.pyplot(viz.plot_component_planes(som.get_weights(), features))
        st.caption("Setiap panel menunjukkan sebaran nilai satu fitur pada peta SOM. Pola yang mirip "
                    "antar panel mengindikasikan fitur-fitur tersebut berkorelasi.")
    elif viz_type == "Hit Histogram":
        hit_counts = compute_hit_map(som, X_scaled)
        st.pyplot(viz.plot_hit_histogram(hit_counts))
    else:
        neuron_counts = df_with_clusters.groupby(["neuron_x", "neuron_y"]).size().reset_index(name="count")
        st.pyplot(viz.plot_cluster_scatter_on_grid(neuron_counts, result.grid_x, result.grid_y))

    styling.section_title("🌐", "Reduksi Dimensi dengan PCA")
    n_components = 3 if X_scaled.shape[1] >= 3 else min(2, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    pca_cols = [f"PC{i + 1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    df_pca["cluster"] = df_with_clusters["cluster"].astype(str).values

    dims = st.radio("Dimensi proyeksi:", [2, 3] if n_components >= 3 else [2], horizontal=True)
    st.plotly_chart(viz.pca_scatter(df_pca, dims=dims), width="stretch")
    st.info(f"Total varians yang dijelaskan oleh {n_components} komponen PCA: "
            f"{np.sum(pca.explained_variance_ratio_):.1%}")


with tab_model:
    render_model()

with tab_viz:
    render_visualisasi()
