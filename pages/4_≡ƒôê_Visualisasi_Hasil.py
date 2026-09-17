"""Halaman: Visualisasi Hasil — U-Matrix, component planes, hit histogram, PCA."""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from src import styling, visualizations as viz
from src.som_model import hit_map as compute_hit_map
from src.som_model import u_matrix as compute_u_matrix

st.set_page_config(page_title="Visualisasi Hasil", page_icon="📈", layout="wide")
styling.inject_global_css()
styling.page_header("📈 Visualisasi Hasil SOM", "Interpretasi visual dari peta Self-Organizing Map yang terlatih.")

if st.session_state.get("som_result") is None:
    st.warning("Latih model terlebih dahulu di halaman **🧠 Model SOM**.")
    st.stop()

result = st.session_state.som_result
som = result.som
X_scaled = st.session_state.X_scaled
features = st.session_state.selected_features
df_with_clusters = st.session_state.df_with_clusters

viz_type = st.selectbox("Pilih tipe visualisasi:",
                         ["U-Matrix", "Component Planes", "Hit Histogram", "Sebaran Kluster pada Grid"])

if viz_type == "U-Matrix":
    st.subheader("U-Matrix (Unified Distance Matrix)")
    counts = compute_hit_map(som, X_scaled)
    fig = viz.plot_umatrix(compute_u_matrix(som), counts)
    st.pyplot(fig)
    st.markdown(
        """
**Cara membaca:**
- **Area gelap** → neuron-neuron yang sangat mirip (kluster padat).
- **Area terang** → batas antar kluster yang berbeda karakteristik.
- **Angka pada tiap sel** → jumlah data yang jatuh pada neuron tersebut.
        """
    )

elif viz_type == "Component Planes":
    st.subheader("Component Planes")
    fig = viz.plot_component_planes(som.get_weights(), features)
    st.pyplot(fig)
    st.markdown(
        "Setiap panel menunjukkan sebaran nilai satu fitur pada peta SOM. Pola yang mirip antar "
        "panel mengindikasikan fitur-fitur tersebut berkorelasi."
    )

elif viz_type == "Hit Histogram":
    st.subheader("Hit Histogram")
    hit_counts = compute_hit_map(som, X_scaled)
    st.pyplot(viz.plot_hit_histogram(hit_counts))

else:
    st.subheader("Sebaran Kluster pada Grid SOM")
    neuron_counts = df_with_clusters.groupby(["neuron_x", "neuron_y"]).size().reset_index(name="count")
    st.pyplot(viz.plot_cluster_scatter_on_grid(neuron_counts, result.grid_x, result.grid_y))

st.divider()
st.subheader("Reduksi Dimensi dengan PCA")
n_components = 3 if X_scaled.shape[1] >= 3 else min(2, X_scaled.shape[1])
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

pca_cols = [f"PC{i + 1}" for i in range(n_components)]
df_pca = pd.DataFrame(X_pca, columns=pca_cols)
df_pca["cluster"] = df_with_clusters["cluster"].astype(str).values

dims = st.radio("Dimensi proyeksi:", [2, 3] if n_components >= 3 else [2], horizontal=True)
st.plotly_chart(viz.pca_scatter(df_pca, dims=dims), width='stretch')
st.info(f"Total varians yang dijelaskan oleh {n_components} komponen PCA: "
        f"{np.sum(pca.explained_variance_ratio_):.1%}")

styling.sidebar_footer()
