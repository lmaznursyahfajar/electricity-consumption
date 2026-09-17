"""
visualizations.py
==================
Kumpulan fungsi pembuat grafik (Matplotlib untuk peta SOM, Plotly untuk
grafik interaktif) yang dipakai berulang kali di berbagai halaman.
Memusatkan kode plotting di sini menghindari duplikasi yang terjadi pada
versi awal aplikasi (setiap halaman menulis ulang kode Plotly serupa).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PLOTLY_TEMPLATE = "plotly_white"


# ---------------------------------------------------------------------------
# Matplotlib — peta SOM
# ---------------------------------------------------------------------------

def plot_umatrix(umatrix: np.ndarray, counts: np.ndarray | None = None):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(umatrix.T, cmap="viridis", origin="lower")
    plt.colorbar(im, ax=ax, label="Jarak rata-rata antar neuron")

    if counts is not None:
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                if counts[i, j] > 0:
                    ax.text(i, j, f"{int(counts[i, j])}", ha="center", va="center",
                            color="white" if umatrix[i, j] > 0.5 else "black", fontsize=7)

    ax.set_title("U-Matrix — Batas & Kepadatan Kluster")
    ax.set_xlabel("Neuron X")
    ax.set_ylabel("Neuron Y")
    fig.tight_layout()
    return fig


def plot_component_planes(som_weights: np.ndarray, feature_names: list[str]):
    n_features = len(feature_names)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, feature in enumerate(feature_names):
        ax = axes[i]
        plane = som_weights[:, :, i]
        im = ax.imshow(plane.T, cmap="coolwarm", origin="lower")
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("Neuron X")
        ax.set_ylabel("Neuron Y")
        plt.colorbar(im, ax=ax, fraction=0.046)

    for i in range(n_features, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    return fig


def plot_hit_histogram(hit_counts: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(hit_counts.T, cmap="YlOrRd", origin="lower")
    max_val = hit_counts.max() if hit_counts.max() > 0 else 1

    for i in range(hit_counts.shape[0]):
        for j in range(hit_counts.shape[1]):
            if hit_counts[i, j] > 0:
                ax.text(i, j, f"{int(hit_counts[i, j])}", ha="center", va="center",
                         color="black" if hit_counts[i, j] < max_val / 2 else "white", fontsize=7)

    ax.set_title("Hit Histogram — Jumlah Data per Neuron")
    ax.set_xlabel("Neuron X")
    ax.set_ylabel("Neuron Y")
    plt.colorbar(im, ax=ax, label="Jumlah data")
    fig.tight_layout()
    return fig


def plot_cluster_scatter_on_grid(neuron_counts: pd.DataFrame, grid_x: int, grid_y: int):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    scatter = ax.scatter(neuron_counts["neuron_x"], neuron_counts["neuron_y"],
                          s=neuron_counts["count"] * 8, c=neuron_counts["count"],
                          cmap="viridis", alpha=0.75, edgecolors="white", linewidths=0.5)
    for _, row in neuron_counts.iterrows():
        ax.text(row["neuron_x"], row["neuron_y"], str(int(row["count"])),
                ha="center", va="center", fontsize=7, color="white")
    ax.set_xlim(-0.5, grid_x - 0.5)
    ax.set_ylim(-0.5, grid_y - 0.5)
    ax.set_xlabel("Neuron X")
    ax.set_ylabel("Neuron Y")
    ax.set_title("Sebaran Data pada Peta SOM")
    ax.grid(alpha=0.25)
    plt.colorbar(scatter, ax=ax, label="Jumlah data")
    fig.tight_layout()
    return fig


def plot_final_segment_map(neuron_cluster_map: np.ndarray, n_final_clusters: int):
    """Peta grid SOM yang diwarnai menurut segmen akhir (hasil K-Means pada neuron)."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = plt.get_cmap("tab10" if n_final_clusters <= 10 else "tab20")
    im = ax.imshow(neuron_cluster_map.T, cmap=cmap, origin="lower",
                    vmin=-0.5, vmax=max(n_final_clusters - 0.5, 0.5))
    for i in range(neuron_cluster_map.shape[0]):
        for j in range(neuron_cluster_map.shape[1]):
            ax.text(i, j, str(int(neuron_cluster_map[i, j])), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
    ax.set_title(f"Peta Segmen Akhir ({n_final_clusters} segmen)")
    ax.set_xlabel("Neuron X")
    ax.set_ylabel("Neuron Y")
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_final_clusters), label="ID Segmen")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plotly — grafik interaktif
# ---------------------------------------------------------------------------

def kpi_daily_trend(daily: pd.DataFrame):
    fig = px.line(daily, x="date", y="consumption_kwh", template=PLOTLY_TEMPLATE,
                  labels={"consumption_kwh": "Konsumsi (kWh)", "date": "Tanggal"},
                  title="Tren Total Konsumsi Harian")
    if len(daily) >= 7:
        moving_avg = daily["consumption_kwh"].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=daily["date"], y=moving_avg, mode="lines",
                                  name="Rata-rata bergerak (7 hari)",
                                  line=dict(color="#F7A61B", width=2.5)))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def bar_by_category(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None):
    fig = px.bar(df, x=x, y=y, color=color or x, template=PLOTLY_TEMPLATE, title=title)
    fig.update_layout(showlegend=False if color is None else True)
    return fig


def pie_distribution(df: pd.DataFrame, names: str, values: str | None, title: str):
    if values:
        fig = px.pie(df, names=names, values=values, template=PLOTLY_TEMPLATE, title=title, hole=0.35)
    else:
        fig = px.pie(df, names=names, template=PLOTLY_TEMPLATE, title=title, hole=0.35)
    return fig


def correlation_heatmap(df_numeric: pd.DataFrame, title: str = "Matriks Korelasi Antar Fitur"):
    fig = px.imshow(df_numeric.corr(), template=PLOTLY_TEMPLATE, title=title,
                     color_continuous_scale="RdBu", zmin=-1, zmax=1, text_auto=".2f")
    return fig


def pca_scatter(df_pca: pd.DataFrame, dims: int = 2, color_col: str | None = "cluster"):
    color = color_col if color_col in df_pca.columns else None
    if dims == 2:
        fig = px.scatter(df_pca, x="PC1", y="PC2", color=color, template=PLOTLY_TEMPLATE,
                          title="Proyeksi PCA 2D")
    else:
        fig = px.scatter_3d(df_pca, x="PC1", y="PC2", z="PC3", color=color, template=PLOTLY_TEMPLATE,
                             title="Proyeksi PCA 3D")
    return fig


def histogram_before_after(before: pd.Series, after: np.ndarray, feature_name: str):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Sebelum Normalisasi", "Setelah Normalisasi"))
    fig.add_trace(go.Histogram(x=before, name="Sebelum", marker_color="#0F62FE"), row=1, col=1)
    fig.add_trace(go.Histogram(x=after, name="Sesudah", marker_color="#F7A61B"), row=1, col=2)
    fig.update_layout(height=380, showlegend=False, template=PLOTLY_TEMPLATE,
                       title=f"Distribusi Fitur: {feature_name}")
    return fig
