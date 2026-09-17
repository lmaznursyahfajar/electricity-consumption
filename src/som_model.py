"""
som_model.py
============
Wrapper berorientasi objek di atas library `minisom` untuk melatih dan
mengevaluasi Self-Organizing Map (SOM), lengkap dengan util pemetaan
kluster kembali ke data asal.

Modul ini tidak mengimpor Streamlit sama sekali — progres pelatihan
dilaporkan lewat callback opsional (`on_progress(fraction: float)`)
sehingga logikanya bisa dites atau dipakai di luar aplikasi web.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from minisom import MiniSom
from sklearn.cluster import KMeans


@dataclass
class SOMTrainingResult:
    som: MiniSom
    grid_x: int
    grid_y: int
    quantization_error: float
    topographic_error: float
    n_features: int
    n_samples: int


def suggest_grid_size(n_samples: int) -> int:
    """Heuristik ukuran grid SOM: 5*sqrt(n) neuron (aturan umum Vesanto)."""
    return max(3, int(np.sqrt(5 * np.sqrt(max(n_samples, 1)))))


def train_som(X_scaled: np.ndarray,
              grid_x: Optional[int] = None,
              grid_y: Optional[int] = None,
              sigma: float = 1.0,
              learning_rate: float = 0.5,
              iterations: int = 2000,
              random_seed: int = 42,
              on_progress: Optional[Callable[[float], None]] = None) -> SOMTrainingResult:
    """Melatih SOM di atas data yang sudah dinormalisasi."""
    n_samples, n_features = X_scaled.shape

    if grid_x is None or grid_y is None:
        dim = suggest_grid_size(n_samples)
        grid_x, grid_y = dim, dim

    som = MiniSom(
        x=grid_x, y=grid_y,
        input_len=n_features,
        sigma=min(sigma, max(grid_x, grid_y) / 2),
        learning_rate=learning_rate,
        random_seed=random_seed,
    )
    som.random_weights_init(X_scaled)

    report_every = max(1, iterations // 50)
    rng = np.random.default_rng(random_seed)
    for i in range(iterations):
        idx = rng.integers(0, n_samples)
        som.update(X_scaled[idx], som.winner(X_scaled[idx]), i, iterations)
        if on_progress is not None and (i % report_every == 0 or i == iterations - 1):
            on_progress((i + 1) / iterations)

    q_error = float(som.quantization_error(X_scaled))
    try:
        t_error = float(som.topographic_error(X_scaled))
    except Exception:
        t_error = float("nan")

    return SOMTrainingResult(
        som=som, grid_x=grid_x, grid_y=grid_y,
        quantization_error=q_error, topographic_error=t_error,
        n_features=n_features, n_samples=n_samples,
    )


def assign_clusters(som: MiniSom, X_scaled: np.ndarray) -> np.ndarray:
    """Menghitung ID neuron pemenang (BMU) untuk setiap baris data.

    ID ini adalah kluster mentah tingkat-neuron — jumlahnya sama dengan
    jumlah neuron pada grid (bisa puluhan hingga ratusan). Untuk analisis
    bisnis, jumlah ini biasanya terlalu banyak; gunakan `cluster_neurons`
    + `map_to_final_clusters` di bawah untuk meringkasnya menjadi segmen
    akhir yang jauh lebih sedikit dan mudah diinterpretasi.
    """
    cluster_ids = np.empty(len(X_scaled), dtype=int)
    neuron_x = np.empty(len(X_scaled), dtype=int)
    neuron_y = np.empty(len(X_scaled), dtype=int)
    for i, row in enumerate(X_scaled):
        wx, wy = som.winner(row)
        neuron_x[i] = wx
        neuron_y[i] = wy
        cluster_ids[i] = wx * som.get_weights().shape[1] + wy
    return cluster_ids, neuron_x, neuron_y


def cluster_neurons(som: MiniSom, n_final_clusters: int, random_seed: int = 42) -> tuple[np.ndarray, KMeans]:
    """
    Meringkas neuron-neuron SOM menjadi sejumlah kecil segmen akhir.

    Ini adalah teknik klasterisasi dua-tahap yang umum dipakai bersama SOM
    (Vesanto & Alhoniemi, 2000): SOM dipakai untuk mempelajari topologi &
    mereduksi dimensi data, lalu K-Means dijalankan pada VEKTOR BOBOT tiap
    neuron (bukan pada data mentah) untuk mengelompokkan neuron-neuron yang
    berdekatan/mirip menjadi `n_final_clusters` segmen akhir. Hasilnya jauh
    lebih ringkas & mudah diinterpretasi dibanding memakai tiap neuron
    sebagai kluster tersendiri, namun tetap memanfaatkan topologi yang
    dipelajari SOM (neuron yang berdekatan pada grid cenderung masuk
    segmen akhir yang sama).

    Mengembalikan:
    - neuron_cluster_map: array 2D (grid_x, grid_y) berisi ID segmen akhir
      (0..n_final_clusters-1) untuk tiap neuron.
    - kmeans: objek KMeans terlatih (untuk keperluan lanjutan bila perlu).
    """
    weights = som.get_weights()
    grid_x, grid_y, n_features = weights.shape
    flat_weights = weights.reshape(-1, n_features)

    n_final_clusters = max(1, min(n_final_clusters, flat_weights.shape[0]))
    kmeans = KMeans(n_clusters=n_final_clusters, random_state=random_seed, n_init=10)
    neuron_labels = kmeans.fit_predict(flat_weights)

    neuron_cluster_map = neuron_labels.reshape(grid_x, grid_y)
    return neuron_cluster_map, kmeans


def map_to_final_clusters(neuron_x: np.ndarray, neuron_y: np.ndarray,
                           neuron_cluster_map: np.ndarray) -> np.ndarray:
    """Memetakan ID neuron (BMU) tiap baris data ke ID segmen akhir hasil `cluster_neurons`."""
    return neuron_cluster_map[neuron_x, neuron_y]


def u_matrix(som: MiniSom) -> np.ndarray:
    """Mengembalikan U-Matrix (jarak rata-rata antar neuron tetangga)."""
    return som.distance_map()


def hit_map(som: MiniSom, X_scaled: np.ndarray) -> np.ndarray:
    """Menghitung jumlah data yang jatuh pada tiap neuron."""
    counts = np.zeros((som.get_weights().shape[0], som.get_weights().shape[1]))
    for row in X_scaled:
        wx, wy = som.winner(row)
        counts[wx, wy] += 1
    return counts
