"""
preprocessing.py
=================
Fungsi-fungsi murni (tanpa ketergantungan Streamlit) untuk menyiapkan
data sebelum dimodelkan dengan SOM:

1. `merge_dataset`      -> menggabungkan tabel fakta + dimensi
2. `add_customer_stats` -> menambahkan fitur agregat per pelanggan
3. `handle_missing`     -> menangani missing value dengan strategi eksplisit
4. `scale_features`     -> normalisasi/standardisasi fitur numerik
5. `sample_for_training`-> mengambil sampel representatif untuk dataset besar

Dipisah dari modul SOM agar setiap langkah dapat diuji & dipakai ulang
secara independen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def merge_dataset(fact: pd.DataFrame, dim_pelanggan: pd.DataFrame,
                   dim_wilayah: pd.DataFrame | None = None) -> pd.DataFrame:
    """Menggabungkan tabel fakta dengan dimensi pelanggan (dan wilayah bila ada)."""
    df = fact.merge(dim_pelanggan, on="customer_id", how="left", validate="many_to_one")
    if dim_wilayah is not None and "region" in df.columns:
        df = df.merge(dim_wilayah, on="region", how="left", suffixes=("", "_wilayah"))

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    return df


def add_customer_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan fitur agregat historis per pelanggan (mean/std/min/max/CV)."""
    stats = df.groupby("customer_id")["consumption_kwh"].agg(
        avg_consumption="mean",
        std_consumption="std",
        min_consumption="min",
        max_consumption="max",
        total_consumption="sum",
    ).reset_index()
    stats["std_consumption"] = stats["std_consumption"].fillna(0.0)
    stats["cv_consumption"] = (stats["std_consumption"] / stats["avg_consumption"].replace(0, np.nan)).fillna(0.0)

    out = df.merge(stats, on="customer_id", how="left")
    out["z_score"] = (out["consumption_kwh"] - out["avg_consumption"]) / out["std_consumption"].replace(0, np.nan)
    out["z_score"] = out["z_score"].fillna(0.0)
    return out


@dataclass
class MissingValueReport:
    kolom: str
    jumlah_missing: int
    persentase: float
    strategi: str


def handle_missing(df: pd.DataFrame, columns: list[str],
                    strategy: str = "mean") -> tuple[pd.DataFrame, list[MissingValueReport]]:
    """
    Menangani missing value pada kolom-kolom numerik terpilih.

    strategy: "mean" | "median" | "drop_rows"
    """
    df = df.copy()
    reports: list[MissingValueReport] = []

    if strategy == "drop_rows":
        before = len(df)
        df = df.dropna(subset=columns)
        removed = before - len(df)
        reports.append(MissingValueReport("(gabungan)", removed, removed / before * 100 if before else 0, strategy))
        return df, reports

    for col in columns:
        n_missing = int(df[col].isna().sum())
        pct = (n_missing / len(df) * 100) if len(df) else 0.0
        if n_missing > 0:
            fill_value = df[col].mean() if strategy == "mean" else df[col].median()
            df[col] = df[col].fillna(fill_value)
        reports.append(MissingValueReport(col, n_missing, pct, strategy))

    return df, reports


def sample_for_training(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """
    Mengambil sampel acak-terstratifikasi (per pelanggan, proporsional)
    agar dataset besar tetap bisa dilatih dengan cepat tanpa bias
    terhadap pelanggan dengan jumlah data lebih banyak.

    Memakai `DataFrameGroupBy.sample` (bukan `.apply(lambda g: g.sample(...))`)
    karena pada pandas >= 3.0, `.apply()` tidak lagi menyertakan kolom
    pengelompokan di dalam grup yang diterima fungsi — kolom kunci
    (`customer_id`) akan hilang dari hasil akhir. `.sample()` bawaan
    groupby tidak punya masalah ini dan tetap mempertahankan semua kolom.
    """
    if "customer_id" not in df.columns:
        n = min(max_rows, len(df))
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    if len(df) <= max_rows:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    frac = max_rows / len(df)
    sampled = df.groupby("customer_id", group_keys=False).sample(frac=frac, random_state=seed)
    return sampled.reset_index(drop=True)


def scale_features(df: pd.DataFrame, features: list[str],
                    method: str = "minmax") -> tuple[np.ndarray, object]:
    """Menormalisasi/menstandardisasi fitur numerik terpilih."""
    X = df[features].to_numpy(dtype=float)
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


REQUIRED_FACT_COLUMNS = {"customer_id", "date", "consumption_kwh"}
REQUIRED_DIM_COLUMNS = {"customer_id", "customer_type", "region"}


def validate_fact_schema(df: pd.DataFrame) -> list[str]:
    """Mengembalikan daftar pesan error bila skema tabel fakta tidak sesuai."""
    errors = []
    missing_cols = REQUIRED_FACT_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Kolom wajib tidak ditemukan: {', '.join(sorted(missing_cols))}")
    return errors


def validate_dim_schema(df: pd.DataFrame) -> list[str]:
    """Mengembalikan daftar pesan error bila skema tabel dimensi tidak sesuai."""
    errors = []
    missing_cols = REQUIRED_DIM_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Kolom wajib tidak ditemukan: {', '.join(sorted(missing_cols))}")
    return errors
