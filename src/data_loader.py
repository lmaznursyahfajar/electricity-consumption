"""
data_loader.py
==============
Memuat dataset bawaan (fact + dimension tables hasil `data_generator.py`)
maupun dataset yang diunggah pengguna sendiri (CSV/Excel), lengkap dengan
validasi skema dasar sebelum data dipakai di halaman-halaman lain.
"""

from __future__ import annotations

import pandas as pd

from . import config
from .data_generator import generate_full_dataset
from .preprocessing import validate_dim_schema, validate_fact_schema


def load_bundled_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Memuat dataset bawaan dari disk; membangkitkan ulang bila file belum ada."""
    if config.FACT_CSV_PATH.exists() and config.DIM_PELANGGAN_PATH.exists() and config.DIM_WILAYAH_PATH.exists():
        fact = pd.read_csv(config.FACT_CSV_PATH, parse_dates=["date"])
        dim_pelanggan = pd.read_csv(config.DIM_PELANGGAN_PATH)
        dim_wilayah = pd.read_csv(config.DIM_WILAYAH_PATH)
        return fact, dim_pelanggan, dim_wilayah

    fact, dim_pelanggan, dim_wilayah = generate_full_dataset()
    return fact, dim_pelanggan, dim_wilayah


def load_uploaded_fact(uploaded_file) -> tuple[pd.DataFrame | None, list[str]]:
    """Membaca file fakta konsumsi yang diunggah pengguna (CSV/XLSX) & memvalidasinya."""
    try:
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        return None, [f"Gagal membaca file: {exc}"]

    errors = validate_fact_schema(df)
    if errors:
        return None, errors

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            errors.append("Beberapa nilai pada kolom 'date' tidak bisa dibaca sebagai tanggal.")

    return df, errors


def load_uploaded_dim_pelanggan(uploaded_file) -> tuple[pd.DataFrame | None, list[str]]:
    """Membaca file dimensi pelanggan yang diunggah pengguna & memvalidasinya."""
    try:
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        return None, [f"Gagal membaca file: {exc}"]

    errors = validate_dim_schema(df)
    return (df, errors) if not errors else (None, errors)
