"""
data_generator.py
==================
Pembangkit dataset sintetis konsumsi listrik dengan skema bergaya data
warehouse: satu tabel fakta (transaksional, harian) dan dua tabel dimensi
(pelanggan & wilayah). Skema ini dipilih dibanding tabel datar tunggal
(seperti pada versi awal aplikasi) karena:

1. Tidak ada duplikasi data pelanggan/wilayah di setiap baris → ukuran
   file jauh lebih kecil untuk jumlah baris yang sama.
2. Mencerminkan praktik pemodelan data yang lazim di industri (fact &
   dimension table) sehingga lebih mudah dikembangkan (mis. menambah
   atribut pelanggan tanpa mengubah tabel fakta).
3. Mempermudah validasi & penggantian salah satu tabel dengan data nyata
   pengguna tanpa mengubah struktur lainnya.

Seluruh perhitungan memakai operasi vektor NumPy/Pandas (bukan loop
Python per-baris) agar pembangkitan ratusan ribu baris tetap cepat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _rng(seed: int = config.RANDOM_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_dim_pelanggan(n_customers: int = config.N_CUSTOMERS,
                            seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Membuat tabel dimensi pelanggan (satu baris per pelanggan)."""
    rng = _rng(seed)

    tariff_names = list(config.TARIFF_CLASSES.keys())
    weights = np.array([config.TARIFF_CLASSES[t]["bobot"] for t in tariff_names])
    weights = weights / weights.sum()

    tariff_choice = rng.choice(tariff_names, size=n_customers, p=weights)
    region_choice = rng.choice(list(config.REGIONS.keys()), size=n_customers)

    daya_va = np.array([config.TARIFF_CLASSES[t]["daya_va"] for t in tariff_choice])
    segmen = np.array([config.TARIFF_CLASSES[t]["segmen"] for t in tariff_choice])
    rp_per_kwh = np.array([config.TARIFF_CLASSES[t]["rp_per_kwh"] for t in tariff_choice])

    # Baseline konsumsi harian per pelanggan (kWh/hari), diambil acak
    # dalam rentang khas kelas tarifnya.
    low = np.array([config.TARIFF_CLASSES[t]["kwh_range"][0] for t in tariff_choice])
    high = np.array([config.TARIFF_CLASSES[t]["kwh_range"][1] for t in tariff_choice])
    base_consumption = rng.uniform(low, high)

    install_dates = pd.to_datetime("2015-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 9, size=n_customers), unit="D"
    )

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i + 1:05d}" for i in range(n_customers)],
        "customer_type": segmen,
        "tariff_class": tariff_choice,
        "daya_terpasang_va": daya_va,
        "tarif_rp_per_kwh": rp_per_kwh,
        "region": region_choice,
        "base_consumption_kwh": base_consumption.round(2),
        "tanggal_pasang": install_dates.date,
    })
    return df


def generate_dim_wilayah() -> pd.DataFrame:
    """Membuat tabel dimensi wilayah."""
    rows = []
    for region, attrs in config.REGIONS.items():
        rows.append({"region": region, **attrs})
    return pd.DataFrame(rows)


def generate_fact_konsumsi_harian(dim_pelanggan: pd.DataFrame,
                                   start_date=config.START_DATE,
                                   end_date=config.END_DATE,
                                   seed: int = config.RANDOM_SEED,
                                   missing_rate: float = config.MISSING_VALUE_RATE,
                                   anomaly_rate: float = config.ANOMALY_RATE) -> pd.DataFrame:
    """
    Membuat tabel fakta konsumsi harian untuk seluruh pelanggan x tanggal,
    sepenuhnya melalui operasi vektor (tanpa loop Python per baris).
    """
    rng = _rng(seed)

    dates = pd.date_range(start_date, end_date, freq="D")
    n_customers = len(dim_pelanggan)
    n_days = len(dates)

    # Cross-join pelanggan x tanggal via np.repeat / np.tile (vektor).
    customer_id = np.repeat(dim_pelanggan["customer_id"].values, n_days)
    base_consumption = np.repeat(dim_pelanggan["base_consumption_kwh"].values, n_days)
    segmen = np.repeat(dim_pelanggan["customer_type"].values, n_days)
    region = np.repeat(dim_pelanggan["region"].values, n_days)

    date_vals = np.tile(dates.values, n_customers)
    day_index = np.tile(np.arange(n_days), n_customers)
    weekday = np.tile(dates.weekday.values, n_customers)
    month = np.tile(dates.month.values, n_customers)
    date_str = np.tile(dates.strftime("%Y-%m-%d").values, n_customers)

    n_rows = n_customers * n_days

    # --- Faktor musiman: puncak berbeda per wilayah (suhu & AC) ---
    region_lookup = config.REGIONS
    amplitudo = np.repeat(
        dim_pelanggan["region"].map(lambda r: region_lookup[r]["amplitudo_musiman"]).values, n_days
    )
    seasonal_factor = 1 + 0.15 * amplitudo * np.sin(2 * np.pi * (day_index - 60) / 365.25)

    # --- Faktor akhir pekan vs weekday (tergantung segmen) ---
    is_weekend = (weekday >= 5).astype(int)
    weekend_boost_household = np.isin(segmen, ["Rumah Tangga"]) & (is_weekend == 1)
    weekend_drop_business = np.isin(segmen, ["Bisnis Kecil", "Bisnis Menengah", "Industri Kecil",
                                              "Pemerintah"]) & (is_weekend == 1)
    weekday_factor = np.ones(n_rows)
    weekday_factor[weekend_boost_household] *= 1.12
    weekday_factor[weekend_drop_business] *= 0.55

    # --- Faktor hari libur nasional ---
    is_holiday = np.isin(date_str, list(config.PUBLIC_HOLIDAYS)).astype(int)
    holiday_boost_household = (is_holiday == 1) & np.isin(segmen, ["Rumah Tangga", "Sosial"])
    holiday_drop_business = (is_holiday == 1) & np.isin(
        segmen, ["Bisnis Kecil", "Bisnis Menengah", "Industri Kecil", "Industri Menengah",
                 "Industri Besar", "Pemerintah"])
    holiday_factor = np.ones(n_rows)
    holiday_factor[holiday_boost_household] *= 1.20
    holiday_factor[holiday_drop_business] *= 0.65

    # --- Faktor acak (noise) ---
    random_factor = rng.normal(1.0, 0.08, size=n_rows)

    consumption = base_consumption * seasonal_factor * weekday_factor * holiday_factor * random_factor
    consumption = np.clip(consumption, a_min=0.5, a_max=None)

    # --- Suhu sintetis (berkorelasi dengan faktor musiman wilayah) ---
    suhu_baseline = np.repeat(
        dim_pelanggan["region"].map(lambda r: region_lookup[r]["suhu_baseline_c"]).values, n_days
    )
    temperature_c = suhu_baseline + 3.0 * amplitudo * np.sin(2 * np.pi * (day_index - 60) / 365.25) \
        + rng.normal(0, 0.8, size=n_rows)

    # --- Anomali (mis. gangguan meter / lonjakan tak wajar) ---
    anomaly_mask = rng.random(n_rows) < anomaly_rate
    anomaly_multiplier = rng.uniform(1.6, 3.2, size=n_rows)
    consumption = np.where(anomaly_mask, consumption * anomaly_multiplier, consumption)

    # --- Beban puncak harian (proksi, sebagai rasio dari konsumsi harian) ---
    pattern = np.repeat(
        dim_pelanggan["customer_type"].map(lambda s: config.SEGMENT_LOAD_PATTERN.get(s, "Siang")).values,
        n_days,
    )
    peak_ratio = np.select(
        [pattern == "24 Jam", pattern == "Malam", pattern == "Siang"],
        [0.06, 0.12, 0.09],
        default=0.09,
    )
    peak_load_kwh = consumption * peak_ratio * rng.uniform(0.9, 1.1, size=n_rows)
    load_factor = np.clip(consumption / (peak_load_kwh * 24 + 1e-6), 0, 1)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "date": pd.to_datetime(date_str),
        "consumption_kwh": consumption.round(3),
        "temperature_c": temperature_c.round(2),
        "peak_load_kwh": peak_load_kwh.round(3),
        "load_factor": load_factor.round(4),
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "is_anomaly": anomaly_mask.astype(int),
    })

    # --- Suntikkan missing value secara acak pada kolom konsumsi ---
    missing_mask = rng.random(n_rows) < missing_rate
    df.loc[missing_mask, "consumption_kwh"] = np.nan

    return df


def generate_full_dataset(n_customers: int = config.N_CUSTOMERS,
                           start_date=config.START_DATE,
                           end_date=config.END_DATE,
                           seed: int = config.RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Membuat ketiga tabel (fact, dim_pelanggan, dim_wilayah) sekaligus."""
    dim_pelanggan = generate_dim_pelanggan(n_customers, seed=seed)
    dim_wilayah = generate_dim_wilayah()
    fact = generate_fact_konsumsi_harian(dim_pelanggan, start_date, end_date, seed=seed)
    return fact, dim_pelanggan, dim_wilayah


if __name__ == "__main__":
    # Jalankan sebagai skrip mandiri untuk membangkitkan & menyimpan dataset:
    #     python -m src.data_generator
    import time

    t0 = time.time()
    fact, dim_pelanggan, dim_wilayah = generate_full_dataset()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    fact.to_csv(config.FACT_CSV_PATH, index=False)
    dim_pelanggan.to_csv(config.DIM_PELANGGAN_PATH, index=False)
    dim_wilayah.to_csv(config.DIM_WILAYAH_PATH, index=False)

    print(f"Selesai dalam {time.time() - t0:.2f} detik")
    print(f"fact_konsumsi_harian : {len(fact):,} baris -> {config.FACT_CSV_PATH}")
    print(f"dim_pelanggan        : {len(dim_pelanggan):,} baris -> {config.DIM_PELANGGAN_PATH}")
    print(f"dim_wilayah          : {len(dim_wilayah):,} baris -> {config.DIM_WILAYAH_PATH}")
