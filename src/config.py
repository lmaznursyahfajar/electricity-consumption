"""
config.py
=========
Konfigurasi terpusat untuk aplikasi Analisis Pola Konsumsi Listrik.

Menyimpan seluruh konstanta (parameter dataset, daftar wilayah, kelas
tarif, hari libur, dsb) di satu tempat sehingga tidak ada "magic number"
yang tersebar di berbagai modul. Jika suatu saat parameter perlu diubah
(mis. menambah wilayah baru), cukup ubah di file ini.
"""

from __future__ import annotations
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Path proyek
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

FACT_CSV_PATH = DATA_DIR / "fact_konsumsi_harian.csv"
DIM_PELANGGAN_PATH = DATA_DIR / "dim_pelanggan.csv"
DIM_WILAYAH_PATH = DATA_DIR / "dim_wilayah.csv"
EXCEL_WORKBOOK_PATH = DATA_DIR / "dataset_konsumsi_listrik.xlsx"

# ---------------------------------------------------------------------------
# Parameter pembangkitan dataset
# ---------------------------------------------------------------------------
N_CUSTOMERS = 600
START_DATE = date(2023, 1, 1)
END_DATE = date(2024, 6, 30)          # 18 bulan data harian
RANDOM_SEED = 42
MISSING_VALUE_RATE = 0.012            # 1.2% nilai konsumsi sengaja dikosongkan
ANOMALY_RATE = 0.02                   # 2% baris dibuat anomali

# Kelas tarif listrik disederhanakan dari golongan tarif PLN.
# Angka Rp/kWh & rentang konsumsi bersifat ILUSTRATIF untuk simulasi,
# BUKAN tarif resmi PLN yang berlaku saat ini.
TARIFF_CLASSES = {
    "R1/900VA":  {"segmen": "Rumah Tangga", "daya_va": 900,   "kwh_range": (40, 130),   "rp_per_kwh": 1352, "bobot": 0.22},
    "R1/1300VA": {"segmen": "Rumah Tangga", "daya_va": 1300,  "kwh_range": (70, 220),   "rp_per_kwh": 1444, "bobot": 0.20},
    "R1/2200VA": {"segmen": "Rumah Tangga", "daya_va": 2200,  "kwh_range": (120, 350),  "rp_per_kwh": 1444, "bobot": 0.12},
    "R2/5500VA": {"segmen": "Rumah Tangga", "daya_va": 5500,  "kwh_range": (250, 700),  "rp_per_kwh": 1699, "bobot": 0.08},
    "B1":        {"segmen": "Bisnis Kecil", "daya_va": 5500,  "kwh_range": (150, 600),  "rp_per_kwh": 1444, "bobot": 0.09},
    "B2":        {"segmen": "Bisnis Menengah", "daya_va": 41500, "kwh_range": (600, 2500), "rp_per_kwh": 1699, "bobot": 0.07},
    "I1":        {"segmen": "Industri Kecil", "daya_va": 13900, "kwh_range": (300, 1200), "rp_per_kwh": 1444, "bobot": 0.08},
    "I2":        {"segmen": "Industri Menengah", "daya_va": 200000, "kwh_range": (1200, 6000), "rp_per_kwh": 1115, "bobot": 0.06},
    "I3/I4":     {"segmen": "Industri Besar", "daya_va": 1000000, "kwh_range": (4000, 20000), "rp_per_kwh": 997, "bobot": 0.03},
    "S2":        {"segmen": "Sosial", "daya_va": 3500, "kwh_range": (80, 400), "rp_per_kwh": 1352, "bobot": 0.03},
    "P1":        {"segmen": "Pemerintah", "daya_va": 13900, "kwh_range": (300, 1500), "rp_per_kwh": 1699, "bobot": 0.02},
}

# Pola beban harian dominan per segmen — dipakai untuk menentukan jam
# beban puncak & bentuk kurva konsumsi (bukan data per-jam, tapi
# memengaruhi rasio beban puncak terhadap rata-rata harian).
SEGMENT_LOAD_PATTERN = {
    "Rumah Tangga": "Malam",
    "Bisnis Kecil": "Siang",
    "Bisnis Menengah": "Siang",
    "Industri Kecil": "Siang",
    "Industri Menengah": "24 Jam",
    "Industri Besar": "24 Jam",
    "Sosial": "Siang",
    "Pemerintah": "Siang",
}

# Wilayah beserta karakteristik iklim (memengaruhi suhu & pemakaian AC)
REGIONS = {
    "Jakarta":   {"provinsi": "DKI Jakarta",       "suhu_baseline_c": 29.5, "amplitudo_musiman": 1.5},
    "Bandung":   {"provinsi": "Jawa Barat",        "suhu_baseline_c": 23.5, "amplitudo_musiman": 1.2},
    "Surabaya":  {"provinsi": "Jawa Timur",        "suhu_baseline_c": 30.5, "amplitudo_musiman": 1.4},
    "Medan":     {"provinsi": "Sumatera Utara",    "suhu_baseline_c": 28.5, "amplitudo_musiman": 1.1},
    "Makassar":  {"provinsi": "Sulawesi Selatan",  "suhu_baseline_c": 28.0, "amplitudo_musiman": 1.0},
    "Semarang":  {"provinsi": "Jawa Tengah",       "suhu_baseline_c": 29.0, "amplitudo_musiman": 1.3},
    "Palembang": {"provinsi": "Sumatera Selatan",  "suhu_baseline_c": 28.0, "amplitudo_musiman": 1.1},
    "Denpasar":  {"provinsi": "Bali",              "suhu_baseline_c": 27.5, "amplitudo_musiman": 1.0},
}

# Hari libur nasional Indonesia yang disimulasikan (2023–2024).
# Daftar disederhanakan untuk keperluan simulasi dataset, bukan rujukan
# kalender resmi.
PUBLIC_HOLIDAYS = {
    "2023-01-01", "2023-01-22", "2023-03-22", "2023-04-07", "2023-04-21",
    "2023-04-22", "2023-05-01", "2023-05-18", "2023-06-01", "2023-06-04",
    "2023-06-29", "2023-08-17", "2023-09-28", "2023-12-25",
    "2024-01-01", "2024-02-08", "2024-02-10", "2024-03-11", "2024-03-29",
    "2024-04-10", "2024-04-11", "2024-05-01", "2024-05-09", "2024-05-23",
    "2024-06-01", "2024-06-17",
}

# ---------------------------------------------------------------------------
# Fitur default untuk pemodelan SOM
# ---------------------------------------------------------------------------
DEFAULT_SOM_FEATURES = [
    "consumption_kwh", "avg_consumption", "std_consumption",
    "load_factor", "month", "is_weekend",
]

MAX_ROWS_FOR_TRAINING_DEFAULT = 15000  # batas sampel default untuk melatih SOM

# ---------------------------------------------------------------------------
# Tampilan Streamlit
# ---------------------------------------------------------------------------
APP_TITLE = "SIGAP Listrik — Sistem Analisis Pola Konsumsi Listrik"
APP_ICON = "⚡"
PRIMARY_COLOR = "#0F62FE"
ACCENT_COLOR = "#F7A61B"

APP_VERSION = "2.2.0"
