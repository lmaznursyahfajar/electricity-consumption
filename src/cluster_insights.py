"""
cluster_insights.py
====================
Menghasilkan profil, nama persona, dan rekomendasi untuk tiap segmen
(kluster akhir hasil SOM + K-Means) — seluruhnya dihitung langsung dari
data segmen tersebut dibandingkan populasi keseluruhan, bukan teks
generik. Modul ini bebas dari Streamlit sehingga bisa diuji mandiri.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ClusterProfile:
    cluster_id: int
    n_customers: int
    pct_customers: float
    n_rows: int
    avg_consumption: float
    overall_avg_consumption: float
    consumption_vs_overall_pct: float
    total_consumption: float
    pct_of_total_consumption: float
    consumption_rank: int
    n_total_clusters: int
    dominant_type: str | None
    dominant_type_pct: float | None
    dominant_region: str | None
    dominant_region_pct: float | None
    avg_load_factor: float | None
    load_factor_vs_overall_pct: float | None
    anomaly_rate_pct: float | None
    overall_anomaly_rate_pct: float | None
    weekend_vs_weekday_pct: float | None
    persona: str
    recommendations: list[str] = field(default_factory=list)


def _pct_diff(value: float, baseline: float) -> float:
    if baseline in (0, None) or pd.isna(baseline) or pd.isna(value):
        return 0.0
    return (value - baseline) / baseline * 100


def _dominant_with_share(series: pd.Series) -> tuple[str | None, float | None]:
    if series.empty or series.mode().empty:
        return None, None
    top = series.mode().iat[0]
    share = (series == top).mean() * 100
    return top, share


def _consumption_level_label(pct_diff: float) -> str:
    if pct_diff >= 80:
        return "Konsumsi Sangat Tinggi"
    if pct_diff >= 25:
        return "Konsumsi Tinggi"
    if pct_diff >= -25:
        return "Konsumsi Sedang"
    if pct_diff >= -55:
        return "Konsumsi Rendah"
    return "Konsumsi Sangat Rendah"


def _build_persona(dominant_type: str | None, pct_diff: float) -> str:
    level = _consumption_level_label(pct_diff)
    if dominant_type:
        return f"{dominant_type} — {level}"
    return level


def _build_recommendations(p: dict) -> list[str]:
    """Menghasilkan daftar rekomendasi bertekstur angka nyata dari statistik segmen `p`."""
    recs: list[str] = []

    # 1) Konsumsi tinggi -> audit energi dengan angka potensi penghematan
    if p["consumption_vs_overall_pct"] > 30:
        potensi_hemat = p["avg_consumption"] * 0.10
        recs.append(
            f"Rata-rata konsumsi segmen ini **{p['avg_consumption']:,.0f} kWh/hari**, "
            f"**{p['consumption_vs_overall_pct']:+.0f}%** dibanding rata-rata seluruh pelanggan "
            f"({p['overall_avg_consumption']:,.0f} kWh/hari). Prioritaskan audit energi untuk "
            f"**{p['n_customers']:,} pelanggan** di segmen ini — efisiensi 10% saja berpotensi "
            f"menghemat sekitar **{potensi_hemat:,.0f} kWh/hari per pelanggan**."
        )

    # 2) Kontributor besar terhadap total konsumsi -> target demand-response
    if p["consumption_rank"] <= max(1, round(p["n_total_clusters"] * 0.3)):
        recs.append(
            f"Segmen ini menyumbang **{p['pct_of_total_consumption']:.1f}%** dari total konsumsi "
            f"seluruh pelanggan (peringkat **#{p['consumption_rank']} dari {p['n_total_clusters']}** "
            f"segmen berdasarkan kontribusi konsumsi). Program *demand-response* atau penjadwalan "
            f"ulang beban pada segmen ini akan berdampak besar pada beban puncak jaringan secara "
            f"keseluruhan."
        )

    # 3) Tingkat anomali di atas rata-rata populasi
    if p["anomaly_rate_pct"] is not None and p["overall_anomaly_rate_pct"] is not None:
        selisih_anomali = p["anomaly_rate_pct"] - p["overall_anomaly_rate_pct"]
        if selisih_anomali > 1.0:
            perkiraan_baris_anomali = round(p["n_rows"] * p["anomaly_rate_pct"] / 100)
            recs.append(
                f"Tingkat anomali **{p['anomaly_rate_pct']:.1f}%** pada segmen ini — "
                f"**{selisih_anomali:+.1f} poin persentase** di atas rata-rata populasi "
                f"({p['overall_anomaly_rate_pct']:.1f}%), setara ±**{perkiraan_baris_anomali:,} "
                f"catatan harian**. Pertimbangkan pemeriksaan meter berkala atau investigasi "
                f"kemungkinan gangguan pencatatan pada segmen ini."
            )

    # 4) Pola akhir pekan vs hari kerja
    if p["weekend_vs_weekday_pct"] is not None and abs(p["weekend_vs_weekday_pct"]) > 15:
        if p["weekend_vs_weekday_pct"] > 0:
            recs.append(
                f"Konsumsi akhir pekan **{p['weekend_vs_weekday_pct']:+.0f}%** lebih tinggi "
                f"dibanding hari kerja — mengindikasikan pola hunian/rumah tangga. Edukasi hemat "
                f"energi domestik atau promosi tarif khusus akhir pekan berpotensi relevan untuk "
                f"segmen ini."
            )
        else:
            recs.append(
                f"Konsumsi hari kerja **{abs(p['weekend_vs_weekday_pct']):.0f}%** lebih tinggi "
                f"dibanding akhir pekan — mengindikasikan pola operasional bisnis/industri. Evaluasi "
                f"jadwal operasional terhadap jam beban puncak jaringan berpotensi menurunkan biaya "
                f"beban puncak segmen ini."
            )

    # 5) Load factor rendah -> beban terkonsentrasi, kandidat BESS/load-shifting
    if (p["load_factor_vs_overall_pct"] is not None and p["avg_load_factor"] is not None
            and p["load_factor_vs_overall_pct"] < -15):
        recs.append(
            f"Load factor rata-rata segmen ini **{p['avg_load_factor']:.2f}** — "
            f"**{p['load_factor_vs_overall_pct']:.0f}%** lebih rendah dari populasi, artinya beban "
            f"lebih terkonsentrasi di jam puncak (kurang merata sepanjang hari). Insentif pergeseran "
            f"beban (*load shifting*) atau penyimpanan energi (BESS) berpotensi menekan biaya "
            f"kapasitas untuk segmen ini."
        )

    # 6) Fallback bila tidak ada sinyal menonjol
    if not recs:
        recs.append(
            f"Segmen ini relatif stabil dengan pola konsumsi mendekati rata-rata populasi "
            f"(**{p['avg_consumption']:,.0f} kWh/hari**, {p['consumption_vs_overall_pct']:+.0f}% dari "
            f"rata-rata keseluruhan). Pertahankan layanan yang ada dan pantau secara berkala; belum "
            f"ada indikasi kuat yang memerlukan tindakan segera."
        )

    return recs


def build_cluster_profiles(df: pd.DataFrame, cluster_col: str = "cluster") -> list[ClusterProfile]:
    """
    Menghitung profil lengkap (statistik + persona + rekomendasi) untuk
    setiap segmen pada kolom `cluster_col`, dibandingkan terhadap populasi
    keseluruhan `df`.

    `cluster_col` dapat diarahkan ke kolom segmen level-hari (`cluster`)
    atau segmen level-pelanggan (`customer_segment`, hasil voting mayoritas
    — lihat `derive_customer_segment`). Untuk analisis & rekomendasi bisnis,
    gunakan segmen level-pelanggan agar tiap pelanggan masuk tepat satu
    segmen dan persentase antar segmen menjumlah 100%.
    """
    overall_avg_consumption = df["consumption_kwh"].mean()
    overall_total_consumption = df["consumption_kwh"].sum()
    overall_anomaly_rate = df["is_anomaly"].mean() * 100 if "is_anomaly" in df.columns else None
    overall_load_factor = df["load_factor"].mean() if "load_factor" in df.columns else None

    n_total_clusters = df[cluster_col].nunique()
    cluster_totals = df.groupby(cluster_col)["consumption_kwh"].sum().sort_values(ascending=False)
    rank_map = {cid: i + 1 for i, cid in enumerate(cluster_totals.index)}

    profiles: list[ClusterProfile] = []

    for cluster_id, group in df.groupby(cluster_col):
        n_customers = group["customer_id"].nunique() if "customer_id" in group.columns else len(group)
        n_rows = len(group)
        avg_consumption = group["consumption_kwh"].mean()
        total_consumption = group["consumption_kwh"].sum()
        consumption_vs_overall_pct = _pct_diff(avg_consumption, overall_avg_consumption)
        pct_of_total_consumption = (total_consumption / overall_total_consumption * 100
                                     if overall_total_consumption else 0.0)

        dominant_type, dominant_type_pct = (
            _dominant_with_share(group["customer_type"]) if "customer_type" in group.columns else (None, None)
        )
        dominant_region, dominant_region_pct = (
            _dominant_with_share(group["region"]) if "region" in group.columns else (None, None)
        )

        avg_load_factor = group["load_factor"].mean() if "load_factor" in group.columns else None
        load_factor_vs_overall_pct = (
            _pct_diff(avg_load_factor, overall_load_factor)
            if avg_load_factor is not None and overall_load_factor else None
        )

        anomaly_rate_pct = group["is_anomaly"].mean() * 100 if "is_anomaly" in group.columns else None

        weekend_vs_weekday_pct = None
        if "is_weekend" in group.columns:
            weekday_avg = group.loc[group["is_weekend"] == 0, "consumption_kwh"].mean()
            weekend_avg = group.loc[group["is_weekend"] == 1, "consumption_kwh"].mean()
            if pd.notna(weekday_avg) and pd.notna(weekend_avg):
                weekend_vs_weekday_pct = _pct_diff(weekend_avg, weekday_avg)

        p = {
            "cluster_id": cluster_id,
            "n_customers": n_customers,
            "pct_customers": (n_customers / df["customer_id"].nunique() * 100
                               if "customer_id" in df.columns and df["customer_id"].nunique() else 0.0),
            "n_rows": n_rows,
            "avg_consumption": avg_consumption,
            "overall_avg_consumption": overall_avg_consumption,
            "consumption_vs_overall_pct": consumption_vs_overall_pct,
            "total_consumption": total_consumption,
            "pct_of_total_consumption": pct_of_total_consumption,
            "consumption_rank": rank_map[cluster_id],
            "n_total_clusters": n_total_clusters,
            "dominant_type": dominant_type,
            "dominant_type_pct": dominant_type_pct,
            "dominant_region": dominant_region,
            "dominant_region_pct": dominant_region_pct,
            "avg_load_factor": avg_load_factor,
            "load_factor_vs_overall_pct": load_factor_vs_overall_pct,
            "anomaly_rate_pct": anomaly_rate_pct,
            "overall_anomaly_rate_pct": overall_anomaly_rate,
            "weekend_vs_weekday_pct": weekend_vs_weekday_pct,
        }

        persona = _build_persona(dominant_type, consumption_vs_overall_pct)
        recommendations = _build_recommendations(p)

        profiles.append(ClusterProfile(persona=persona, recommendations=recommendations, **p))

    return sorted(profiles, key=lambda prof: prof.cluster_id)


def derive_customer_segment(df: pd.DataFrame, daily_cluster_col: str = "cluster",
                             customer_col: str = "customer_id") -> pd.Series:
    """
    Menurunkan SATU segmen akhir per pelanggan dari kolom segmen harian.

    Klasterisasi SOM+K-Means berjalan pada tiap baris (observasi harian),
    sehingga satu pelanggan bisa punya hari-hari yang jatuh ke segmen
    berbeda (mis. pola weekday vs weekend yang berbeda cukup jauh). Untuk
    analisis & rekomendasi bisnis, tiap pelanggan sebaiknya diberi TEPAT
    SATU label segmen. Fungsi ini mengambil segmen yang paling sering
    muncul (modus) di antara hari-hari tiap pelanggan sebagai representasi
    segmen "khas"-nya.

    Mengembalikan Series selaras index dengan `df`, siap dipakai sebagai
    kolom baru (mis. `df['customer_segment'] = derive_customer_segment(df)`).
    """
    majority = df.groupby(customer_col)[daily_cluster_col].agg(lambda s: s.mode().iat[0])
    return df[customer_col].map(majority)
