"""Halaman: Analisis Kluster — profil, rekomendasi, dan ekspor hasil segmentasi."""

from datetime import datetime

import streamlit as st

from src import styling, visualizations as viz

st.set_page_config(page_title="Analisis Kluster", page_icon="📋", layout="wide")
styling.inject_global_css()
styling.page_header("📋 Analisis Kluster & Segmentasi Pelanggan",
                     "Profil tiap kluster hasil SOM beserta rekomendasi tindak lanjut.")

if st.session_state.get("df_with_clusters") is None:
    st.warning("Latih model terlebih dahulu di halaman **🧠 Model SOM**.")
    st.stop()

df = st.session_state.df_with_clusters

agg_dict = {"customer_id": "nunique", "consumption_kwh": ["mean", "std", "sum"]}
if "customer_type" in df.columns:
    agg_dict["customer_type"] = lambda s: s.mode().iat[0] if not s.mode().empty else "Tidak diketahui"
if "region" in df.columns:
    agg_dict["region"] = lambda s: s.mode().iat[0] if not s.mode().empty else "Tidak diketahui"

st.subheader("Ringkasan Tiap Kluster")
cluster_stats = df.groupby("cluster").agg(agg_dict).round(2)
cluster_stats.columns = ["_".join([c for c in col if c]).strip("_") for col in cluster_stats.columns.values]
cluster_stats = cluster_stats.reset_index().rename(columns={
    "customer_id_nunique": "Jumlah Pelanggan",
    "consumption_kwh_mean": "Konsumsi Rata-rata",
    "consumption_kwh_std": "Std Dev Konsumsi",
    "consumption_kwh_sum": "Total Konsumsi",
    "customer_type_<lambda>": "Tipe Dominan",
    "region_<lambda>": "Wilayah Dominan",
})
st.dataframe(cluster_stats, width='stretch')

st.subheader("Analisis Detail per Kluster")
clusters = sorted(df["cluster"].unique())
selected_cluster = st.selectbox("Pilih kluster:", clusters)
cluster_data = df[df["cluster"] == selected_cluster]

k1, k2, k3 = st.columns(3)
k1.metric("Jumlah Pelanggan", cluster_data["customer_id"].nunique())
k2.metric("Rata-rata Konsumsi (kWh)", f"{cluster_data['consumption_kwh'].mean():.2f}")
k3.metric("Total Konsumsi (kWh)", f"{cluster_data['consumption_kwh'].sum():,.0f}")

if "customer_type" in cluster_data.columns and "region" in cluster_data.columns:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.pie_distribution(cluster_data, "customer_type", None,
                                              f"Distribusi Tipe Pelanggan — Kluster {selected_cluster}"),
                         width='stretch')
    with c2:
        st.plotly_chart(viz.pie_distribution(cluster_data, "region", None,
                                              f"Distribusi Wilayah — Kluster {selected_cluster}"),
                         width='stretch')

if "day_of_week" in cluster_data.columns:
    st.subheader(f"Pola Konsumsi Mingguan — Kluster {selected_cluster}")
    weekly = cluster_data.groupby("day_of_week")["consumption_kwh"].mean().reset_index()
    day_names = {0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis", 4: "Jumat", 5: "Sabtu", 6: "Minggu"}
    weekly["Hari"] = weekly["day_of_week"].map(day_names)
    st.plotly_chart(viz.bar_by_category(weekly, "Hari", "consumption_kwh",
                                         "Rata-rata Konsumsi per Hari dalam Minggu"), width='stretch')

if "is_anomaly" in cluster_data.columns:
    st.subheader(f"Deteksi Anomali — Kluster {selected_cluster}")
    anomaly_counts = cluster_data["is_anomaly"].value_counts().reindex([0, 1], fill_value=0)
    anomaly_df = __import__("pandas").DataFrame({"Status": ["Normal", "Anomali"], "Jumlah": anomaly_counts.values})
    st.plotly_chart(viz.pie_distribution(anomaly_df, "Status", "Jumlah", "Distribusi Data Normal vs Anomali"),
                     width='stretch')
    if anomaly_counts.get(1, 0) > 0:
        anomalies = cluster_data[cluster_data["is_anomaly"] == 1]
        st.write(f"Ditemukan **{len(anomalies):,}** baris anomali pada kluster ini (menampilkan 10 pertama):")
        show_cols = [c for c in ["customer_id", "date", "consumption_kwh", "z_score"] if c in anomalies.columns]
        st.dataframe(anomalies[show_cols].head(10), width='stretch')

st.subheader(f"💡 Rekomendasi untuk Kluster {selected_cluster}")
recommendations = []
avg_consumption = cluster_data["consumption_kwh"].mean()
overall_avg = df["consumption_kwh"].mean()

if avg_consumption > overall_avg * 1.5:
    recommendations.append("**Konsumsi tinggi** — pertimbangkan audit energi untuk peluang efisiensi.")
if "customer_type" in cluster_data.columns and not cluster_data["customer_type"].mode().empty:
    dominant_type = cluster_data["customer_type"].mode().iat[0]
    if dominant_type == "Rumah Tangga":
        recommendations.append("**Didominasi Rumah Tangga** — edukasi hemat energi & promosi panel surya atap.")
    elif "Industri" in dominant_type:
        recommendations.append("**Didominasi Industri** — pertimbangkan skema *demand-response* & optimalisasi beban puncak.")
    elif "Bisnis" in dominant_type:
        recommendations.append("**Didominasi Bisnis** — evaluasi jam operasional vs jam beban puncak jaringan.")
if "std_consumption" in cluster_data.columns and cluster_data["std_consumption"].mean() > df["std_consumption"].mean():
    recommendations.append("**Variasi konsumsi tinggi** — pertimbangkan penyimpanan energi (BESS) untuk menyeimbangkan beban.")
if not recommendations:
    recommendations.append("**Kluster relatif stabil** — pertahankan layanan yang ada dan pantau berkala.")

for rec in recommendations:
    st.markdown(f"✅ {rec}")

st.subheader("⬇️ Ekspor Hasil")
e1, e2 = st.columns(2)
with e1:
    st.download_button(
        "Unduh Data + Kluster (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"data_dengan_kluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv", width='stretch',
    )
with e2:
    st.download_button(
        "Unduh Statistik Kluster (CSV)",
        data=cluster_stats.to_csv(index=False).encode("utf-8"),
        file_name=f"statistik_kluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv", width='stretch',
    )

styling.sidebar_footer()
