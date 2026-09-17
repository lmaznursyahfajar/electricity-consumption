"""
app.py
======
Entry point tunggal aplikasi "SIGAP Listrik". File ini bertindak sebagai
router: mengatur konfigurasi halaman, branding (logo), tema visual, dan
memuat data awal ke session state — lalu merender salah satu dari 4 menu
melalui `st.navigation`.

Jalankan aplikasi:
    streamlit run app.py
"""

import streamlit as st

from src import config, styling
from src.data_loader import load_bundled_dataset

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.logo("assets/logo.svg", icon_image="assets/logo_icon.svg", size="medium")
styling.inject_global_css()


def _ensure_session_data() -> None:
    """Memuat dataset bawaan sekali di awal sesi, dan menyiapkan kunci session state lain."""
    if "fact" not in st.session_state:
        with st.spinner("Memuat dataset awal..."):
            fact, dim_pelanggan, dim_wilayah = load_bundled_dataset()
        st.session_state.fact = fact
        st.session_state.dim_pelanggan = dim_pelanggan
        st.session_state.dim_wilayah = dim_wilayah
        st.session_state.data_source = "bawaan"

    st.session_state.setdefault("som_result", None)
    st.session_state.setdefault("X_scaled", None)
    st.session_state.setdefault("selected_features", None)
    st.session_state.setdefault("preprocessed_df", None)
    st.session_state.setdefault("df_with_clusters", None)
    st.session_state.setdefault("neuron_cluster_map", None)
    st.session_state.setdefault("n_final_clusters", None)


_ensure_session_data()

pages = [
    st.Page("views/beranda.py", title="Beranda", icon=":material/home:", default=True),
    st.Page("views/data_praproses.py", title="Data & Praproses", icon=":material/database:"),
    st.Page("views/model_visualisasi.py", title="Model & Visualisasi", icon=":material/hub:"),
    st.Page("views/kluster_dokumentasi.py", title="Kluster & Dokumentasi", icon=":material/insights:"),
]

pg = st.navigation(pages, position="sidebar")

with st.sidebar:
    n_pelanggan = st.session_state.dim_pelanggan["customer_id"].nunique()
    n_baris = len(st.session_state.fact)
    status_model = "Sudah dilatih" if st.session_state.som_result else "Belum dilatih"
    status_kind = "success" if st.session_state.som_result else "neutral"
    st.markdown(
        f"""
        <div style="font-size:0.82rem; line-height:1.9; color:#4B5563;">
        📦 <b>{n_pelanggan:,}</b> pelanggan · <b>{n_baris:,}</b> baris data<br/>
        🧠 Model SOM: {styling.badge(status_model, status_kind)}
        </div>
        """,
        unsafe_allow_html=True,
    )

pg.run()
styling.sidebar_footer()
