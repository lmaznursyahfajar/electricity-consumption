"""
styling.py
==========
Sistem desain terpusat untuk seluruh aplikasi: warna, tipografi, kartu,
badge status, dan header halaman/section yang konsisten. Dipusatkan di
sini agar setiap view punya tampilan yang seragam tanpa menyalin markup
HTML/CSS yang sama berulang kali.
"""

from __future__ import annotations

import streamlit as st

from . import config


def inject_global_css() -> None:
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}

        :root {{
            --primary: {config.PRIMARY_COLOR};
            --primary-dark: #163A8F;
            --accent: {config.ACCENT_COLOR};
            --success: #12B76A;
            --danger: #F04438;
            --text-primary: #111827;
            --text-secondary: #6B7280;
            --border-soft: #E5E7EB;
            --bg-soft: #F7F9FC;
        }}

        .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }}

        /* ---------- Hero header (atas tiap halaman) ---------- */
        .hero-header {{
            padding: 1.6rem 2rem;
            border-radius: 20px;
            background: linear-gradient(120deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            margin-bottom: 1.6rem;
            box-shadow: 0 8px 24px rgba(15, 98, 254, 0.18);
        }}
        .hero-header .eyebrow {{
            font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em;
            text-transform: uppercase; opacity: 0.75; margin-bottom: 0.3rem;
        }}
        .hero-header h1 {{ margin: 0; font-size: 1.65rem; font-weight: 800; line-height: 1.25; }}
        .hero-header p {{ margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 0.96rem; max-width: 62ch; }}

        /* ---------- Section title (dalam tab/isi halaman) ---------- */
        .section-title {{
            display: flex; align-items: center; gap: 0.55rem;
            margin: 1.6rem 0 0.5rem 0;
            padding-bottom: 0.45rem;
            border-bottom: 2px solid var(--border-soft);
        }}
        .section-title .icon {{
            font-size: 1.15rem; width: 2.1rem; height: 2.1rem; min-width: 2.1rem;
            border-radius: 10px; background: var(--bg-soft);
            display: flex; align-items: center; justify-content: center;
        }}
        .section-title h3 {{ margin: 0; font-size: 1.08rem; font-weight: 700; color: var(--text-primary); }}
        .section-title span.sub {{ display: block; font-size: 0.82rem; color: var(--text-secondary); font-weight: 400; }}

        /* ---------- Badge status ---------- */
        .badge {{
            display: inline-flex; align-items: center; gap: 0.3rem;
            padding: 0.22rem 0.7rem; border-radius: 999px;
            font-size: 0.78rem; font-weight: 600;
        }}
        .badge-success {{ background: rgba(18, 183, 106, 0.12); color: #0D8A50; }}
        .badge-neutral {{ background: rgba(107, 114, 128, 0.12); color: #4B5563; }}
        .badge-warning {{ background: rgba(247, 166, 27, 0.15); color: #B4740E; }}

        /* ---------- Kartu workflow / fitur di Beranda ---------- */
        .flow-card {{
            border: 1px solid var(--border-soft); border-radius: 16px;
            padding: 1.1rem 1.2rem; height: 100%;
            background: white; transition: box-shadow 0.15s ease;
        }}
        .flow-card:hover {{ box-shadow: 0 6px 18px rgba(17, 24, 39, 0.08); }}
        .flow-card .step-no {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 1.7rem; height: 1.7rem; border-radius: 999px;
            background: var(--primary); color: white; font-weight: 700; font-size: 0.82rem;
            margin-bottom: 0.55rem;
        }}
        .flow-card h4 {{ margin: 0 0 0.3rem 0; font-size: 0.98rem; font-weight: 700; color: var(--text-primary); }}
        .flow-card p {{ margin: 0; font-size: 0.85rem; color: var(--text-secondary); line-height: 1.45; }}

        /* ---------- Rapikan komponen bawaan Streamlit ---------- */
        [data-testid="stMetric"] {{ padding: 0.2rem 0; }}
        [data-testid="stMetricLabel"] {{ font-size: 0.82rem; color: var(--text-secondary); }}
        [data-testid="stMetricValue"] {{ font-size: 1.55rem; }}
        div[data-testid="stTabs"] button[role="tab"] {{ font-weight: 600; font-size: 0.92rem; }}
        section[data-testid="stSidebar"] {{ border-right: 1px solid var(--border-soft); }}
        section[data-testid="stSidebar"] .stRadio label {{ font-size: 0.93rem; }}
        div[data-testid="stExpander"] summary {{ font-weight: 600; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    """Header besar bergradasi di bagian atas tiap halaman utama."""
    eyebrow_html = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(
        f"""
        <div class="hero-header">
            {eyebrow_html}
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(icon: str, title: str, subtitle: str = "") -> None:
    """Header kecil untuk tiap bagian/section di dalam sebuah tab."""
    sub_html = f'<span class="sub">{subtitle}</span>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-title">
            <div class="icon">{icon}</div>
            <h3>{title}{sub_html}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "neutral") -> str:
    """Mengembalikan potongan HTML badge/pill — sisipkan lewat st.markdown(..., unsafe_allow_html=True)."""
    return f'<span class="badge badge-{kind}">{text}</span>'


def flow_card(step_no: int, title: str, description: str) -> str:
    """Mengembalikan potongan HTML kartu langkah kerja (dipakai di grid kolom Beranda)."""
    return f"""
    <div class="flow-card">
        <div class="step-no">{step_no}</div>
        <h4>{title}</h4>
        <p>{description}</p>
    </div>
    """


def sidebar_footer() -> None:
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"**{config.APP_TITLE}** · v{config.APP_VERSION}  \n"
        "Dibangun dengan Streamlit, MiniSom & Scikit-learn."
    )
