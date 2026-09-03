import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Untuk model SOM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from minisom import MiniSom
from sklearn.cluster import KMeans

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Pola Konsumsi Listrik dengan SOM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul aplikasi
st.title("⚡ Analisis Pola Konsumsi Listrik Menggunakan Big Data dan Self-Organizing Map (SOM)")
st.markdown("""
Aplikasi ini menganalisis pola konsumsi listrik dengan teknik **Self-Organizing Map (SOM)** untuk mengelompokkan 
konsumsi berdasarkan pola harian, mingguan, dan musiman. Aplikasi ini dapat membantu mengidentifikasi:
- Pola konsumsi pelanggan
- Anomali dalam penggunaan listrik
- Segmentasi pelanggan berdasarkan pola konsumsi
- Potensi penghematan energi
""")

# Sidebar untuk navigasi
st.sidebar.title("⚙️ Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["📊 Data Overview", "🔧 Preprocessing", "🧠 Model SOM", "📈 Visualisasi Hasil", "📋 Analisis Kluster"]
)

# Fungsi untuk membuat dataset contoh
@st.cache_data
def generate_sample_data():
    """Membuat dataset contoh konsumsi listrik"""
    
    # Parameter dataset
    n_customers = 500
    n_days = 365
    start_date = datetime(2023, 1, 1)
    
    # Buat data tanggal
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Buat data untuk setiap pelanggan
    data = []
    customer_ids = []
    
    for i in range(n_customers):
        customer_id = f"CUST_{i+1:04d}"
        customer_ids.append(customer_id)
        
        # Tentukan tipe pelanggan (random)
        customer_type = np.random.choice(["Rumah Tangga", "Industri Kecil", "Industri Besar", "Komersial"], 
                                        p=[0.5, 0.2, 0.1, 0.2])
        
        # Tentukan wilayah (random)
        region = np.random.choice(["Jakarta", "Bandung", "Surabaya", "Medan", "Makassar"])
        
        # Generate pola konsumsi berdasarkan tipe pelanggan
        base_consumption = 0
        
        if customer_type == "Rumah Tangga":
            base_consumption = np.random.uniform(50, 200)
            # Pola: puncak di malam hari, rendah di siang
            pattern = "Malam"
        elif customer_type == "Industri Kecil":
            base_consumption = np.random.uniform(200, 800)
            pattern = "Siang"
        elif customer_type == "Industri Besar":
            base_consumption = np.random.uniform(1000, 5000)
            pattern = "24 Jam"
        else:  # Komersial
            base_consumption = np.random.uniform(300, 1500)
            pattern = "Siang"
        
        # Generate konsumsi harian dengan pola musiman dan acak
        for j, date in enumerate(dates):
            # Faktor musiman (lebih tinggi di musim panas)
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * j / 365)
            
            # Faktor hari dalam minggu (akhir pekan vs weekday)
            weekday_factor = 0.9 if date.weekday() >= 5 else 1.1
            
            # Faktor acak
            random_factor = np.random.normal(1, 0.1)
            
            # Hitung konsumsi
            consumption = base_consumption * seasonal_factor * weekday_factor * random_factor
            
            # Tambahkan beberapa outlier/anomali
            if np.random.random() < 0.02:  # 2% peluang anomali
                consumption *= np.random.uniform(1.5, 3)
            
            data.append({
                'customer_id': customer_id,
                'date': date,
                'consumption_kwh': consumption,
                'customer_type': customer_type,
                'region': region,
                'consumption_pattern': pattern,
                'month': date.month,
                'day_of_week': date.weekday(),
                'is_weekend': 1 if date.weekday() >= 5 else 0
            })
    
    df = pd.DataFrame(data)
    
    # Tambahkan fitur agregat untuk setiap pelanggan
    customer_stats = df.groupby('customer_id').agg({
        'consumption_kwh': ['mean', 'std', 'min', 'max', 'sum']
    }).reset_index()
    
    customer_stats.columns = ['customer_id', 'avg_consumption', 'std_consumption', 
                              'min_consumption', 'max_consumption', 'total_consumption']
    
    # Gabungkan dengan data asli
    df = pd.merge(df, customer_stats, on='customer_id')
    
    # Tambahkan flag untuk anomali (berdasarkan z-score)
    df['z_score'] = (df['consumption_kwh'] - df['avg_consumption']) / df['std_consumption']
    df['is_anomaly'] = (df['z_score'].abs() > 3).astype(int)
    
    return df

# Fungsi untuk preprocessing data
def preprocess_data(df, features):
    """Preprocessing data untuk SOM"""
    
    # Pilih fitur yang akan digunakan
    X = df[features].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Normalisasi data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

# Fungsi untuk melatih model SOM
def train_som(X_scaled, grid_size, sigma, learning_rate, iterations):
    """Melatih model Self-Organizing Map"""
    
    # Hitung ukuran grid optimal jika tidak ditentukan
    if grid_size == "auto":
        grid_size = int(np.sqrt(5 * np.sqrt(X_scaled.shape[0])))
        st.info(f"Ukuran grid otomatis: {grid_size}x{grid_size}")
    
    # Inisialisasi SOM
    som = MiniSom(
        x=grid_size, 
        y=grid_size, 
        input_len=X_scaled.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        random_seed=42
    )
    
    # Inisialisasi weights
    som.random_weights_init(X_scaled)
    
    # Training progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Training SOM
    for i in range(iterations):
        # Pilih sampel acak
        rand_index = np.random.randint(0, X_scaled.shape[0])
        som.update(X_scaled[rand_index], som.winner(X_scaled[rand_index]), i, iterations)
        
        # Update progress bar setiap 10%
        if i % (iterations // 10) == 0:
            progress = (i + 1) / iterations
            progress_bar.progress(progress)
            status_text.text(f"Training SOM: {int(progress * 100)}% selesai")
    
    progress_bar.progress(1.0)
    status_text.text("Training SOM selesai!")
    
    return som

# Fungsi untuk visualisasi U-Matrix
def plot_umatrix(som, X_scaled):
    """Plot U-Matrix dari SOM"""
    
    # Hitung U-Matrix
    umatrix = som.distance_map()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot U-Matrix
    im = ax.imshow(umatrix.T, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax, label='Jarak')
    
    # Tampilkan jumlah data pada setiap neuron
    counts = np.zeros((som.x, som.y))
    for i, x in enumerate(X_scaled):
        w = som.winner(x)
        counts[w[0], w[1]] += 1
    
    # Tampilkan jumlah data
    for i in range(som.x):
        for j in range(som.y):
            if counts[i, j] > 0:
                ax.text(i, j, f'{int(counts[i, j])}', 
                       ha='center', va='center', 
                       color='white' if umatrix[i, j] > 0.5 else 'black',
                       fontsize=8)
    
    ax.set_title('U-Matrix dan Jumlah Data per Neuron')
    ax.set_xlabel('Neuron X')
    ax.set_ylabel('Neuron Y')
    
    return fig

# Fungsi untuk visualisasi komponen
def plot_component_planes(som, feature_names):
    """Plot component planes untuk setiap fitur"""
    
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_names):
        ax = axes[i]
        component_plane = som.get_weights()[:, :, i]
        
        im = ax.imshow(component_plane.T, cmap='coolwarm', origin='lower')
        ax.set_title(f'Komponen: {feature}')
        ax.set_xlabel('Neuron X')
        ax.set_ylabel('Neuron Y')
        
        plt.colorbar(im, ax=ax)
    
    # Sembunyikan axes yang tidak terpakai
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

# Inisialisasi session state
if 'df' not in st.session_state:
    st.session_state.df = generate_sample_data()

if 'som_model' not in st.session_state:
    st.session_state.som_model = None

if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None

if 'features' not in st.session_state:
    st.session_state.features = None

# Halaman Data Overview
if page == "📊 Data Overview":
    st.header("📊 Data Overview")
    
    st.markdown("""
    ### Dataset Konsumsi Listrik
    Dataset ini berisi data konsumsi listrik harian dari berbagai pelanggan dengan berbagai karakteristik.
    """)
    
    # Tampilkan dataset
    st.subheader("Preview Data")
    st.dataframe(st.session_state.df.head(100), use_container_width=True)
    
    # Statistik dataset
    st.subheader("Statistik Dataset")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Jumlah Pelanggan", st.session_state.df['customer_id'].nunique())
    
    with col2:
        st.metric("Jumlah Data", f"{len(st.session_state.df):,}")
    
    with col3:
        st.metric("Rentang Tanggal", 
                 f"{st.session_state.df['date'].min().date()} hingga {st.session_state.df['date'].max().date()}")
    
    with col4:
        st.metric("Total Konsumsi (kWh)", f"{st.session_state.df['consumption_kwh'].sum():,.0f}")
    
    # Distribusi tipe pelanggan
    st.subheader("Distribusi Tipe Pelanggan")
    customer_type_dist = st.session_state.df.groupby('customer_type')['customer_id'].nunique().reset_index()
    customer_type_dist.columns = ['Tipe Pelanggan', 'Jumlah']
    
    fig = px.bar(customer_type_dist, x='Tipe Pelanggan', y='Jumlah',
                 color='Tipe Pelanggan', title='Distribusi Tipe Pelanggan')
    st.plotly_chart(fig, use_container_width=True)
    
    # Konsumsi per wilayah
    st.subheader("Konsumsi per Wilayah")
    region_consumption = st.session_state.df.groupby('region')['consumption_kwh'].sum().reset_index()
    region_consumption.columns = ['Wilayah', 'Total Konsumsi (kWh)']
    
    fig = px.pie(region_consumption, values='Total Konsumsi (kWh)', names='Wilayah',
                 title='Distribusi Konsumsi per Wilayah')
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series konsumsi
    st.subheader("Trend Konsumsi Harian")
    
    # Agregat konsumsi harian
    daily_consumption = st.session_state.df.groupby('date')['consumption_kwh'].sum().reset_index()
    
    fig = px.line(daily_consumption, x='date', y='consumption_kwh',
                  title='Total Konsumsi Harian',
                  labels={'consumption_kwh': 'Konsumsi (kWh)', 'date': 'Tanggal'})
    
    # Tambahkan rata-rata bergerak
    daily_consumption['moving_avg'] = daily_consumption['consumption_kwh'].rolling(window=7).mean()
    fig.add_scatter(x=daily_consumption['date'], y=daily_consumption['moving_avg'],
                    mode='lines', name='Rata-rata Bergerak (7 hari)',
                    line=dict(color='red', width=2))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Opsi untuk menggunakan data sendiri
    st.subheader("Gunakan Data Sendiri")
    uploaded_file = st.file_uploader("Unggah file CSV dengan data konsumsi listrik", type=['csv'])
    
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.session_state.df = user_df
            st.success("Data berhasil diunggah! Data contoh telah digantikan dengan data Anda.")
            st.dataframe(user_df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error membaca file: {e}")
    
    # Tombol untuk generate data contoh baru
    if st.button("Generate Data Contoh Baru"):
        st.session_state.df = generate_sample_data()
        st.success("Data contoh baru telah di-generate!")
        st.rerun()

# Halaman Preprocessing
elif page == "🔧 Preprocessing":
    st.header("🔧 Preprocessing Data")
    
    st.markdown("""
    ### Persiapan Data untuk SOM
    Pada tahap ini, kita akan:
    1. Memilih fitur yang akan digunakan
    2. Menangani missing values
    3. Normalisasi data
    """)
    
    df = st.session_state.df
    
    # Pilih fitur numerik
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Hapus kolom yang tidak perlu
    exclude_cols = ['is_anomaly', 'z_score']
    available_features = [col for col in numeric_cols if col not in exclude_cols]
    
    st.subheader("Pilih Fitur untuk Analisis SOM")
    
    # Pilih fitur secara manual atau otomatis
    feature_selection = st.radio(
        "Metode Pemilihan Fitur:",
        ["Pilih secara manual", "Gunakan semua fitur numerik", "Gunakan fitur rekomendasi"]
    )
    
    if feature_selection == "Pilih secara manual":
        selected_features = st.multiselect(
            "Pilih fitur untuk analisis SOM:",
            available_features,
            default=['consumption_kwh', 'avg_consumption', 'std_consumption', 'month', 'day_of_week']
        )
    elif feature_selection == "Gunakan semua fitur numerik":
        selected_features = available_features
    else:  # Fitur rekomendasi
        selected_features = ['consumption_kwh', 'avg_consumption', 'std_consumption', 
                           'min_consumption', 'max_consumption', 'month', 'day_of_week', 'is_weekend']
        selected_features = [f for f in selected_features if f in available_features]
    
    st.info(f"Fitur yang dipilih: {', '.join(selected_features)}")
    
    # Preprocessing data
    if selected_features:
        # Pilih metode normalisasi
        st.subheader("Pengaturan Normalisasi")
        normalization_method = st.selectbox(
            "Metode Normalisasi:",
            ["MinMax Scaling (0-1)", "Standard Scaling (mean=0, std=1)"]
        )
        
        # Preprocess data
        X = df[selected_features].copy()
        
        # Handle missing values
        missing_values = X.isnull().sum().sum()
        if missing_values > 0:
            st.warning(f"Ditemukan {missing_values} missing values. Mengisi dengan mean.")
            X = X.fillna(X.mean())
        
        # Normalisasi
        if normalization_method == "MinMax Scaling (0-1)":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X)
        
        # Simpan ke session state
        st.session_state.X_scaled = X_scaled
        st.session_state.features = selected_features
        st.session_state.scaler = scaler
        
        # Tampilkan hasil preprocessing
        st.subheader("Hasil Preprocessing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Sebelum Normalisasi (5 baris pertama)**")
            st.dataframe(X.head(), use_container_width=True)
        
        with col2:
            # Buat dataframe scaled
            X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)
            st.markdown("**Data Setelah Normalisasi (5 baris pertama)**")
            st.dataframe(X_scaled_df.head(), use_container_width=True)
        
        # Visualisasi distribusi sebelum dan sesudah
        st.subheader("Perbandingan Distribusi Fitur")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Sebelum Normalisasi", "Setelah Normalisasi")
        )
        
        # Pilih satu fitur untuk contoh visualisasi
        if 'consumption_kwh' in selected_features:
            feature_idx = selected_features.index('consumption_kwh')
            
            fig.add_trace(
                go.Histogram(x=X.iloc[:, feature_idx], name="Sebelum"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=X_scaled[:, feature_idx], name="Sesudah"),
                row=1, col=2
            )
        else:
            # Gunakan fitur pertama
            fig.add_trace(
                go.Histogram(x=X.iloc[:, 0], name="Sebelum"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=X_scaled[:, 0], name="Sesudah"),
                row=1, col=2
            )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Korelasi antara fitur
        st.subheader("Matriks Korelasi Fitur")
        
        fig = px.imshow(X.corr(),
                       title="Matriks Korelasi Antar Fitur",
                       color_continuous_scale='RdBu',
                       zmin=-1, zmax=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("Preprocessing selesai! Data siap untuk training SOM.")
    else:
        st.warning("Silakan pilih setidaknya satu fitur untuk dilanjutkan.")

# Halaman Model SOM
elif page == "🧠 Model SOM":
    st.header("🧠 Training Model Self-Organizing Map")
    
    if st.session_state.X_scaled is None:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu di halaman '🔧 Preprocessing'.")
    else:
        st.markdown("""
        ### Konfigurasi Model SOM
        Atur parameter untuk training Self-Organizing Map.
        """)
        
        X_scaled = st.session_state.X_scaled
        features = st.session_state.features
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Parameter SOM
            grid_size_option = st.selectbox(
                "Ukuran Grid SOM:",
                ["auto", "manual"]
            )
            
            if grid_size_option == "manual":
                grid_x = st.slider("Lebar Grid (X)", min_value=5, max_value=20, value=10)
                grid_y = st.slider("Tinggi Grid (Y)", min_value=5, max_value=20, value=10)
                grid_size = (grid_x, grid_y)
            else:
                grid_size = "auto"
            
            sigma = st.slider(
                "Radius Awal (Sigma)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Radius tetangga pada awal training"
            )
        
        with col2:
            learning_rate = st.slider(
                "Learning Rate Awal",
                min_value=0.01,
                max_value=1.0,
                value=0.5,
                step=0.01
            )
            
            iterations = st.slider(
                "Jumlah Iterasi",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
        
        # Training SOM
        if st.button("🚀 Train SOM Model", type="primary"):
            with st.spinner("Training SOM..."):
                # Tentukan ukuran grid
                if grid_size == "auto":
                    grid_dim = int(np.sqrt(5 * np.sqrt(X_scaled.shape[0])))
                    grid_size_tuple = (grid_dim, grid_dim)
                    st.info(f"Ukuran grid otomatis: {grid_dim}x{grid_dim}")
                else:
                    grid_size_tuple = grid_size
                
                # Training SOM
                som = MiniSom(
                    x=grid_size_tuple[0],
                    y=grid_size_tuple[1],
                    input_len=X_scaled.shape[1],
                    sigma=sigma,
                    learning_rate=learning_rate,
                    random_seed=42
                )
                
                # Inisialisasi weights
                som.random_weights_init(X_scaled)
                
                # Training dengan progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(iterations):
                    # Pilih sampel acak
                    rand_index = np.random.randint(0, X_scaled.shape[0])
                    som.update(X_scaled[rand_index], som.winner(X_scaled[rand_index]), i, iterations)
                    
                    # Update progress bar setiap 5%
                    if i % (iterations // 20) == 0:
                        progress = (i + 1) / iterations
                        progress_bar.progress(progress)
                        status_text.text(f"Training SOM: {int(progress * 100)}% selesai")
                
                progress_bar.progress(1.0)
                status_text.text("Training SOM selesai!")
                
                # Simpan model ke session state
                st.session_state.som_model = som
                
                st.success(f"Model SOM berhasil dilatih dengan ukuran grid {grid_size_tuple[0]}x{grid_size_tuple[1]}!")
                
                # Hitung quantization error
                q_error = som.quantization_error(X_scaled)
                st.metric("Quantization Error", f"{q_error:.4f}")
        
        # Tampilkan informasi model jika sudah ada
        if st.session_state.som_model is not None:
            st.subheader("Model SOM yang Telah Dilatih")
            som = st.session_state.som_model
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ukuran Grid", f"{som.x}x{som.y}")
            with col2:
                st.metric("Jumlah Neuron", som.x * som.y)
            with col3:
                q_error = som.quantization_error(X_scaled)
                st.metric("Quantization Error", f"{q_error:.4f}")
            
            # Hitung kluster untuk setiap data point
            st.subheader("Kluster Hasil SOM")
            
            # Tentukan kluster untuk setiap data point
            df = st.session_state.df.copy()
            winner_coordinates = []
            for i, x in enumerate(X_scaled):
                winner = som.winner(x)
                winner_coordinates.append(winner)
            
            # Konversi koordinat menjadi kluster ID
            cluster_ids = [coord[0] * som.y + coord[1] for coord in winner_coordinates]
            df['cluster'] = cluster_ids
            df['neuron_x'] = [coord[0] for coord in winner_coordinates]
            df['neuron_y'] = [coord[1] for coord in winner_coordinates]
            
            # Hitung jumlah data per kluster
            cluster_counts = df['cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Kluster', 'Jumlah Data']
            
            fig = px.bar(cluster_counts.head(20), x='Kluster', y='Jumlah Data',
                        title='Distribusi Data per Kluster (Top 20)',
                        color='Jumlah Data')
            st.plotly_chart(fig, use_container_width=True)
            
            # Simpan dataframe dengan kluster ke session state
            st.session_state.df_with_clusters = df

# Halaman Visualisasi Hasil
elif page == "📈 Visualisasi Hasil":
    st.header("📈 Visualisasi Hasil SOM")
    
    if st.session_state.som_model is None:
        st.warning("Silakan train model SOM terlebih dahulu di halaman '🧠 Model SOM'.")
    else:
        som = st.session_state.som_model
        X_scaled = st.session_state.X_scaled
        features = st.session_state.features
        df = st.session_state.df
        
        st.markdown("""
        ### Visualisasi Self-Organizing Map
        Berikut adalah visualisasi dari hasil training SOM.
        """)
        
        # Pilih tipe visualisasi
        viz_type = st.selectbox(
            "Pilih Tipe Visualisasi:",
            ["U-Matrix", "Component Planes", "Kluster pada SOM", "Hit Histogram"]
        )
        
        if viz_type == "U-Matrix":
            st.subheader("U-Matrix (Unified Distance Matrix)")
            st.markdown("""
            U-Matrix menunjukkan jarak antara neuron. Area dengan warna terang menunjukkan 
            batas antara kluster, sedangkan area gelap menunjukkan neuron yang serupa.
            """)
            
            fig = plot_umatrix(som, X_scaled)
            st.pyplot(fig)
            
            st.markdown("""
            **Interpretasi U-Matrix:**
            - **Area biru gelap**: Neuron-neuron yang sangat mirip (kluster yang padat)
            - **Area kuning/terang**: Batas antara kluster yang berbeda
            - **Angka pada setiap sel**: Jumlah data point yang dipetakan ke neuron tersebut
            """)
        
        elif viz_type == "Component Planes":
            st.subheader("Component Planes")
            st.markdown("""
            Component planes menunjukkan distribusi nilai untuk setiap fitur pada peta SOM.
            Ini membantu memahami karakteristik setiap area pada peta.
            """)
            
            fig = plot_component_planes(som, features)
            st.pyplot(fig)
            
            st.markdown("""
            **Interpretasi Component Planes:**
            - Setiap subplot menunjukkan distribusi satu fitur
            - Pola yang serupa antar component planes menunjukkan korelasi antar fitur
            - Area dengan nilai tinggi/rendah untuk beberapa fitur menunjukkan pola yang khas
            """)
        
        elif viz_type == "Kluster pada SOM":
            st.subheader("Visualisasi Kluster pada Peta SOM")
            
            # Tentukan kluster untuk setiap data point
            winner_coordinates = []
            for i, x in enumerate(X_scaled):
                winner = som.winner(x)
                winner_coordinates.append(winner)
            
            # Konversi ke dataframe
            df_viz = pd.DataFrame(winner_coordinates, columns=['x', 'y'])
            df_viz['count'] = 1
            
            # Agregat per neuron
            neuron_counts = df_viz.groupby(['x', 'y']).count().reset_index()
            
            # Buat plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Buat scatter plot
            scatter = ax.scatter(neuron_counts['x'], neuron_counts['y'], 
                               s=neuron_counts['count']*10,  # Ukuran berdasarkan jumlah data
                               c=neuron_counts['count'], 
                               cmap='viridis', alpha=0.7)
            
            # Tambahkan label jumlah
            for i, row in neuron_counts.iterrows():
                ax.text(row['x'], row['y'], str(row['count']),
                       ha='center', va='center', fontsize=8, color='white')
            
            ax.set_xlabel('Neuron X')
            ax.set_ylabel('Neuron Y')
            ax.set_title('Distribusi Data pada Peta SOM')
            ax.set_xlim(-0.5, som.x - 0.5)
            ax.set_ylim(-0.5, som.y - 0.5)
            ax.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, ax=ax, label='Jumlah Data')
            st.pyplot(fig)
        
        else:  # Hit Histogram
            st.subheader("Hit Histogram")
            
            # Hitung hits per neuron
            hit_map = np.zeros((som.x, som.y))
            for i, x in enumerate(X_scaled):
                winner = som.winner(x)
                hit_map[winner[0], winner[1]] += 1
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(hit_map.T, cmap='YlOrRd', origin='lower')
            
            # Tambahkan nilai pada setiap sel
            for i in range(som.x):
                for j in range(som.y):
                    if hit_map[i, j] > 0:
                        ax.text(i, j, f'{int(hit_map[i, j])}', 
                               ha='center', va='center', 
                               color='black' if hit_map[i, j] < np.max(hit_map)/2 else 'white',
                               fontsize=8)
            
            ax.set_title('Hit Histogram - Jumlah Data per Neuron')
            ax.set_xlabel('Neuron X')
            ax.set_ylabel('Neuron Y')
            plt.colorbar(im, ax=ax, label='Jumlah Data')
            st.pyplot(fig)
        
        # Tambahkan analisis tambahan
        st.subheader("Analisis Dimensionality Reduction dengan PCA")
        
        # Lakukan PCA untuk visualisasi 2D/3D
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        # Buat dataframe untuk plot
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
        
        # Tambahkan kluster jika ada
        if 'df_with_clusters' in st.session_state:
            df_pca['cluster'] = st.session_state.df_with_clusters['cluster']
        
        # Pilih visualisasi PCA
        pca_viz = st.radio(
            "Visualisasi PCA:",
            ["2D Scatter Plot", "3D Scatter Plot"]
        )
        
        if pca_viz == "2D Scatter Plot":
            if 'cluster' in df_pca.columns:
                fig = px.scatter(df_pca, x='PC1', y='PC2', color='cluster',
                                title='PCA 2D - Proyeksi Data dengan Warna Kluster',
                                hover_data={'cluster': True})
            else:
                fig = px.scatter(df_pca, x='PC1', y='PC2',
                                title='PCA 2D - Proyeksi Data')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            if 'cluster' in df_pca.columns:
                fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='cluster',
                                   title='PCA 3D - Proyeksi Data dengan Warna Kluster')
            else:
                fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3',
                                   title='PCA 3D - Proyeksi Data')
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"Varians yang dijelaskan oleh 3 komponen PCA: {np.sum(pca.explained_variance_ratio_):.2%}")

# Halaman Analisis Kluster
elif page == "📋 Analisis Kluster":
    st.header("📋 Analisis Kluster dan Segmentasi Pelanggan")
    
    if 'df_with_clusters' not in st.session_state:
        st.warning("Silakan train model SOM terlebih dahulu untuk mendapatkan hasil kluster.")
    else:
        df = st.session_state.df_with_clusters
        som = st.session_state.som_model
        
        st.markdown("""
        ### Analisis Segmentasi Pelanggan Berdasarkan Pola Konsumsi
        Halaman ini menganalisis karakteristik setiap kluster yang dihasilkan oleh SOM.
        """)
        
        # Ringkasan kluster
        st.subheader("Ringkasan Kluster")
        
        # Hitung statistik per kluster
        cluster_stats = df.groupby('cluster').agg({
            'customer_id': 'nunique',
            'consumption_kwh': ['mean', 'std', 'sum'],
            'avg_consumption': 'mean',
            'std_consumption': 'mean',
            'customer_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
            'region': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        # Flatten multi-index columns
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
        cluster_stats = cluster_stats.reset_index()
        cluster_stats = cluster_stats.rename(columns={
            'customer_id_nunique': 'Jumlah Pelanggan',
            'consumption_kwh_mean': 'Konsumsi Rata-rata',
            'consumption_kwh_std': 'Std Dev Konsumsi',
            'consumption_kwh_sum': 'Total Konsumsi',
            'avg_consumption_mean': 'Rata-rata Konsumsi Hist.',
            'std_consumption_mean': 'Std Dev Konsumsi Hist.',
            'customer_type_<lambda>': 'Tipe Pelanggan Dominan',
            'region_<lambda>': 'Wilayah Dominan'
        })
        
        # Tampilkan tabel statistik kluster
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Pilih kluster untuk dianalisis lebih detail
        st.subheader("Analisis Detail per Kluster")
        
        clusters = sorted(df['cluster'].unique())
        selected_cluster = st.selectbox("Pilih Kluster untuk Analisis Detail:", clusters)
        
        # Filter data untuk kluster yang dipilih
        cluster_data = df[df['cluster'] == selected_cluster]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Jumlah Pelanggan", cluster_data['customer_id'].nunique())
        
        with col2:
            st.metric("Rata-rata Konsumsi (kWh)", f"{cluster_data['consumption_kwh'].mean():.2f}")
        
        with col3:
            st.metric("Total Konsumsi (kWh)", f"{cluster_data['consumption_kwh'].sum():,.0f}")
        
        # Visualisasi karakteristik kluster
        st.subheader(f"Karakteristik Kluster {selected_cluster}")
        
        # Distribusi tipe pelanggan dalam kluster
        fig1 = px.pie(cluster_data, names='customer_type', 
                     title=f'Distribusi Tipe Pelanggan - Kluster {selected_cluster}')
        
        # Distribusi wilayah dalam kluster
        fig2 = px.pie(cluster_data, names='region', 
                     title=f'Distribusi Wilayah - Kluster {selected_cluster}')
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        # Time series untuk beberapa pelanggan dalam kluster
        st.subheader(f"Pola Konsumsi Harian - Kluster {selected_cluster}")
        
        # Pilih beberapa pelanggan acak dari kluster
        sample_customers = cluster_data['customer_id'].unique()[:5]
        sample_data = cluster_data[cluster_data['customer_id'].isin(sample_customers)]
        
        fig = px.line(sample_data, x='date', y='consumption_kwh', color='customer_id',
                     title='Contoh Pola Konsumsi Harian',
                     labels={'consumption_kwh': 'Konsumsi (kWh)', 'date': 'Tanggal'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Analisis pola mingguan
        st.subheader(f"Pola Konsumsi Mingguan - Kluster {selected_cluster}")
        
        # Hitung rata-rata konsumsi per hari dalam minggu
        weekly_pattern = cluster_data.groupby('day_of_week')['consumption_kwh'].mean().reset_index()
        weekly_pattern['day_name'] = weekly_pattern['day_of_week'].map({
            0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 
            4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'
        })
        
        fig = px.bar(weekly_pattern, x='day_name', y='consumption_kwh',
                    title='Rata-rata Konsumsi per Hari dalam Minggu',
                    labels={'consumption_kwh': 'Konsumsi Rata-rata (kWh)', 'day_name': 'Hari'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Analisis anomali
        st.subheader(f"Deteksi Anomali - Kluster {selected_cluster}")
        
        if 'is_anomaly' in cluster_data.columns:
            anomaly_counts = cluster_data['is_anomaly'].value_counts()
            
            fig = px.pie(values=anomaly_counts.values, names=['Normal', 'Anomali'],
                        title='Distribusi Data Normal vs Anomali')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tampilkan data anomali
            if anomaly_counts.get(1, 0) > 0:
                anomalies = cluster_data[cluster_data['is_anomaly'] == 1]
                st.write(f"Ditemukan {len(anomalies)} data anomali:")
                st.dataframe(anomalies[['customer_id', 'date', 'consumption_kwh', 'z_score']].head(10), 
                           use_container_width=True)
        
        # Rekomendasi berdasarkan kluster
        st.subheader(f"Rekomendasi untuk Kluster {selected_cluster}")
        
        # Berikan rekomendasi berdasarkan karakteristik kluster
        avg_consumption = cluster_data['consumption_kwh'].mean()
        customer_type_mode = cluster_data['customer_type'].mode()[0]
        
        recommendations = []
        
        if avg_consumption > df['consumption_kwh'].mean() * 1.5:
            recommendations.append("✅ **Konsumsi tinggi**: Pertimbangkan audit energi untuk mengidentifikasi peluang penghematan.")
        
        if customer_type_mode == "Rumah Tangga":
            recommendations.append("✅ **Pelanggan rumah tangga**: Promosikan penggunaan peralatan hemat energi dan panel surya atap.")
        elif customer_type_mode == "Industri":
            recommendations.append("✅ **Pelanggan industri**: Pertimbangkan program demand-response dan optimalisasi beban puncak.")
        
        if cluster_data['std_consumption'].mean() > df['std_consumption'].mean():
            recommendations.append("✅ **Variasi konsumsi tinggi**: Pertimbangkan penyimpanan energi untuk menyeimbangkan beban.")
        
        if len(recommendations) == 0:
            recommendations.append("✅ **Kluster stabil**: Pertahankan layanan yang ada dan pantau secara berkala.")
        
        for rec in recommendations:
            st.markdown(rec)
        
        # Ekspor hasil analisis
        st.subheader("Ekspor Hasil Analisis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Ekspor Data dengan Kluster"):
                # Konversi dataframe ke CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"data_konsumsi_dengan_kluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Ekspor Statistik Kluster"):
                # Ekspor statistik kluster
                csv = cluster_stats.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"statistik_kluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Tentang Aplikasi")
st.sidebar.info("""
Aplikasi ini dikembangkan untuk analisis pola konsumsi listrik menggunakan algoritma **Self-Organizing Map (SOM)**.

**Fitur utama:**
- Analisis data konsumsi listrik
- Segmentasi pelanggan menggunakan SOM
- Deteksi pola dan anomali
- Visualisasi interaktif

**Teknologi:** Streamlit, MiniSOM, Scikit-learn, Plotly
""")

# Run aplikasi
if __name__ == "__main__":
    # Untuk menjalankan aplikasi, simpan kode ini sebagai file .py dan jalankan:
    # streamlit run nama_file.py
    pass