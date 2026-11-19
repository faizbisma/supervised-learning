"""
Kasus 08 - Prediksi Keterlambatan Paket
Aplikasi Streamlit: Sistem Pendukung Keputusan Staf Operasional & Layanan Pelanggan.
"""

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Path berkas proyek
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.features import add_derived_features, FITUR_INPUT

MODEL_PATH = BASE / "artifacts" / "model.joblib"
META_PATH = BASE / "artifacts" / "metadata.json"

# Muat model dan metadata
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

model, metadata = load_resources()
THRESHOLD = metadata["threshold"]
VOLUME_Q75 = metadata["volume_q75_train"]

# Opsi data baku
KOTA_OPTIONS = [
    "Jakarta", "Bekasi", "Surabaya", "Medan", "Yogyakarta", "Denpasar",
    "Bogor", "Semarang", "Tasikmalaya", "Cirebon", "Malang", "Makassar",
    "Palembang", "Balikpapan", "Pontianak",
]
LAYANAN_OPTIONS = ["EXPRESS", "REG", "HEMAT"]
METODE_BAYAR_OPTIONS = ["COD", "non_COD"]
CUACA_OPTIONS = ["cerah", "berawan", "hujan", "hujan_lebat", "tidak_diketahui"]

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Prediksi Keterlambatan Paket - Hub Bandung",
    page_icon="truck",
    layout="wide",
)

# Kustomisasi CSS: Tema Logistik Minimalis Modern (Aksen Merah Kargo)
st.markdown("""
<style>
    /* Font dan warna dasar */
    .main {
        background-color: #f8fafc;
    }
    
    /* Header aplikasi */
    .cargo-header {
        background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%);
        padding: 24px 30px;
        border-radius: 8px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .cargo-header h1 {
        color: #ffffff !important;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .cargo-header p {
        color: #fecaca;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Kartu kontainer */
    .cargo-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .cargo-card-title {
        font-size: 16px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #f1f5f9;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Kotak Hasil Prediksi */
    .result-card-danger {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 6px solid #dc2626;
        border-radius: 6px;
        padding: 18px;
        margin-top: 10px;
    }
    .result-card-warning {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 6px solid #d97706;
        border-radius: 6px;
        padding: 18px;
        margin-top: 10px;
    }
    .result-card-success {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #16a34a;
        border-radius: 6px;
        padding: 18px;
        margin-top: 10px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 700;
        border-radius: 4px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-danger { background-color: #fee2e2; color: #991b1b; }
    .badge-warning { background-color: #fef3c7; color: #92400e; }
    .badge-success { background-color: #dcfce7; color: #166534; }

    /* Tombol utama */
    div.stButton > button:first-child {
        background-color: #b91c1c;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 15px;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #991b1b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header Banner
st.markdown("""
<div class="cargo-header">
    <h1>LOGISTIK HUB BANDUNG — PREDIKSI KETERLAMBATAN</h1>
    <p>Sistem Operasional Penjemputan Paket dan Mitigasi Keterlambatan Waktu Tiba</p>
</div>
""", unsafe_allow_html=True)

# Sidebar: Ringkasan Model & Biaya Bisnis
with st.sidebar:
    st.markdown("### Spesifikasi Model")
    st.markdown(f"""
    * **Algoritma:** {metadata.get('model', 'LogisticRegression')}
    * **Ambang Batas (*Threshold*):** `{THRESHOLD}`
    * **Metrik PR-AUC Validation:** `{metadata['metrik_validation']['pr_auc']}`
    * **Metrik PR-AUC Test:** `{metadata['metrik_test']['pr_auc']}`
    """)
    st.divider()
    st.markdown("### Ketentuan Biaya Klien")
    st.markdown("""
    * **Paket terlambat tanpa kabar (FN):** Rp50.000 per paket.
    * **Salah peringatan dini (FP):** Rp5.000 per paket.
    
    *Threshold* disetel pada `0,3446` untuk meminimalkan total kerugian operasional.
    """)
    st.divider()
    st.caption("Proyek Machine Learning · Kasus 08")

# Layout Form Masukan
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown("""
    <div class="cargo-card">
        <div class="cargo-card-title">1. Data Waktu & Rute Pengiriman</div>
    """, unsafe_allow_html=True)
    
    col_tgl, col_jam = st.columns(2)
    with col_tgl:
        tanggal_pickup = st.date_input("Tanggal Penjemputan", value=date.today())
    with col_jam:
        jam_pickup = st.number_input("Jam Penjemputan (7-22)", min_value=7, max_value=22, value=14, step=1)
        
    kota_tujuan = st.selectbox("Kota Tujuan", options=KOTA_OPTIONS, index=0)
    
    col_jarak, col_transit = st.columns(2)
    with col_jarak:
        jarak_km = st.number_input("Jarak Tempuh (km)", min_value=1.0, max_value=5000.0, value=150.0, step=10.0)
    with col_transit:
        jumlah_transit = st.number_input("Jumlah Hub Transit", min_value=0, max_value=10, value=1, step=1)
        
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="cargo-card">
        <div class="cargo-card-title">2. Spesifikasi Paket & Beban Hub</div>
    """, unsafe_allow_html=True)
    
    col_lay, col_sla = st.columns(2)
    with col_lay:
        layanan = st.selectbox("Layanan", options=LAYANAN_OPTIONS, index=1)
    with col_sla:
        janji_hari = st.number_input("Janji Waktu Tiba / SLA (Hari)", min_value=1, max_value=14, value=2, step=1)
        
    col_berat, col_bayar = st.columns(2)
    with col_berat:
        berat_kg = st.number_input("Berat Paket (kg)", min_value=0.01, max_value=100.0, value=1.5, step=0.1)
    with col_bayar:
        metode_bayar = st.selectbox("Metode Pembayaran", options=METODE_BAYAR_OPTIONS, index=0)
        
    col_cuaca, col_vol = st.columns(2)
    with col_cuaca:
        cuaca_saat_pickup = st.selectbox("Kondisi Cuaca Saat Pickup", options=CUACA_OPTIONS, index=0)
    with col_vol:
        volume_hub_hari = st.number_input("Volume Hub Hari Ini (Paket)", min_value=1, max_value=30000, value=5000, step=100)
        
    st.markdown("</div>", unsafe_allow_html=True)

# Tombol Eksekusi Prediksi
submit_button = st.button("Hitung Estimasi Keterlambatan", use_container_width=True)

if submit_button:
    # Validasi masukan pengguna
    error_list = []
    if jarak_km <= 0:
        error_list.append("Jarak tempuh harus bernilai lebih dari 0 km.")
    if berat_kg <= 0:
        error_list.append("Berat paket harus bernilai lebih dari 0 kg.")
    if volume_hub_hari <= 0:
        error_list.append("Volume paket di hub harus bernilai lebih dari 0.")
        
    if error_list:
        for err in error_list:
            st.error(f"Kesalahan Input: {err}")
    else:
        try:
            # Bentuk data inferensi
            raw_input = {
                "tanggal_pickup": str(tanggal_pickup),
                "jam_pickup": int(jam_pickup),
                "kota_tujuan": kota_tujuan,
                "jarak_km": float(jarak_km),
                "jumlah_transit": int(jumlah_transit),
                "layanan": layanan,
                "janji_hari": int(janji_hari),
                "berat_kg": float(berat_kg),
                "metode_bayar": metode_bayar,
                "cuaca_saat_pickup": cuaca_saat_pickup,
                "volume_hub_hari": int(volume_hub_hari),
            }
            
            df_input = pd.DataFrame([raw_input])
            df_input["tanggal_pickup"] = pd.to_datetime(df_input["tanggal_pickup"])
            df_features = add_derived_features(df_input, volume_q75=VOLUME_Q75)
            
            # Hitung probabilitas keterlambatan
            prob_terlambat = float(model.predict_proba(df_features[FITUR_INPUT])[:, 1][0])
            is_terlambat = prob_terlambat >= THRESHOLD
            
            # Tentukan kategori risiko dan rekomendasi
            if prob_terlambat >= 0.70:
                tingkat_risiko = "TINGGI"
                card_class = "result-card-danger"
                badge_class = "badge-danger"
                rekomendasi_teks = (
                    "Paket memiliki potensi keterlambatan sangat tinggi. Segera beri stiker prioritas pengiriman "
                    "pada hub transit dan kirimkan notifikasi awal ke pelanggan mengenai potensi perpanjangan estimasi tiba."
                )
            elif is_terlambat:
                tingkat_risiko = "SEDANG"
                card_class = "result-card-warning"
                badge_class = "badge-warning"
                rekomendasi_teks = (
                    "Paket berpotensi melewati janji tiba (melebihi ambang batas operasional). "
                    "Kirimkan informasi berkala kepada pelanggan dan monitor jadwal keberangkatan armada malam ini."
                )
            else:
                tingkat_risiko = "RENDAH"
                card_class = "result-card-success"
                badge_class = "badge-success"
                rekomendasi_teks = (
                    "Paket diproyeksikan tiba tepat waktu sesuai SLA. Paket dapat diproses menggunakan rute dan jadwal reguler."
                )
            
            # Render Kartu Hasil
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 18px; font-weight: 700; color: #1e293b;">
                        Hasil Estimasi: {'BERPOTENSI TERLAMBAT' if is_terlambat else 'TEPAT WAKTU'}
                    </span>
                    <span class="status-badge {badge_class}">Risiko {tingkat_risiko}</span>
                </div>
                <div style="font-size: 14px; color: #334155; margin-bottom: 12px;">
                    Probabilitas Keterlambatan: <strong>{prob_terlambat:.2%}</strong> (Ambang Batas Biaya: <strong>{THRESHOLD}</strong>)
                </div>
                <div style="font-size: 14px; color: #1e293b; line-height: 1.5;">
                    <strong>Rekomendasi Tindakan:</strong> {rekomendasi_teks}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detail Parameter Inferensi
            with st.expander("Rincian Parameter dan Fitur Turunan"):
                st.write({
                    "data_masukan": raw_input,
                    "fitur_turunan": {
                        "pickup_sore": int(df_features["pickup_sore"].iloc[0]),
                        "bulan_puncak": int(df_features["bulan_puncak"].iloc[0]),
                        "hari_senin": int(df_features["hari_senin"].iloc[0]),
                        "rasio_jarak_janji": round(float(df_features["rasio_jarak_janji"].iloc[0]), 2),
                        "transit_per_janji": round(float(df_features["transit_per_janji"].iloc[0]), 2),
                        "volume_tinggi": int(df_features["volume_tinggi"].iloc[0]),
                    },
                    "metrik_keputusan": {
                        "probabilitas": round(prob_terlambat, 4),
                        "ambang_batas": THRESHOLD,
                        "terlambat": bool(is_terlambat),
                    }
                })
                
        except Exception as e:
            st.error(f"Terjadi kendala saat memproses prediksi: {str(e)}")

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; font-size: 12px; color: #64748b;'>"
    "Proyek Akhir Supervised Learning — Kasus 08: Keterlambatan Paket | Hub Operasional Bandung"
    "</div>",
    unsafe_allow_html=True
)
