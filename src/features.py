"""
Kasus 08 – Prediksi Keterlambatan Paket
Modul fitur: transformasi konsisten untuk pelatihan & inferensi.

Semua transformasi ditulis sebagai fungsi murni agar bisa dipakai
di notebook pelatihan maupun di pipeline API/Streamlit tanpa duplikasi.
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Daftar fitur & konstanta
# ──────────────────────────────────────────────

COLS_KATEGORI = ["layanan", "kota_tujuan", "metode_bayar", "cuaca_saat_pickup"]

COLS_NUMERIK = [
    "jam_pickup", "jarak_km", "jumlah_transit", "janji_hari",
    "berat_kg", "volume_hub_hari",
]

# Kolom turunan yang akan dibuat oleh add_derived_features()
COLS_TURUNAN = [
    "pickup_sore",           # jam_pickup >= 17
    "bulan",                 # bulan dari tanggal_pickup (1-12)
    "hari_minggu",           # hari dalam minggu (0=Senin, 6=Minggu)
    "bulan_puncak",          # Maret atau Nov-Des
    "hari_senin",            # 1 jika Senin
    "rasio_jarak_janji",     # jarak_km / janji_hari
    "transit_per_janji",     # jumlah_transit / janji_hari
    "volume_tinggi",         # volume_hub_hari >= persentil 75 dari data train
]

# Semua fitur yang masuk ke model (setelah encoding)
# Urutan ini yang akan ditegakkan oleh Pipeline/ColumnTransformer
FITUR_INPUT = COLS_NUMERIK + COLS_TURUNAN + COLS_KATEGORI


# ──────────────────────────────────────────────
# Fungsi-fungsi transformasi
# ──────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame,
                         volume_q75: float | None = None) -> pd.DataFrame:
    """Tambahkan fitur-fitur turunan.

    Parameters
    ----------
    df : DataFrame
        Harus memiliki kolom: tanggal_pickup, jam_pickup, jarak_km,
        janji_hari, jumlah_transit, volume_hub_hari.
    volume_q75 : float | None
        Persentil 75 volume_hub_hari dari data train.
        Jika None, dihitung dari df itu sendiri (hanya untuk eksplorasi).

    Returns
    -------
    DataFrame dengan kolom turunan ditambahkan (in-place).
    """
    df = df.copy()

    # Pastikan tanggal_pickup bertipe datetime
    if not pd.api.types.is_datetime64_any_dtype(df["tanggal_pickup"]):
        df["tanggal_pickup"] = pd.to_datetime(df["tanggal_pickup"])

    df["pickup_sore"] = (df["jam_pickup"] >= 17).astype(int)
    df["bulan"] = df["tanggal_pickup"].dt.month
    df["hari_minggu"] = df["tanggal_pickup"].dt.dayofweek  # 0=Mon
    df["bulan_puncak"] = df["bulan"].isin([3, 11, 12]).astype(int)
    df["hari_senin"] = (df["hari_minggu"] == 0).astype(int)

    # Hindari pembagian nol
    df["rasio_jarak_janji"] = df["jarak_km"] / df["janji_hari"].replace(0, np.nan)
    df["transit_per_janji"] = df["jumlah_transit"] / df["janji_hari"].replace(0, np.nan)

    if volume_q75 is None:
        volume_q75 = df["volume_hub_hari"].quantile(0.75)
    df["volume_tinggi"] = (df["volume_hub_hari"] >= volume_q75).astype(int)

    return df


def prepare_single_input(data: dict, volume_q75: float) -> pd.DataFrame:
    """Ubah dict input tunggal (dari API/form) menjadi DataFrame 1-baris
    siap masuk pipeline model.

    Parameters
    ----------
    data : dict
        Berisi key sesuai kolom raw minimal:
        tanggal_pickup, jam_pickup, kota_tujuan, jarak_km,
        jumlah_transit, layanan, janji_hari, berat_kg,
        metode_bayar, cuaca_saat_pickup, volume_hub_hari.
    volume_q75 : float
        Persentil 75 volume_hub_hari dari data train.

    Returns
    -------
    DataFrame 1-baris dengan semua fitur turunan.
    """
    df = pd.DataFrame([data])
    df = add_derived_features(df, volume_q75=volume_q75)
    return df
