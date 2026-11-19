"""
Kasus 08 – Prediksi Keterlambatan Paket
API FastAPI: menerima data satu paket, mengembalikan probabilitas & rekomendasi.

Jalankan:
    uvicorn app.api:app --reload
"""

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# ── Path artefak ───────────────────────────────
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.features import add_derived_features, FITUR_INPUT

MODEL_PATH = BASE / "artifacts" / "model.joblib"
META_PATH = BASE / "artifacts" / "metadata.json"

# ── Muat model & metadata ─────────────────────
model = joblib.load(MODEL_PATH)

with open(META_PATH, "r") as f:
    metadata = json.load(f)

THRESHOLD = metadata["threshold"]
VOLUME_Q75 = metadata["volume_q75_train"]

# ── Konstanta validasi ─────────────────────────
KOTA_VALID = [
    "Jakarta", "Bekasi", "Surabaya", "Medan", "Yogyakarta", "Denpasar",
    "Bogor", "Semarang", "Tasikmalaya", "Cirebon", "Malang", "Makassar",
    "Palembang", "Balikpapan", "Pontianak",
]
LAYANAN_VALID = ["EXPRESS", "REG", "HEMAT"]
METODE_BAYAR_VALID = ["COD", "non_COD"]
CUACA_VALID = ["cerah", "berawan", "hujan", "hujan_lebat", "tidak_diketahui"]

# ── Skema input ────────────────────────────────

class InputPaket(BaseModel):
    tanggal_pickup: date
    jam_pickup: int
    kota_tujuan: str
    jarak_km: float
    jumlah_transit: int
    layanan: str
    janji_hari: int
    berat_kg: float
    metode_bayar: str
    cuaca_saat_pickup: Optional[str] = "tidak_diketahui"
    volume_hub_hari: int

    @field_validator("jam_pickup")
    @classmethod
    def validasi_jam(cls, v):
        if not (7 <= v <= 22):
            raise ValueError("jam_pickup harus antara 7 dan 22.")
        return v

    @field_validator("jarak_km")
    @classmethod
    def validasi_jarak(cls, v):
        if v <= 0:
            raise ValueError("jarak_km harus lebih besar dari 0.")
        if v > 5000:
            raise ValueError("jarak_km terlalu besar (maksimum 5.000 km).")
        return v

    @field_validator("jumlah_transit")
    @classmethod
    def validasi_transit(cls, v):
        if v < 0:
            raise ValueError("jumlah_transit tidak boleh negatif.")
        if v > 10:
            raise ValueError("jumlah_transit terlalu besar (maksimum 10).")
        return v

    @field_validator("layanan")
    @classmethod
    def validasi_layanan(cls, v):
        v = v.strip().upper()
        if v not in LAYANAN_VALID:
            raise ValueError(
                f"Layanan '{v}' tidak dikenali. Pilihan: {', '.join(LAYANAN_VALID)}."
            )
        return v

    @field_validator("janji_hari")
    @classmethod
    def validasi_janji(cls, v):
        if v <= 0:
            raise ValueError("janji_hari harus lebih besar dari 0.")
        if v > 14:
            raise ValueError("janji_hari terlalu besar (maksimum 14 hari).")
        return v

    @field_validator("berat_kg")
    @classmethod
    def validasi_berat(cls, v):
        if v <= 0:
            raise ValueError("berat_kg harus lebih besar dari 0.")
        if v > 100:
            raise ValueError("berat_kg terlalu besar (maksimum 100 kg).")
        return v

    @field_validator("kota_tujuan")
    @classmethod
    def validasi_kota(cls, v):
        v = v.strip().title()
        if v not in KOTA_VALID:
            raise ValueError(
                f"Kota '{v}' tidak dikenali. Pilihan: {', '.join(KOTA_VALID)}."
            )
        return v

    @field_validator("metode_bayar")
    @classmethod
    def validasi_bayar(cls, v):
        v = v.strip()
        if v not in METODE_BAYAR_VALID:
            raise ValueError(
                f"Metode bayar '{v}' tidak dikenali. Pilihan: {', '.join(METODE_BAYAR_VALID)}."
            )
        return v

    @field_validator("cuaca_saat_pickup")
    @classmethod
    def validasi_cuaca(cls, v):
        if v is None:
            return "tidak_diketahui"
        v = v.strip().lower()
        if v not in CUACA_VALID:
            raise ValueError(
                f"Cuaca '{v}' tidak dikenali. Pilihan: {', '.join(CUACA_VALID)}."
            )
        return v

    @field_validator("volume_hub_hari")
    @classmethod
    def validasi_volume(cls, v):
        if v <= 0:
            raise ValueError("volume_hub_hari harus lebih besar dari 0.")
        return v


# ── Aplikasi FastAPI ───────────────────────────

app = FastAPI(
    title="Prediksi Keterlambatan Paket",
    description="API untuk memprediksi apakah paket akan terlambat berdasarkan data saat pickup.",
    version=metadata.get("versi", "1.0.0"),
)


def buat_rekomendasi(probabilitas: float, threshold: float) -> dict:
    """Buat rekomendasi berdasarkan probabilitas dan threshold."""
    if probabilitas >= threshold:
        if probabilitas >= 0.7:
            level = "TINGGI"
            aksi = "Prioritaskan pengiriman dan segera kabari pelanggan bahwa paket kemungkinan besar terlambat."
        else:
            level = "SEDANG"
            aksi = "Kabari pelanggan bahwa paket berpotensi terlambat agar mereka dapat mengantisipasi."
    else:
        level = "RENDAH"
        aksi = "Paket diperkirakan tiba tepat waktu. Tidak perlu tindakan khusus."

    return {
        "risiko_keterlambatan": level,
        "aksi_disarankan": aksi,
    }


@app.get("/", tags=["Info"])
def root():
    """Endpoint informasi API."""
    return {
        "nama": "API Prediksi Keterlambatan Paket",
        "versi": metadata.get("versi", "1.0.0"),
        "model": metadata.get("model", "unknown"),
        "threshold": THRESHOLD,
    }


@app.get("/health", tags=["Info"])
def health():
    """Health check."""
    return {"status": "sehat", "model_dimuat": model is not None}


@app.post("/prediksi", tags=["Prediksi"])
def prediksi(input_paket: InputPaket):
    """Prediksi keterlambatan satu paket.

    Mengembalikan probabilitas terlambat, keputusan (terlambat/tepat waktu),
    dan rekomendasi aksi.
    """
    try:
        data = input_paket.model_dump()
        data["tanggal_pickup"] = str(data["tanggal_pickup"])

        # Buat DataFrame dan tambahkan fitur turunan
        df = pd.DataFrame([data])
        df["tanggal_pickup"] = pd.to_datetime(df["tanggal_pickup"])
        df = add_derived_features(df, volume_q75=VOLUME_Q75)

        # Prediksi
        X = df[FITUR_INPUT]
        proba = float(model.predict_proba(X)[:, 1][0])
        terlambat = int(proba >= THRESHOLD)

        rekomendasi = buat_rekomendasi(proba, THRESHOLD)

        return {
            "probabilitas_terlambat": round(proba, 4),
            "prediksi_terlambat": bool(terlambat),
            "threshold_dipakai": THRESHOLD,
            **rekomendasi,
            "input_diterima": {
                "kota_tujuan": data["kota_tujuan"],
                "layanan": data["layanan"],
                "jarak_km": data["jarak_km"],
                "berat_kg": data["berat_kg"],
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat memproses prediksi: {str(e)}",
        )


@app.get("/metadata", tags=["Info"])
def get_metadata():
    """Mengembalikan metadata model."""
    return metadata
