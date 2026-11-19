# Prediksi Keterlambatan Paket Tim Ekspedisi Bandung

**Kasus:** 08 - Keterlambatan Paket · **Aplikasi:** https://ekspedisibandoeng.streamlit.app/ · **Video:** https://drive.google.com/drive/folders/14tV_cKBTyRyPy-v6wkQK1cYx-Ui-VA2b?usp=sharing
| Anggota | NIM | Bagian yang Dikerjakan |
|---|---|---|
| Fauzan Rizki Pratama | 2404389 | Eksplorasi data (EDA), pembersihan data, rekayasa fitur, pelatihan baseline dan model final, evaluasi metrik |
| Faiz Bisma Alfarid | 2407903 | Pengembangan backend API (FastAPI), antarmuka web (Streamlit), validasi input, pengujian sistem, dan deployment |

---

## 1. Masalah dan Pengguna

* **Pengguna Aplikasi:** Staf operasional hub pengiriman dan staf layanan pelanggan (*customer service*) ekspedisi di Bandung.
* **Keputusan yang Dibantu:** Menentukan paket mana yang berisiko tiba melewati estimasi janji tiba (SLA) sejak paket pertama kali dijemput (*pickup*).
* **Dampak Bisnis:** Paket terlambat tanpa pemberitahuan menimbulkan biaya keluhan dan kompensasi senilai **Rp50.000** (False Negative), sedangkan peringatan dini yang keliru hanya menimbulkan biaya konfirmasi layanan senilai **Rp5.000** (False Positive).

---

## 2. Data dan Pembersihan

Data mentah berisi **40.500 baris** transaksi paket (Januari 2025 – Juni 2026). Setelah pembersihan, diperoleh **40.000 baris bersih**:

| Masalah yang Ditemukan | Jumlah Baris | Cara Menangani | Alasan |
|---|---|---|---|
| Baris duplikat penuh | 500 | `drop_duplicates` | Satu nomor resi hanya boleh tercatat satu kali agar data tidak terhitung ganda |
| Penulisan layanan tidak seragam (`EXP`, `Express`, `hemat`, `reg`) | 800 | Standarisasi ke 3 nilai baku: `EXPRESS`, `REG`, `HEMAT` | Perbedaan format huruf kapital dan singkatan dianggap sebagai kategori berbeda oleh model |
| Penulisan kota tujuan tidak baku (28 varian ejaan) | 1.132 | Pemetaan terarah ke 15 kota baku (misal: "Bali" dipetakan ke "Denpasar") | Varian singkatan dan wilayah dirapikan ke hub kota tujuan yang valid |
| Nilai anomali `berat_kg == -1` | 80 | Diubah menjadi NaN, lalu diimputasi median per layanan | Nilai negatif merupakan kode entri rusak/kosong |
| Nilai `berat_kg >= 100` | 1.197 | Dibagi 1.000 menjadi satuan kilogram | Data dientri dalam satuan gram sehingga perlu disetarakan ke kilogram |
| Nilai kosong pada `jarak_km` | 800 | Diimputasi median jarak per kota tujuan | Jarak fisik berkaitan erat dengan kota tujuan |
| Nilai kosong pada `cuaca_saat_pickup` | 400 | Diisi kategori `tidak_diketahui` | Sistem produksi harus mampu memproses paket meskipun data cuaca sensor tidak tersedia |

Catatan: Fluktuasi nilai `volume_hub_hari` pada tanggal yang sama dan ~285 baris kombinasi layanan vs janji hari langka dicatat sebagai karakteristik data lapangan dan dibiarkan apa adanya.

---

## 3. Pembagian Data dan Fitur

Data dibagi secara kronologis (berdasarkan waktu) untuk mencegah kebocoran informasi masa depan (*data leakage*):
* **Train:** Januari 2025 – Desember 2025 (26.633 baris, rasio kelas terlambat ~17,8%)
* **Validation:** Januari 2026 – Maret 2026 (6.623 baris, rasio kelas terlambat ~21,9%)
* **Test:** April 2026 – Juni 2026 (6.744 baris, rasio kelas terlambat ~13,9%)

### Fitur yang Digunakan:
1. **Fitur Asli:** `jam_pickup`, `jarak_km`, `jumlah_transit`, `janji_hari`, `berat_kg`, `volume_hub_hari`, `layanan`, `kota_tujuan`, `metode_bayar`, `cuaca_saat_pickup`.
2. **Fitur Turunan (`src/features.py`):**
   * `pickup_sore`: penjemputan jam 17 ke atas berisiko tertinggal jadwal armada malam.
   * `bulan_puncak`: indikator lonjakan beban pengiriman (Maret serta November–Desember).
   * `hari_senin`: penumpukan paket di awal pekan.
   * `rasio_jarak_janji`: rasio jarak tempuh per hari janji SLA.
   * `transit_per_janji`: kepadatan transit perpindahan hub per hari SLA.
   * `volume_tinggi`: volume paket hub melebihi persentil 75 data latih.

---

## 4. Hasil Pemodelan

Metrik utama yang digunakan adalah **PR-AUC** (*Precision-Recall Area Under Curve*) karena target tidak seimbang (~18% positif).

| Model | PR-AUC Validation | PR-AUC Test | Status |
|---|---|---|---|
| **Baseline: Logistic Regression** (class_weight='balanced') | **0,6587** | **0,3737** | **Terpilih (Model Final)** |
| **Kandidat: HistGradientBoostingClassifier** | 0,6531 | - | Tidak dipilih (performa di bawah baseline) |

* **Alasan Pemilihan Model:** Logistic Regression menghasilkan PR-AUC validasi sedikit lebih tinggi daripada HistGradientBoosting. Model linier terbukti lebih stabil, cepat dalam proses inferensi, dan interpretasi bobot koefisiennya mudah dipahami oleh operasional.
* **Ambang Batas (*Threshold*) Optimal:** Ditetapkan pada **0,3446** berdasarkan kurva minimalisasi total biaya bisnis pada data validasi.
  * Biaya validasi minimum: **Rp19.580.000**
  * Evaluasi akhir data test (dilakukan 1 kali): **PR-AUC 0,3737** dengan total biaya risiko **Rp21.945.000**.

---

## 5. Cara Menjalankan Proyek

### Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### Eksekusi Notebook
1. `notebooks/01_eda_pembersihan.ipynb`: Membaca `data/raw.csv`, melakukan pembersihan, analisis visual, dan menyimpan `data/clean.csv`.
2. `notebooks/02_modeling_evaluation.ipynb`: Melatih baseline dan model kandidat, mencari threshold biaya optimal, evaluasi test set, dan mengekspor `artifacts/model.joblib` serta `artifacts/metadata.json`.

### Menjalankan API (FastAPI)
```bash
uvicorn app.api:app --reload
```
Akses dokumentasi interaktif Swagger di: `http://127.0.0.1:8000/docs`

### Menjalankan Aplikasi Web (Streamlit)
```bash
streamlit run app/streamlit_app.py
```
Akses antarmuka web di: `http://localhost:8501`

---

## 6. Struktur Repositori

```text
├── .streamlit/
│   └── config.toml               # Konfigurasi server Streamlit
├── app/
│   ├── __init__.py
│   ├── api.py                    # Layanan REST API (FastAPI)
│   └── streamlit_app.py          # Antarmuka web pengguna (Streamlit)
├── artifacts/
│   ├── confusion_matrix_test.png # Grafik matriks kebingungan data uji
│   ├── feature_importance.png    # Grafik bobot fitur terpenting
│   ├── grafik_biaya_threshold.png# Kurva optimasi threshold vs biaya bisnis
│   ├── grafik_terlambat_per_bulan.png
│   ├── grafik_terlambat_per_jam.png
│   ├── grafik_terlambat_per_transit.png
│   ├── kurva_pr_validation.png   # Kurva Precision-Recall validasi
│   ├── metadata.json             # Catatan versi, threshold, metrik, dan fitur
│   └── model.joblib              # Pipeline model terlatih
├── data/
│   ├── clean.csv                 # Data 40.000 baris siap latih
│   ├── raw.csv                   # Data transaksi mentah
│   └── tabel_masalah_data.csv    # Dokumentasi penanganan kualitas data
├── notebooks/
│   ├── 01_eda_pembersihan.ipynb  # Notebook pembersihan dan visualisasi EDA
│   └── 02_modeling_evaluation.ipynb # Notebook pelatihan dan evaluasi model
├── src/
│   ├── __init__.py
│   └── features.py               # Modul transformasi fitur tanpa data leakage
├── .gitignore
├── brief.md                      # Panduan kasus klien
├── requirements.txt              # Daftar dependensi pustaka
└── README.md                     # Dokumentasi utama repositori
```

---

## 7. Keterbatasan dan Mitigasi

1. **Penurunan Rasio Keterlambatan pada Periode Uji:** Proporsi kelas terlambat pada data test menurun menjadi 13,9% (dibandingkan validation 21,9%), yang memengaruhi nilai absolut PR-AUC di periode pengujian. Model perlu dikalibrasi ulang secara berkala tiap kuartal.
2. **Ketergantungan pada Estimasi Awal:** Fitur cuaca saat penjemputan bersifat lokal (Bandung) dan belum mencakup dinamika cuaca ekstrem di sepanjang perjalanan jalur darat atau hub transit kota lain.

