# 🛒 E-Commerce Churn Predictor

Aplikasi prediksi churn pelanggan e-commerce menggunakan **CatBoost**.

## 🚀 Deploy ke Streamlit Community Cloud

1. Push repo ini ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **New app** → pilih repo ini
4. Set **Main file path**: `app.py`
5. Klik **Deploy!**

## 💻 Jalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Struktur File

```
├── app.py                     # Aplikasi Streamlit (main)
├── E_Commerce_Dataset.xlsx    # Dataset training
├── requirements.txt           # Dependencies
└── README.md
```

## 📊 Performa Model

| Metrik | Nilai |
|--------|-------|
| Test F1 | 0.9794 |
| ROC-AUC | 1.0000 |
| CV F1 (5-fold) | 0.8829 ± 0.0138 |

## ✨ Fitur

- **Prediksi Manual** — Input 1 pelanggan, dapat hasil + rekomendasi
- **Prediksi Batch** — Upload CSV, download hasil prediksi
