"""
E-Commerce Churn Prediction
Deploy: Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Churn Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 12px;
        color: white; text-align: center; margin-bottom: 2rem;
    }
    .churn-box {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        padding: 1.5rem; border-radius: 12px;
        color: white; text-align: center;
    }
    .nochurn-box {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 1.5rem; border-radius: 12px;
        color: white; text-align: center;
    }
    .info-box {
        background: #f0f4ff; padding: 1rem;
        border-radius: 8px; border: 1px solid #d0d9ff; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TRAIN MODEL (cached — hanya sekali per session)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Melatih model... (~30 detik pertama kali)")
def train_and_cache_model():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import f1_score, roc_auc_score
    from catboost import CatBoostClassifier

    # Load data dari repo
    df = pd.read_excel("E_Commerce_Dataset.xlsx", sheet_name="E Comm")

    # Standardisasi nilai kategorikal
    df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace(
        {'COD': 'Cash on Delivery', 'CC': 'Credit Card'})
    df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace(
        {'Phone': 'Mobile Phone'})

    FEATURES = [
        'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
        'PreferredPaymentMode', 'Gender', 'HourSpendOnApp',
        'NumberOfDeviceRegistered', 'PreferedOrderCat', 'SatisfactionScore',
        'MaritalStatus', 'NumberOfAddress', 'Complain',
        'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
        'DaySinceLastOrder', 'CashbackAmount'
    ]
    NUM_COLS = [
        'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
        'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
        'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed',
        'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
    ]
    CAT_COLS = [
        'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
        'PreferedOrderCat', 'MaritalStatus'
    ]

    X = df[FEATURES]
    y = df['Churn']

    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
        ]), NUM_COLS),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value',
                                       unknown_value=-1)),
        ]), CAT_COLS),
    ])

    # Split untuk evaluasi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', CatBoostClassifier(
            iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=3,
            auto_class_weights='Balanced', eval_metric='F1',
            od_type='Iter', od_wait=50, random_state=42, verbose=0,
        ))
    ])

    # CV F1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipeline, X_train, y_train,
                            cv=cv, scoring='f1', n_jobs=-1)

    # OOF threshold optimization
    thresholds  = np.arange(0.20, 0.71, 0.01)
    oof_proba   = np.zeros(len(X_train))
    oof_true    = np.zeros(len(X_train))
    for _, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        clone = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', CatBoostClassifier(
                iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=3,
                auto_class_weights='Balanced', eval_metric='F1',
                od_type='Iter', od_wait=50, random_state=42, verbose=0,
            ))
        ])
        clone.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_proba[val_idx] = clone.predict_proba(
            X_train.iloc[val_idx])[:, 1]
        oof_true[val_idx] = y_train.iloc[val_idx]

    f1_per_thresh = [
        f1_score(oof_true, (oof_proba >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_threshold = round(thresholds[np.argmax(f1_per_thresh)], 2)

    # Retrain on full data untuk produksi
    pipeline.fit(X, y)

    # Evaluasi test set
    y_proba  = pipeline.predict_proba(X_test)[:, 1]
    y_pred   = (y_proba >= best_threshold).astype(int)
    test_f1  = round(f1_score(y_test, y_pred), 4)
    test_auc = round(roc_auc_score(y_test, y_proba), 4)

    return {
        'model':      pipeline,
        'threshold':  best_threshold,
        'features':   FEATURES,
        'cv_f1_mean': round(cv_f1.mean(), 4),
        'cv_f1_std':  round(cv_f1.std(),  4),
        'test_f1':    test_f1,
        'test_auc':   test_auc,
    }


# ─────────────────────────────────────────────
# LOAD / TRAIN MODEL
# ─────────────────────────────────────────────
try:
    bundle = train_and_cache_model()
    model     = bundle['model']
    threshold = bundle['threshold']
    features  = bundle['features']
except Exception as e:
    st.error(f"❌ Gagal melatih model: {e}")
    st.info("Pastikan file `E_Commerce_Dataset.xlsx` ada di folder yang sama dengan `app.py`")
    st.stop()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛒 E-Commerce Churn Predictor</h1>
    <p style="font-size:1.1rem; opacity:0.9;">
        Prediksi pelanggan yang berpotensi berhenti berbelanja · CatBoost Model
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🤖 Model",      "CatBoost")
c2.metric("🎯 Test F1",    bundle['test_f1'])
c3.metric("📈 ROC-AUC",    bundle['test_auc'])
c4.metric("📊 CV F1",      f"{bundle['cv_f1_mean']} ± {bundle['cv_f1_std']}")
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Navigasi")
mode = st.sidebar.radio(
    "Pilih Mode Prediksi:",
    ["🧑 Prediksi Manual (1 Pelanggan)", "📂 Prediksi Batch (Upload CSV)"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**ℹ️ Info Model**
- **Algoritma :** CatBoost
- **Threshold  :** `{threshold}`
- **Fitur      :** {len(features)} variabel
- **Target     :** Churn (1=Ya, 0=Tidak)
""")


# ═══════════════════════════════════════════════
# MODE 1: PREDIKSI MANUAL
# ═══════════════════════════════════════════════
if mode == "🧑 Prediksi Manual (1 Pelanggan)":
    st.subheader("🧑 Input Data Pelanggan")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Informasi Dasar**")
        tenure         = st.number_input("Tenure (bulan)", 0, 60, 6)
        city_tier      = st.selectbox("City Tier", [1, 2, 3],
                                       help="1=Kota Besar, 3=Kota Kecil")
        gender         = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status",
                                       ["Single", "Married", "Divorced"])
        num_address    = st.number_input("Jumlah Alamat", 1, 20, 2)

    with col2:
        st.markdown("**📱 Aktivitas Digital**")
        preferred_login   = st.selectbox("Login Device",
                                          ["Mobile Phone", "Computer"])
        hour_spend        = st.slider("Jam/Hari di App", 0, 5, 2)
        num_device        = st.slider("Jumlah Perangkat", 1, 6, 3)
        preferred_payment = st.selectbox("Metode Pembayaran",
                                          ["Debit Card", "Credit Card",
                                           "E wallet", "Cash on Delivery", "UPI"])
        preferred_order   = st.selectbox("Kategori Order Favorit",
                                          ["Laptop & Accessory", "Mobile",
                                           "Mobile Phone", "Fashion",
                                           "Grocery", "Others"])

    with col3:
        st.markdown("**🛍️ Aktivitas Belanja**")
        warehouse_to_home = st.number_input("Jarak Gudang→Rumah (km)", 0, 130, 15)
        satisfaction      = st.slider("Satisfaction Score (1–5)", 1, 5, 3)
        complain          = st.selectbox("Ada Komplain?", ["Tidak", "Ya"])
        order_hike        = st.number_input("Kenaikan Order dari Tahun Lalu (%)",
                                             11, 26, 15)
        coupon_used       = st.number_input("Kupon Dipakai", 0, 16, 1)
        order_count       = st.number_input("Jumlah Order Bulan Ini", 1, 16, 2)
        day_since_last    = st.number_input("Hari Sejak Order Terakhir", 0, 46, 3)
        cashback          = st.number_input("Cashback Rata-Rata", 0, 325, 160)

    st.markdown("---")

    if st.button("🔮 Prediksi Sekarang", use_container_width=True):
        input_df = pd.DataFrame([{
            'Tenure':                      tenure,
            'PreferredLoginDevice':        preferred_login,
            'CityTier':                    city_tier,
            'WarehouseToHome':             warehouse_to_home,
            'PreferredPaymentMode':        preferred_payment,
            'Gender':                      gender,
            'HourSpendOnApp':              hour_spend,
            'NumberOfDeviceRegistered':    num_device,
            'PreferedOrderCat':            preferred_order,
            'SatisfactionScore':           satisfaction,
            'MaritalStatus':               marital_status,
            'NumberOfAddress':             num_address,
            'Complain':                    1 if complain == "Ya" else 0,
            'OrderAmountHikeFromlastYear': order_hike,
            'CouponUsed':                  coupon_used,
            'OrderCount':                  order_count,
            'DaySinceLastOrder':           day_since_last,
            'CashbackAmount':              float(cashback),
        }])

        proba    = model.predict_proba(input_df)[0][1]
        is_churn = proba >= threshold

        r1, r2 = st.columns([1, 2])
        with r1:
            if is_churn:
                st.markdown(f"""
                <div class="churn-box">
                    <h2>⚠️ CHURN</h2>
                    <h3>Pelanggan Berisiko Pergi</h3>
                    <h1>{proba*100:.1f}%</h1>
                    <p>Probabilitas Churn</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="nochurn-box">
                    <h2>✅ AMAN</h2>
                    <h3>Pelanggan Kemungkinan Bertahan</h3>
                    <h1>{proba*100:.1f}%</h1>
                    <p>Probabilitas Churn</p>
                </div>""", unsafe_allow_html=True)

        with r2:
            st.markdown("**📊 Detail Probabilitas**")
            st.progress(float(proba))
            st.write(f"Churn Probability : **{proba*100:.1f}%**")
            st.write(f"Threshold         : **{threshold}**")
            st.markdown("---")
            if is_churn:
                st.markdown("""
**💡 Rekomendasi:**
- 🎁 Berikan voucher / diskon eksklusif
- 📞 Follow-up personal dari tim retention
- 🔔 Kirim notifikasi promosi yang dipersonalisasi
- ⚡ Prioritaskan resolusi jika ada komplain""")
            else:
                st.markdown("""
**💡 Rekomendasi:**
- 🌟 Pertahankan kualitas layanan saat ini
- 💎 Tawarkan program loyalitas / membership
- 📈 Upsell produk premium yang relevan""")


# ═══════════════════════════════════════════════
# MODE 2: BATCH PREDICTION
# ═══════════════════════════════════════════════
else:
    st.subheader("📂 Prediksi Batch — Upload File CSV")

    st.markdown("""
<div class="info-box">
📌 <b>Format CSV yang diperlukan — kolom wajib ada:</b><br>
<code>Tenure, PreferredLoginDevice, CityTier, WarehouseToHome,
PreferredPaymentMode, Gender, HourSpendOnApp, NumberOfDeviceRegistered,
PreferedOrderCat, SatisfactionScore, MaritalStatus, NumberOfAddress,
Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount,
DaySinceLastOrder, CashbackAmount</code>
</div>""", unsafe_allow_html=True)

    template = pd.DataFrame([{
        'Tenure': 6, 'PreferredLoginDevice': 'Mobile Phone',
        'CityTier': 1, 'WarehouseToHome': 15,
        'PreferredPaymentMode': 'Debit Card', 'Gender': 'Male',
        'HourSpendOnApp': 3, 'NumberOfDeviceRegistered': 3,
        'PreferedOrderCat': 'Mobile', 'SatisfactionScore': 3,
        'MaritalStatus': 'Single', 'NumberOfAddress': 2,
        'Complain': 0, 'OrderAmountHikeFromlastYear': 15,
        'CouponUsed': 1, 'OrderCount': 2,
        'DaySinceLastOrder': 3, 'CashbackAmount': 160.0,
    }])
    st.download_button("⬇️ Download Template CSV",
                       data=template.to_csv(index=False),
                       file_name="template_churn.csv",
                       mime="text/csv")

    uploaded = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)

            # Standardisasi
            df_up['PreferredPaymentMode'] = df_up['PreferredPaymentMode'].replace(
                {'COD': 'Cash on Delivery', 'CC': 'Credit Card'})
            df_up['PreferredLoginDevice'] = df_up['PreferredLoginDevice'].replace(
                {'Phone': 'Mobile Phone'})

            st.write(f"**{df_up.shape[0]} baris** akan diprediksi")
            st.dataframe(df_up.head(), use_container_width=True)

            if st.button("🔮 Prediksi Semua", use_container_width=True):
                X_b     = df_up[features]
                prob_b  = model.predict_proba(X_b)[:, 1]
                pred_b  = (prob_b >= threshold).astype(int)

                result              = df_up.copy()
                result['Prob_Churn'] = np.round(prob_b, 4)
                result['Prediksi']   = pred_b
                result['Status']     = result['Prediksi'].map(
                    {1: '⚠️ CHURN', 0: '✅ AMAN'})

                n_churn = pred_b.sum()
                n_total = len(pred_b)
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Pelanggan", n_total)
                m2.metric("⚠️ Churn",  n_churn,
                           f"{n_churn/n_total*100:.1f}%", delta_color="inverse")
                m3.metric("✅ Aman", n_total - n_churn,
                           f"{(n_total-n_churn)/n_total*100:.1f}%")

                st.dataframe(result, use_container_width=True)
                st.download_button("⬇️ Download Hasil",
                                   data=result.to_csv(index=False),
                                   file_name="hasil_prediksi_churn.csv",
                                   mime="text/csv")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# FOOTER
st.markdown("---")
st.markdown("""<p style="text-align:center; color:#888; font-size:0.85rem;">
🛒 E-Commerce Churn Predictor · CatBoost · Streamlit</p>""",
unsafe_allow_html=True)
