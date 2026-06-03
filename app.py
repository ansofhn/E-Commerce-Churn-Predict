import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="E-Commerce Churn Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        padding:2rem;border-radius:12px;color:white;
        text-align:center;margin-bottom:2rem;
    }
    .churn-box {
        background:linear-gradient(135deg,#ff416c,#ff4b2b);
        padding:1.5rem;border-radius:12px;color:white;text-align:center;
    }
    .nochurn-box {
        background:linear-gradient(135deg,#11998e,#38ef7d);
        padding:1.5rem;border-radius:12px;color:white;text-align:center;
    }
    .info-box {
        background:#f0f4ff;padding:1rem;border-radius:8px;
        border:1px solid #d0d9ff;margin-bottom:1rem;
    }
    .link-box {
        background:#1a1d2e;padding:0.8rem 1rem;border-radius:8px;
        border:1px solid #667eea;font-family:monospace;
        font-size:0.82rem;color:#e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL — dengan fallback retrain jika pkl gagal
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Memuat model...")
def load_model():
    pkl_path = os.path.join(os.path.dirname(__file__), "model.pkl")

    # Coba load pkl
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                bundle = pickle.load(f)
            return bundle, "pkl"
        except Exception as e:
            st.warning(f"⚠️ Gagal load model.pkl ({e}), akan retrain otomatis...")

    # Fallback: retrain on-the-fly
    return _retrain(), "retrain"


def _retrain():
    """Retrain CatBoost dari dataset — dipakai jika pkl gagal load."""
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import f1_score, roc_auc_score
    from catboost import CatBoostClassifier

    df = pd.read_excel("E_Commerce_Dataset.xlsx", sheet_name="E Comm")
    df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace(
        {'COD':'Cash on Delivery','CC':'Credit Card'})
    df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace(
        {'Phone':'Mobile Phone'})

    FEATURES = [
        'Tenure','PreferredLoginDevice','CityTier','WarehouseToHome',
        'PreferredPaymentMode','Gender','HourSpendOnApp',
        'NumberOfDeviceRegistered','PreferedOrderCat','SatisfactionScore',
        'MaritalStatus','NumberOfAddress','Complain',
        'OrderAmountHikeFromlastYear','CouponUsed','OrderCount',
        'DaySinceLastOrder','CashbackAmount'
    ]
    NUM_COLS = ['Tenure','CityTier','WarehouseToHome','HourSpendOnApp',
                'NumberOfDeviceRegistered','SatisfactionScore','NumberOfAddress',
                'Complain','OrderAmountHikeFromlastYear','CouponUsed',
                'OrderCount','DaySinceLastOrder','CashbackAmount']
    CAT_COLS = ['PreferredLoginDevice','PreferredPaymentMode','Gender',
                'PreferedOrderCat','MaritalStatus']

    X, y = df[FEATURES], df['Churn']

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp',SimpleImputer(strategy='median')),
                          ('sc', StandardScaler())]), NUM_COLS),
        ('cat', Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                          ('enc',OrdinalEncoder(handle_unknown='use_encoded_value',
                                               unknown_value=-1))]), CAT_COLS),
    ])
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', CatBoostClassifier(
            iterations=600,depth=6,learning_rate=0.05,l2_leaf_reg=3,
            auto_class_weights='Balanced',eval_metric='F1',
            od_type='Iter',od_wait=50,random_state=42,verbose=0,
        ))
    ])

    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    cv_f1 = cross_val_score(pipeline,X,y,cv=cv,scoring='f1',n_jobs=-1)

    # OOF threshold
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    oof_p = np.zeros(len(X_tr)); oof_t = np.zeros(len(X_tr))
    from sklearn.base import clone as sklearn_clone
    for _,(tri,vli) in enumerate(cv.split(X_tr,y_tr)):
        c = sklearn_clone(pipeline)
        c.fit(X_tr.iloc[tri],y_tr.iloc[tri])
        oof_p[vli] = c.predict_proba(X_tr.iloc[vli])[:,1]
        oof_t[vli] = y_tr.iloc[vli]

    thresholds = np.arange(0.20,0.71,0.01)
    f1s = [f1_score(oof_t,(oof_p>=t).astype(int),zero_division=0) for t in thresholds]
    best_thresh = round(thresholds[np.argmax(f1s)],2)

    pipeline.fit(X,y)
    yp = pipeline.predict_proba(X_te)[:,1]
    ypred = (yp>=best_thresh).astype(int)

    return {
        'model':      pipeline,
        'threshold':  best_thresh,
        'features':   FEATURES,
        'cv_f1_mean': round(cv_f1.mean(),4),
        'cv_f1_std':  round(cv_f1.std(),4),
        'test_f1':    round(f1_score(y_te,ypred),4),
        'test_auc':   round(roc_auc_score(y_te,yp),4),
        'model_name': 'CatBoost',
    }


# ── Load ──
try:
    bundle, load_method = load_model()
    model     = bundle["model"]
    threshold = bundle["threshold"]
    features  = bundle["features"]
except Exception as e:
    st.error(f"❌ Fatal error: {e}")
    st.stop()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛒 E-Commerce Churn Predictor</h1>
    <p style="font-size:1.1rem;opacity:0.9;">
        Prediksi pelanggan yang berpotensi berhenti berbelanja · CatBoost Model
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🤖 Model",   bundle.get("model_name","CatBoost"))
c2.metric("🎯 Test F1", bundle.get("test_f1","-"))
c3.metric("📈 ROC-AUC", bundle.get("test_auc","-"))
c4.metric("📊 CV F1",   f"{bundle.get('cv_f1_mean','-')} ± {bundle.get('cv_f1_std','-')}")

if load_method == "retrain":
    st.info("ℹ️ Model dilatih ulang dari dataset (model.pkl tidak kompatibel dengan Python versi ini)")

st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Navigasi")
mode = st.sidebar.radio(
    "Pilih Mode:",
    ["🧑 Prediksi Manual (1 Pelanggan)", "📂 Prediksi Batch (Upload CSV)"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**ℹ️ Info Model**
- **Algoritma  :** CatBoost
- **Threshold  :** `{threshold}`
- **Fitur      :** {len(features)} variabel
- **Load via   :** `{load_method}`
""")

GITHUB_REPO = "https://github.com/ansofhn/E-Commerce-Churn-Predict" 
st.sidebar.markdown("---")
st.sidebar.markdown("**🔗 Link Repository**")
st.sidebar.markdown(f"""
<div class="link-box">
📁 <a href="{GITHUB_REPO}" target="_blank" style="color:#667eea">GitHub Repo</a><br>
🥒 <a href="{GITHUB_REPO}/raw/main/model.pkl" target="_blank" style="color:#667eea">model.pkl (download)</a><br>
📓 <a href="{GITHUB_REPO}/blob/main/ECommerce_Churn_Prediction.ipynb" target="_blank" style="color:#667eea">Notebook (.ipynb)</a>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# MODE 1: PREDIKSI MANUAL
# ═══════════════════════════════════════════════
if mode == "🧑 Prediksi Manual (1 Pelanggan)":
    st.subheader("🧑 Input Data Pelanggan")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Informasi Dasar**")
        tenure         = st.number_input("Tenure (bulan)", 0, 60, 6)
        city_tier      = st.selectbox("City Tier",[1,2,3], help="1=Kota Besar, 3=Kota Kecil")
        gender         = st.selectbox("Gender",["Male","Female"])
        marital_status = st.selectbox("Marital Status",["Single","Married","Divorced"])
        num_address    = st.number_input("Jumlah Alamat",1,20,2)

    with col2:
        st.markdown("**📱 Aktivitas Digital**")
        preferred_login   = st.selectbox("Login Device",["Mobile Phone","Computer"])
        hour_spend        = st.slider("Jam/Hari di App",0,5,2)
        num_device        = st.slider("Jumlah Perangkat",1,6,3)
        preferred_payment = st.selectbox("Metode Pembayaran",
                                          ["Debit Card","Credit Card","E wallet",
                                           "Cash on Delivery","UPI"])
        preferred_order   = st.selectbox("Kategori Order Favorit",
                                          ["Laptop & Accessory","Mobile","Mobile Phone",
                                           "Fashion","Grocery","Others"])

    with col3:
        st.markdown("**🛍️ Aktivitas Belanja**")
        warehouse_to_home = st.number_input("Jarak Gudang→Rumah (km)",0,130,15)
        satisfaction      = st.slider("Satisfaction Score (1–5)",1,5,3)
        complain          = st.selectbox("Ada Komplain Bulan Ini?",["Tidak","Ya"])
        order_hike        = st.number_input("Kenaikan Order dari Tahun Lalu (%)",11,26,15)
        coupon_used       = st.number_input("Kupon Dipakai",0,16,1)
        order_count       = st.number_input("Jumlah Order Bulan Ini",1,16,2)
        day_since_last    = st.number_input("Hari Sejak Order Terakhir",0,46,3)
        cashback          = st.number_input("Cashback Rata-Rata",0,325,160)

    st.markdown("---")
    if st.button("🔮 Prediksi Sekarang", use_container_width=True):
        input_df = pd.DataFrame([{
            "Tenure":                      tenure,
            "PreferredLoginDevice":        preferred_login,
            "CityTier":                    city_tier,
            "WarehouseToHome":             warehouse_to_home,
            "PreferredPaymentMode":        preferred_payment,
            "Gender":                      gender,
            "HourSpendOnApp":              hour_spend,
            "NumberOfDeviceRegistered":    num_device,
            "PreferedOrderCat":            preferred_order,
            "SatisfactionScore":           satisfaction,
            "MaritalStatus":               marital_status,
            "NumberOfAddress":             num_address,
            "Complain":                    1 if complain=="Ya" else 0,
            "OrderAmountHikeFromlastYear": order_hike,
            "CouponUsed":                  coupon_used,
            "OrderCount":                  order_count,
            "DaySinceLastOrder":           day_since_last,
            "CashbackAmount":              float(cashback),
        }])

        proba    = model.predict_proba(input_df)[0][1]
        is_churn = proba >= threshold

        r1, r2 = st.columns([1,2])
        with r1:
            if is_churn:
                st.markdown(f"""<div class="churn-box">
                    <h2>⚠️ CHURN</h2><h3>Pelanggan Berisiko Pergi</h3>
                    <h1>{proba*100:.1f}%</h1><p>Probabilitas Churn</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="nochurn-box">
                    <h2>✅ AMAN</h2><h3>Pelanggan Kemungkinan Bertahan</h3>
                    <h1>{proba*100:.1f}%</h1><p>Probabilitas Churn</p>
                </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown("**📊 Detail Probabilitas**")
            st.progress(float(proba))
            st.write(f"Churn Probability : **{proba*100:.1f}%**")
            st.write(f"Threshold         : **{threshold}**")
            st.markdown("---")
            if is_churn:
                st.markdown("""**💡 Rekomendasi:**
- 🎁 Voucher / diskon eksklusif
- 📞 Follow-up personal dari tim retention
- 🔔 Notifikasi promosi yang dipersonalisasi
- ⚡ Prioritaskan resolusi komplain""")
            else:
                st.markdown("""**💡 Rekomendasi:**
- 🌟 Pertahankan kualitas layanan
- 💎 Program loyalitas / membership
- 📈 Upsell produk premium""")


# ═══════════════════════════════════════════════
# MODE 2: BATCH PREDICTION
# ═══════════════════════════════════════════════
else:
    st.subheader("📂 Prediksi Batch — Upload File CSV")
    st.markdown("""<div class="info-box">
📌 <b>Kolom CSV yang diperlukan:</b><br>
<code>Tenure, PreferredLoginDevice, CityTier, WarehouseToHome,
PreferredPaymentMode, Gender, HourSpendOnApp, NumberOfDeviceRegistered,
PreferedOrderCat, SatisfactionScore, MaritalStatus, NumberOfAddress,
Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount,
DaySinceLastOrder, CashbackAmount</code></div>""", unsafe_allow_html=True)

    template = pd.DataFrame([{
        "Tenure":6,"PreferredLoginDevice":"Mobile Phone","CityTier":1,
        "WarehouseToHome":15,"PreferredPaymentMode":"Debit Card","Gender":"Male",
        "HourSpendOnApp":3,"NumberOfDeviceRegistered":3,"PreferedOrderCat":"Mobile",
        "SatisfactionScore":3,"MaritalStatus":"Single","NumberOfAddress":2,
        "Complain":0,"OrderAmountHikeFromlastYear":15,"CouponUsed":1,
        "OrderCount":2,"DaySinceLastOrder":3,"CashbackAmount":160.0,
    }])
    st.download_button("⬇️ Download Template CSV",
                       data=template.to_csv(index=False),
                       file_name="template_churn.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            df_up["PreferredPaymentMode"] = df_up["PreferredPaymentMode"].replace(
                {"COD":"Cash on Delivery","CC":"Credit Card"})
            df_up["PreferredLoginDevice"] = df_up["PreferredLoginDevice"].replace(
                {"Phone":"Mobile Phone"})

            st.write(f"**{df_up.shape[0]} baris** akan diprediksi")
            st.dataframe(df_up.head(), use_container_width=True)

            if st.button("🔮 Prediksi Semua", use_container_width=True):
                prob_b = model.predict_proba(df_up[features])[:,1]
                pred_b = (prob_b >= threshold).astype(int)
                result = df_up.copy()
                result["Prob_Churn"] = np.round(prob_b, 4)
                result["Prediksi"]   = pred_b
                result["Status"]     = result["Prediksi"].map({1:"⚠️ CHURN",0:"✅ AMAN"})

                n_churn = pred_b.sum(); n_total = len(pred_b)
                m1,m2,m3 = st.columns(3)
                m1.metric("Total",n_total)
                m2.metric("⚠️ Churn",n_churn,f"{n_churn/n_total*100:.1f}%",delta_color="inverse")
                m3.metric("✅ Aman",n_total-n_churn,f"{(n_total-n_churn)/n_total*100:.1f}%")
                st.dataframe(result, use_container_width=True)
                st.download_button("⬇️ Download Hasil",
                                   data=result.to_csv(index=False),
                                   file_name="hasil_prediksi_churn.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.markdown("---")
st.markdown(f"""<p style="text-align:center;color:#888;font-size:0.85rem;">
🛒 E-Commerce Churn Predictor · CatBoost · Streamlit |
<a href="{GITHUB_REPO}" target="_blank" style="color:#667eea">GitHub</a></p>""",
unsafe_allow_html=True)
