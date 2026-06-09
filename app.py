import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman Web
st.set_page_config(page_title="Prediksi Churn", page_icon="📊", layout="centered")

# Mengambil "Otak" Machine Learning
try:
    model = joblib.load('model_churn.pkl')
except FileNotFoundError:
    st.error("Gagal memuat model: File 'model_churn.pkl' tidak ditemukan di repositori.")
    st.stop()

# Header Aplikasi
st.title("📊 Aplikasi Prediksi Churn Pelanggan")
st.markdown("Masukkan profil dan data langganan pelanggan di bawah ini untuk memprediksi apakah mereka berisiko berhenti berlangganan (*churn*).")

# Membuat Form Antarmuka Pengguna
with st.form("form_prediksi"):
    st.subheader("Data Demografi & Akun")
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Apakah Senior Citizen (Lansia)?", ["No", "Yes"])
        partner = st.selectbox("Memiliki Pasangan?", ["No", "Yes"])
        dependent = st.selectbox("Memiliki Tanggungan?", ["No", "Yes"])
    
    with col2:
        tenure = st.slider("Lama Berlangganan (Bulan)", min_value=0, max_value=72, value=12)
        contract = st.selectbox("Jenis Kontrak", ["monthly", "bimonthly", "quarterly"])
        payment_method = st.selectbox("Metode Pembayaran", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Tagihan Bulanan ($)", min_value=10.0, value=50.0, step=1.0)
        
    submit_button = st.form_submit_button(label="Jalankan Analisis Prediktif")

# --- LOGIKA MESIN PREDIKSI ---
if submit_button:
    # 1. Menampung data dari Form UI
    data_input = {
        'gender': gender,
        'senior_citizen': 1.0 if senior_citizen == "Yes" else 0.0,
        'partner': partner,
        'dependent': dependent,
        'tenure': float(tenure),
        'contract': contract,
        'payment_method': payment_method,
        'monthly_charges': float(monthly_charges)
    }
    
    df_baru = pd.DataFrame([data_input])
    
    # 2. Menerjemahkan data (One-Hot Encoding)
    df_baru_encoded = pd.get_dummies(df_baru)
    
    # 3. Penyelarasan Fitur (Feature Alignment Mutlak)
    # Memastikan format kolom persis dengan format data saat model dilatih
    expected_columns = [
        'senior_citizen', 'tenure', 'monthly_charges', 
        'gender_Male', 'partner_Yes', 'dependent_Yes', 
        'contract_monthly', 'contract_quarterly', 
        'payment_method_Credit card (automatic)', 
        'payment_method_Electronic check', 'payment_method_Mailed check'
    ]
    df_baru_encoded = df_baru_encoded.reindex(columns=expected_columns, fill_value=0)
    
    # 4. Melakukan Proses Klasifikasi
    hasil_prediksi = model.predict(df_baru_encoded)
    probabilitas = model.predict_proba(df_baru_encoded)[0]
    
    # 5. Output Keputusan Bisnis
    st.markdown("---")
    st.subheader("💡 Hasil Prediksi Mesin")
    
    if hasil_prediksi[0] == 1:
        st.error("⚠️ **ZONA MERAH: Pelanggan Berisiko Tinggi Churn**")
        st.markdown(f"**Keyakinan Algoritma:** `{probabilitas[1]*100:.2f}%`")
        st.markdown("> **Rekomendasi Aksi:** Hubungi segera. Berikan atensi khusus, tawarkan diskon perpanjangan atau identifikasi ketidakpuasan layanan untuk mencegah *drop-out*.")
    else:
        st.success("✅ **ZONA AMAN: Pelanggan Loyal**")
        st.markdown(f"**Keyakinan Algoritma:** `{probabilitas[0]*100:.2f}%`")
        st.markdown("> **Rekomendasi Aksi:** Monitor seperti biasa. Performa layanan saat ini dinilai memuaskan ekspektasi pelanggan.")
