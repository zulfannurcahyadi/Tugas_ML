import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import os

# Judul Aplikasi
st.title("Aplikasi Prediksi Harga Berlian")

# Fungsi untuk memuat model dan scaler dengan penanganan error
@st.cache_resource
def load_assets():
    if not os.path.exists('model_diamond.pkl') or not os.path.exists('scaler.pkl'):
        return None, None
    
    with open('model_diamond.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

# Memuat model
model, scaler = load_assets()

if model is None or scaler is None:
    st.error("File model atau scaler tidak ditemukan! Pastikan keduanya ada di repositori GitHub.")
else:
    st.success("Model berhasil dimuat!")

    # Contoh Input User (Sesuaikan dengan fitur dataset Anda)
    # Misalkan fitur Anda adalah: carat, depth, table, x, y, z
    carat = st.number_input("Berat Karat", min_value=0.1, max_value=5.0, value=1.0)
    depth = st.number_input("Depth", min_value=40.0, max_value=80.0, value=60.0)
    
    if st.button("Prediksi"):
        # Masukkan data ke list sesuai urutan saat training
        data_input = pd.DataFrame([[carat, depth]], columns=['carat', 'depth'])
        
        # Scaling data
        data_scaled = scaler.transform(data_input)
        
        # Prediksi
        prediksi = model.predict(data_scaled)
        
        st.write(f"Hasil Prediksi Harga: ${prediksi[0]:,.2f}")
