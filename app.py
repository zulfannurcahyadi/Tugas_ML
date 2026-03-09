import streamlit as st
import pandas as pd
import xgboost as xgb

st.title("Aplikasi Prediksi Harga Diamond")

# Memuat data
@st.cache_data
def load_data():
    return pd.read_csv('diamonds.csv')

df = load_data()
st.write("Preview Data:", df.head())

# Input dari user
karat = st.slider("Pilih berat karat:", 0.2, 5.0, 1.0)
st.write(f"Anda memilih {karat} karat.")

# Contoh tombol untuk prediksi
if st.button("Prediksi"):
    st.write("Model XGBoost akan memproses data Anda di sini...")