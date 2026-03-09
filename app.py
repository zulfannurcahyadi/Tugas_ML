import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb # Penting agar model bisa dimuat

# Memuat model dan scaler
with open('model_diamond.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Prediksi Harga Berlian")

# Di sini Anda bisa menambahkan input untuk user
# karat = st.number_input("Masukkan Karat")
# ... lalu lakukan scaling sebelum prediksi
# data_input = scaler.transform([[karat, ...]])
# prediksi = model.predict(data_input)
