import streamlit as st
import pandas as pd
import pickle

# Memuat model dan scaler
@st.cache_resource
def load_assets():
    with open('model_diamond.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

model, scaler = load_assets()

st.title("Aplikasi Prediksi Harga Berlian")

# Input fitur sesuai dataset (Urutan harus sama dengan saat training)
# Fitur: ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
st.subheader("Masukkan Spesifikasi Berlian:")

carat = st.number_input("Carat", 0.2, 5.0, 1.0)
cut = st.selectbox("Cut (0: Fair, 1: Good, 2: Ideal, 3: Premium, 4: Very Good)", [0, 1, 2, 3, 4])
color = st.selectbox("Color (0: D, 1: E, 2: F, 3: G, 4: H, 5: I, 6: J)", [0, 1, 2, 3, 4, 5, 6])
clarity = st.selectbox("Clarity (0: I1, 1: IF, 2: SI1, 3: SI2, 4: VS1, 5: VS2, 6: VVS1, 7: VVS2)", [0, 1, 2, 3, 4, 5, 6, 7])
depth = st.number_input("Depth", 40.0, 80.0, 60.0)
table = st.number_input("Table", 40.0, 95.0, 50.0)
x = st.number_input("X (Length)", 0.0, 10.0, 5.0)
y = st.number_input("Y (Width)", 0.0, 10.0, 5.0)
z = st.number_input("Z (Depth)", 0.0, 10.0, 3.0)

if st.button("Prediksi Harga"):
    # Membuat dataframe input
    input_data = pd.DataFrame([[carat, cut, color, clarity, depth, table, x, y, z]], 
                              columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])
    
    # Scaling data
    scaled_data = scaler.transform(input_data)
    
    # Prediksi
    harga = model.predict(scaled_data)
    
    st.success(f"Estimasi harga berlian Anda adalah: ${harga[0]:,.2f}")