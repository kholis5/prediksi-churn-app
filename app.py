
import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('model_churn.pkl')

st.title("Aplikasi Prediksi Churn Pelanggan")
st.write("Aplikasi ini sedang dalam tahap pengembangan.")
