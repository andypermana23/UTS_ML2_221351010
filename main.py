import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# --- Load Model dan Tools ---
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="stroke.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_utils():
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return scaler, label_encoder

interpreter = load_model()
scaler, le = load_utils()

# --- Fungsi Prediksi ---
def predict(input_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_array.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# --- UI Streamlit ---
st.title("🧠 Prediksi Risiko Stroke")
st.write("Isi data pasien di bawah untuk mengetahui kemungkinan terkena stroke.")

gender = st.radio("Jenis Kelamin", ['Perempuan', 'Laki-laki'])
gender_val = 0 if gender == 'Perempuan' else 1

age = st.slider("Umur", 1, 100, 30)
hypertension = st.selectbox("Penderita Hipertensi?", ['Tidak', 'Ya'])
ht_val = 0 if hypertension == 'Tidak' else 1

heart_disease = st.selectbox("Punya Riwayat Penyakit Jantung?", ['Tidak', 'Ya'])
hd_val = 0 if heart_disease == 'Tidak' else 1

bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)

# --- Prediksi ---
if st.button("🔍 Prediksi"):
    input_data = np.array([[gender_val, age, ht_val, hd_val, bmi]])
    input_scaled = scaler.transform(input_data)

    pred = predict(input_scaled)
    pred_label = np.argmax(pred)
    confidence = np.max(pred)

    if pred_label == 1:
        st.error(f"⚠ Pasien berisiko terkena stroke! (Confidence: {confidence:.2f})")
    else:
        st.success(f"✅ Pasien tidak berisiko stroke. (Confidence: {confidence:.2f})")

# --- Footer ---
st.caption("Model dikembangkan menggunakan TensorFlow, Keras, dan Streamlit.")