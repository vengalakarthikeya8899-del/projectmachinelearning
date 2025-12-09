# ==========================================================
# Streamlit Dashboard — Real-Time Earthquake Detection (with Refresh)
# ==========================================================
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Earthquake Detection Dashboard", layout="centered")
st.title("🌋 Real-Time Earthquake Detection Dashboard")

# ----------------------------------------------------------
# Load trained model and dataset
# ----------------------------------------------------------
@st.cache_resource
def load_components():
    model = load_model("earthquake_cnn_lstm_model.h5", compile=False)

    data = np.load("seismic_dataset.npz")
    return model, data["X"], data["y"]

model, X, y = load_components()

# ----------------------------------------------------------
# Refresh button
# ----------------------------------------------------------
if st.button("🔄 Load New Seismic Signal"):
    st.session_state["idx"] = np.random.randint(0, len(X))

if "idx" not in st.session_state:
    st.session_state["idx"] = np.random.randint(0, len(X))

idx = st.session_state["idx"]
signal = X[idx]
label = y[idx]

# ----------------------------------------------------------
# Model prediction
# ----------------------------------------------------------
prob = model.predict(np.expand_dims(signal, axis=0), verbose=0)[0][0]

# ----------------------------------------------------------
# Visualization
# ----------------------------------------------------------
st.subheader("Incoming Seismic Waveform")
fig, ax = plt.subplots()
ax.plot(signal[:, 0], color='royalblue')
ax.set_xlabel("Time (samples)")
ax.set_ylabel("Amplitude")
ax.grid(True)
st.pyplot(fig)

# ----------------------------------------------------------
# Output classification result
# ----------------------------------------------------------
if prob > 0.6:
    st.error(f"🌋 Earthquake Detected! (Probability = {prob:.2f})")
else:
    st.success(f"🌎 Normal Noise (Probability = {prob:.2f})")

st.caption(f"True label: {'Earthquake' if label==1 else 'Noise'}")
# This Streamlit app provides a dashboard for real-time earthquake detection using a pre-trained CNN + LSTM model.