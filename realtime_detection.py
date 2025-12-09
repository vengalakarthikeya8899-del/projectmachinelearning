# ===============================================================
# Real-Time Earthquake Detection with Continuous Wave Tracking
# ===============================================================
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------
# Load trained model and dataset
# ---------------------------------------------------------------
model = load_model("earthquake_cnn_lstm_model.h5", compile=False)
data = np.load("seismic_dataset.npz")
X, y = data["X"], data["y"]

# Parameters
window_size = 100     # number of samples per prediction window
sleep_time = 0.1      # delay between updates (controls speed)
num_cycles = 30       # number of real-time samples to process

print(f"\n📡 Starting continuous real-time detection for {num_cycles} samples...\n")

# ---------------------------------------------------------------
# Initialize live plot
# ---------------------------------------------------------------
plt.ion()
fig, (ax_wave, ax_prob) = plt.subplots(2, 1, figsize=(8, 6))
fig.suptitle("🌋 Real-Time Earthquake Waveform Tracking", fontsize=14)

# Waveform plot
line_wave, = ax_wave.plot([], [], color='royalblue')
ax_wave.set_xlim(0, window_size)
ax_wave.set_ylim(-2, 2)
ax_wave.set_xlabel("Time (samples)")
ax_wave.set_ylabel("Amplitude")
ax_wave.set_title("Live Seismic Waveform")

# Probability tracking plot
prob_values = []
line_prob, = ax_prob.plot([], [], color='darkred', lw=2)
ax_prob.set_xlim(0, num_cycles)
ax_prob.set_ylim(0, 1.05)
ax_prob.set_xlabel("Cycle Number")
ax_prob.set_ylabel("Detection Probability")
ax_prob.set_title("Prediction Probability Over Time")

# ---------------------------------------------------------------
# Real-time detection loop
# ---------------------------------------------------------------
for cycle in range(num_cycles):
    sample_idx = np.random.randint(0, len(X))
    signal = X[sample_idx]
    label = y[sample_idx]
    true_label = "🌋 Earthquake" if label == 1 else "🌎 Noise"

    # Run model prediction
    x_input = np.expand_dims(signal, axis=0)
    prob = model.predict(x_input, verbose=0)[0][0]
    prob_values.append(prob)

    # Classification result
    status = "🌋 EARTHQUAKE DETECTED!" if prob > 0.6 else "🌎 Normal Noise"
    print(f"Cycle {cycle+1:02d}/{num_cycles} | Prob={prob:.3f} | Pred={status} | True={true_label}")

    # Update waveform plot
    line_wave.set_data(np.arange(len(signal)), signal[:, 0])
    ax_wave.relim()
    ax_wave.autoscale_view()

    # Update probability tracking plot
    line_prob.set_data(np.arange(len(prob_values)), prob_values)
    ax_prob.axhline(0.6, color='gray', linestyle='--', linewidth=1)
    plt.pause(sleep_time)

plt.ioff()
plt.show()

print("\n✅ Real-time wave tracking complete — 30 samples processed.\n")
# This script simulates real-time earthquake detection by continuously processing seismic waveform data through a pre-trained CNN + LSTM model and visualizing the results.