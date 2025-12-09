# ==========================================================
# Real-Time Earthquake Detection (Virtual Sensor + Live Graph)
# ==========================================================
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ----------------------------------------------------------
# Load model
# ----------------------------------------------------------
model = load_model("earthquake_cnn_lstm_model.h5", compile=False)

print("\n🌍 Starting Real-Time Earthquake Detection (Virtual Sensor Mode)\n")

# ----------------------------------------------------------
# Virtual Sensor Simulation
# ----------------------------------------------------------
def simulate_virtual_sensor_signal(length=100, noise_level=0.05):
    """
    Simulates a vibration sensor:
    - Earthquake: sinusoidal burst + noise
    - Normal: random noise
    """
    is_earthquake = np.random.rand() > 0.7  # 30% chance of earthquake
    t = np.linspace(0, 1, length)

    if is_earthquake:
        envelope = np.exp(-((t - 0.3) ** 2) / 0.01)
        freq = np.random.uniform(5, 15)
        signal = envelope * np.sin(2 * np.pi * freq * t)
        signal += noise_level * np.random.randn(length)
        label = 1
    else:
        signal = noise_level * np.random.randn(length)
        label = 0

    signal_3ch = np.stack([
        signal,
        0.8 * signal + noise_level * np.random.randn(length),
        1.2 * signal + noise_level * np.random.randn(length)
    ], axis=1)

    return signal_3ch, label

# ----------------------------------------------------------
# Live Plot Setup
# ----------------------------------------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot([], [], lw=2, color='royalblue')
ax.set_xlim(0, 100)
ax.set_ylim(-2, 2)
ax.set_xlabel("Time (samples)")
ax.set_ylabel("Amplitude")
ax.set_title("🌋 Real-Time Seismic Waveform (Virtual Sensor)")

text_box = ax.text(0.02, 1.6, "", fontsize=12, color="darkred", weight="bold")

# ----------------------------------------------------------
# Detection Loop
# ----------------------------------------------------------
for i in range(30):  # Run for 30 cycles
    signal, label = simulate_virtual_sensor_signal()
    x_input = np.expand_dims(signal, axis=0)
    prob = model.predict(x_input, verbose=0)[0][0]

    status = "🌋 EARTHQUAKE DETECTED!" if prob > 0.6 else "🌎 Normal Noise"
    print(f"Cycle {i+1:02d} | Prob={prob:.3f} | {status} | True: {'Earthquake' if label==1 else 'Noise'}")

    # Update plot
    line.set_data(np.arange(len(signal)), signal[:, 0])
    text_box.set_text(f"{status}\nProb={prob:.2f}")
    plt.pause(0.8)

plt.ioff()
plt.show()

print("\n✅ Simulation complete. Stopped after 30 detections.\n")
