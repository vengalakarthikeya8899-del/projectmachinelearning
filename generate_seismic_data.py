# generate_seismic_data.py
import numpy as np

def generate_earthquake_waveform(length=100, num_signals=1000):
    X = []
    for _ in range(num_signals):
        t = np.linspace(0, 1, length)
        envelope = np.exp(-((t - 0.3) ** 2) / 0.01)
        freq = np.random.uniform(5, 15)
        signal = envelope * np.sin(2 * np.pi * freq * t)
        waveform = np.stack([
            signal + 0.05 * np.random.randn(length),
            0.8 * signal + 0.05 * np.random.randn(length),
            1.2 * signal + 0.05 * np.random.randn(length)
        ], axis=1)
        X.append(waveform)
    return np.array(X)

def generate_noise_waveform(length=100, num_signals=1000):
    X = []
    for _ in range(num_signals):
        noise = 0.1 * np.random.randn(length, 3)
        X.append(noise)
    return np.array(X)

def create_dataset():
    X_eq = generate_earthquake_waveform()
    X_noise = generate_noise_waveform()
    X = np.concatenate([X_eq, X_noise])
    y = np.concatenate([np.ones(len(X_eq)), np.zeros(len(X_noise))])
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    np.savez("seismic_dataset.npz", X=X, y=y)
    print("✅ Dataset saved as 'seismic_dataset.npz'")

if __name__ == "__main__":
    create_dataset()
