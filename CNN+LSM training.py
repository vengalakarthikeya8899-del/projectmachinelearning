# train_cnn_lstm.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def build_cnn_lstm(input_shape):
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.LSTM(64),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    data = np.load("seismic_dataset.npz")
    X, y = data["X"], data["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_cnn_lstm((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    model.save("earthquake_cnn_lstm_model.h5")
    print("✅ Model saved as 'earthquake_cnn_lstm_model.h5'")

    # Evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss'); plt.legend()
    plt.show()

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Noise','Earthquake']).plot(cmap='Blues')
    plt.show()

if __name__ == "__main__":
    train_model()
# This script trains a CNN+LSTM model on seismic waveform data to classify between earthquake signals and noise.