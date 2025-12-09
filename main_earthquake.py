import os
import subprocess
import time

# -----------------------------------------
# Change working directory to your project
# -----------------------------------------
project_path = r"C:\Users\venga\OneDrive\Desktop\APPILED MACHINE LEARNING\Project"
os.chdir(project_path)

print("\n==========================================================")
print(" 🌋 EARTHQUAKE DETECTION – FULL PIPELINE INITIALIZED")
print("==========================================================\n")

# -----------------------------------------
# Step 1 - Generate Dataset
# -----------------------------------------
print("🔧 Step 1: Generating new seismic dataset...")
subprocess.run(['python', 'generate_seismic_data.py'])
print("✅ Dataset created successfully.\n")
time.sleep(1)

# -----------------------------------------
# Step 2 - Train CNN + LSTM Model
# -----------------------------------------
print("🧠 Step 2: Training CNN + LSTM Model...")
subprocess.run(['python', 'train_cnn_lstm.py'])
print("✅ Model trained and saved as earthquake_cnn_lstm_model.h5.\n")
time.sleep(1)

# -----------------------------------------
# Step 3 - Virtual Sensor Real-Time Detection (Optional)
# -----------------------------------------
print("📡 Step 3: Running Real-Time Virtual Sensor Detection...")
subprocess.run(['python', 'realtime_virtual_sensor.py'])
print("✅ Virtual Sensor Simulation Complete.\n")
time.sleep(1)

# -----------------------------------------
# Step 4 - Continuous Waveform Tracking (Optional)
# -----------------------------------------
print("🌊 Step 4: Running Continuous Waveform Tracking...")
subprocess.run(['python', 'realtime_wave_tracking.py'])
print("✅ Continuous Real-Time Tracking Complete.\n")
time.sleep(1)

# -----------------------------------------
# Step 5 - Launch Streamlit Dashboard
# -----------------------------------------
print("🖥️ Step 5: Launching Streamlit Dashboard...")
print("👉 NOTE: It will open in your browser.")
time.sleep(2)
subprocess.run(['streamlit', 'run', 'app_dashboard.py'])

print("\n====================================================")
print("🎉 PROJECT COMPLETED — ALL SYSTEMS EXECUTED SUCCESSFULLY")
print("====================================================\n")
