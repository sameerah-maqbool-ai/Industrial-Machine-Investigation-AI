import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

# -------------------------------
# 1️⃣ Set main dataset folder
# -------------------------------
dataset_path = r"C:/Users/ART/Desktop/motor_data"  # <- change if your folder is elsewhere

# -------------------------------
# 2️⃣ Helper function to get audio files (max 3 per device)
# -------------------------------
def get_audio_files_limited(base_path, max_per_device=3):
    audio_files = []
    counts = {}
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.wav'):
                device_type = root.split(os.sep)[-2]  # fan/gearbox/pump/valve
                if device_type not in counts:
                    counts[device_type] = 0
                if counts[device_type] < max_per_device:
                    full_path = os.path.join(root, file)
                    dataset_type = root.split(os.sep)[-1]  # train/source_test/target_test
                    audio_files.append((full_path, device_type, dataset_type))
                    counts[device_type] += 1
    return audio_files

# -------------------------------
# 3️⃣ Collect limited audio files
# -------------------------------
audio_files = get_audio_files_limited(dataset_path)
print(f"Total audio files selected: {len(audio_files)}")

df = pd.DataFrame(audio_files, columns=['file_path','device','dataset_type'])
print(df.head())

# -------------------------------
# 4️⃣ Plot device distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='device', data=df)
plt.title("Device Type Distribution (sample)")
plt.show()

# -------------------------------
# 5️⃣ Audio loading
# -------------------------------
def load_audio(file_path, sr=16000):
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

# -------------------------------
# 6️⃣ Plot waveform, FFT, spectrogram for each sample
# -------------------------------
for idx, row in df.iterrows():
    y, sr = load_audio(row['file_path'])
    
    # Waveform
    plt.figure(figsize=(10,3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform - {row['device']} ({row['dataset_type']})")
    plt.show()
    
    # FFT
    N = len(y)
    Y = np.abs(np.fft.fft(y)[:N//2])
    freqs = np.fft.fftfreq(N, 1/sr)[:N//2]
    plt.figure(figsize=(10,3))
    plt.plot(freqs, Y)
    plt.title(f"FFT Spectrum - {row['device']}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.show()
    
    # Spectrogram
    D = librosa.stft(y)
    DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(format="%+2.f dB")
    plt.title(f"Spectrogram - {row['device']}")
    plt.show()

# -------------------------------
# 7️⃣ Quick feature extraction
# -------------------------------
features_list = []

for idx, row in df.iterrows():
    y, sr = load_audio(row['file_path'])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features_list.append([row['device'], row['dataset_type'], rms, zcr, spec_cent, *mfcc_mean])

columns = ['device','dataset_type','rms','zcr','spectral_centroid'] + [f'mfcc{i}' for i in range(1,14)]
features_df = pd.DataFrame(features_list, columns=columns)
print(features_df.head())
