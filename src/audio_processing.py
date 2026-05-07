import pandas as pd
import librosa
import numpy as np
import os


def audio_to_mel_spectrogram(audio_path, n_mels=128, hop_length=512):
    y, sr = librosa.load(audio_path, duration=45)
    # Compute the Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    # Convert to decibel scale
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB


def preprocess_deam(audio_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for filename in os.listdir(audio_dir):
        if filename.endswith(".mp3"):
            audio_path = os.path.join(audio_dir, filename)
            mel_spectrogram = audio_to_mel_spectrogram(audio_path)
            save_path = os.path.join(save_dir, os.path.splitext(filename)[0] + ".npy")
            np.save(save_path, mel_spectrogram)


def load_mean_valence_arousal(file_path):
    data = pd.DataFrame()
    for file in os.listdir(file_path):
        if file.endswith(".csv"):
            loaded = pd.read_csv(os.path.join(file_path, file))
            loaded.columns = loaded.columns.str.strip()
            loaded = loaded[["song_id", "valence_mean", "arousal_mean"]]
            data = pd.concat([data, loaded], ignore_index=False)
    data = data[["song_id", "valence_mean", "arousal_mean"]]
    data.set_index("song_id", inplace=True)
    return data
