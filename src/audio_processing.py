import pandas as pd
import librosa
import numpy as np
import os
from tensorflow.keras.utils import Sequence


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


def load_mean_valence_arousal(dir_path):
    file_path = os.path.join(dir_path, "DEAM/annotations/annotations averaged per song/song_level/")
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


def reduce_time_columns(df, num_cols=5, suffix=""):
    id_col = df[["song_id"]].reset_index(drop=True)
    time_cols = df.drop(columns=["song_id"])
    grouped = []
    cols = time_cols.columns.tolist()

    for i in range(0, len(cols), num_cols):
        group = time_cols[cols[i : (i + num_cols)]]
        group_mean = group.mean(axis=1)
        # group_mean = time_cols[group].mean(axis=1)
        group_name = f"{cols[i]}_{suffix}"
        grouped.append(group_mean.rename(group_name))

    result = pd.concat([id_col, pd.concat(grouped, axis=1)], axis=1)
    return result


def load_valence_arousal(dir_path, span=0.5):
    file_path = os.path.join(
        dir_path, "DEAM/annotations/annotations averaged per song/dynamic (per second annotations)/"
    )
    df1 = pd.read_csv(os.path.join(file_path, "arousal.csv"))
    df2 = pd.read_csv(os.path.join(file_path, "valence.csv"))
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    if span > 0.5:
        df1 = reduce_time_columns(df1, num_cols=int(span * 2), suffix="arousal")
        df2 = reduce_time_columns(df2, num_cols=int(span * 2), suffix="valence")

    data = pd.merge(df1, df2, on=["song_id"])

    return data


class DEAMGenerator(Sequence):
    def __init__(self, song_ids, labels, data_dir, batch_size=32, shuffle=True):
        self.song_ids = song_ids
        self.labels = labels
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.song_ids) / self.batch_size))

    def __getitem__(self, index):
        batch_song_ids = self.song_ids[index * self.batch_size : (index + 1) * self.batch_size]

        X = []
        y = []

        for song_id in batch_song_ids:
            mel_spectrogram = np.load(os.path.join(self.data_dir, f"{song_id}.npy"))

            # mel_spectrogram = # przycinanie lub padding do stałego rozmiaru (np. 128x128)

            X.append(mel_spectrogram)
            y.append(self.labels.loc[song_id].values)

        X = np.array(X)[..., np.newaxis]  # Add channel dimension
        y = np.array(y)
        return X, y
