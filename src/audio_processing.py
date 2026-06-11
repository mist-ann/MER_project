import re
import pandas as pd
import librosa
import numpy as np
import os
from tensorflow.keras.utils import Sequence


def _extract_time_ms(column_name):
    match = re.search(r"sample_(\d+)ms", column_name)
    if match:
        return int(match.group(1))

    match = re.search(r"\d+", column_name)
    if match:
        return int(match.group())

    return None


def audio_to_mel_spectrogram(audio_path, n_mels=128, hop_length=512, sr=22050):
    y, sr = librosa.load(audio_path, duration=45, sr=sr)
    # Compute the Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    # Convert to decibel scale
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB


def preprocess_deam(audio_dir, save_dir, n_mels=128, hop_length=512, sr=22050):
    os.makedirs(save_dir, exist_ok=True)
    for filename in os.listdir(audio_dir):
        if filename.endswith(".mp3"):
            audio_path = os.path.join(audio_dir, filename)
            mel_spectrogram = audio_to_mel_spectrogram(audio_path, n_mels=n_mels, hop_length=hop_length, sr=sr)
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
    time_cols = sorted(
        [col for col in df.columns if col != "song_id"],
        key=lambda col: _extract_time_ms(col) if _extract_time_ms(col) is not None else -1,
    )
    grouped = []

    for i in range(0, len(time_cols), num_cols):
        group_cols = time_cols[i : (i + num_cols)]
        if len(group_cols) < num_cols:
            break

        group = df[group_cols]
        group_mean = group.mean(axis=1)
        time_ms = _extract_time_ms(group_cols[0])
        group_name = f"sample_{time_ms}ms"
        if suffix:
            group_name = f"{group_name}_{suffix}"
        grouped.append(group_mean.rename(group_name))

    result = pd.concat([id_col, pd.concat(grouped, axis=1)], axis=1)
    return result


def load_valence_arousal(dir_path, span=0.5):
    if span < 0.5:
        raise ValueError("span must be at least 0.5 seconds for DEAM dynamic annotations.")

    file_path = os.path.join(
        dir_path, "DEAM/annotations/annotations averaged per song/dynamic (per second annotations)/"
    )
    arousal_df = pd.read_csv(os.path.join(file_path, "arousal.csv"))
    valence_df = pd.read_csv(os.path.join(file_path, "valence.csv"))
    arousal_df.columns = arousal_df.columns.str.strip()
    valence_df.columns = valence_df.columns.str.strip()

    arousal_time_cols = [col for col in arousal_df.columns if col != "song_id"]
    valence_time_cols = [col for col in valence_df.columns if col != "song_id"]
    common_time_cols = sorted(
        set(arousal_time_cols).intersection(valence_time_cols),
        key=lambda col: _extract_time_ms(col) if _extract_time_ms(col) is not None else -1,
    )

    arousal_df = arousal_df[["song_id"] + common_time_cols]
    valence_df = valence_df[["song_id"] + common_time_cols]

    base_step_seconds = 0.5
    num_cols = int(round(span / base_step_seconds))
    if not np.isclose(num_cols * base_step_seconds, span):
        raise ValueError("span must be a multiple of 0.5 seconds.")

    arousal_df = reduce_time_columns(arousal_df, num_cols=num_cols, suffix="arousal")
    valence_df = reduce_time_columns(valence_df, num_cols=num_cols, suffix="valence")

    data = pd.merge(arousal_df, valence_df, on=["song_id"])

    return data


def get_song_ids_and_labels(labels_df):
    dynamic_labels = {}

    for _, row in labels_df.iterrows():
        song_id = str(int(row["song_id"]))
        dynamic_labels[song_id] = {}
        arousal_cols = sorted(
            [col for col in labels_df.columns if col.endswith("_arousal")],
            key=lambda col: _extract_time_ms(col) if _extract_time_ms(col) is not None else -1,
        )
        for col in arousal_cols:
            time_ms = _extract_time_ms(col)
            col_valence = col.replace("_arousal", "_valence")
            if time_ms is None or col_valence not in labels_df.columns:
                continue

            value_arousal = row[col]
            value_valence = row[col_valence]
            if pd.isna(value_arousal) or pd.isna(value_valence):
                continue

            dynamic_labels[song_id][time_ms] = (value_valence, value_arousal)
    return dynamic_labels


class DEAMSegmentGenerator(Sequence):
    def __init__(
        self,
        song_ids,
        labels_dict,
        data_dir,
        segment_width=128,
        hop_size=512,
        sr=22050,
        batch_size=32,
        shuffle=True,
        window_seconds=3.0,
        label_start_ms=15000,
        normalize_segments=False,
    ):
        self.song_ids = [str(song_id) for song_id in song_ids]
        self.labels = labels_dict
        self.data_dir = data_dir
        self.segment_width = segment_width
        self.hop_size = hop_size
        self.sr = sr
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.window_seconds = window_seconds
        self.label_start_ms = label_start_ms
        self.normalize_segments = normalize_segments

        self.samples = []
        self.hop_size_frames = max(1, int(round((self.window_seconds * self.sr) / self.hop_size)))

        # Tworzymy listę wszystkich możliwych okien ze wszystkich utworów
        for song_id in self.song_ids:
            if song_id not in self.labels or not self.labels[song_id]:
                continue

            file_path = os.path.join(self.data_dir, f"{song_id}.npy")
            if os.path.exists(file_path):
                # spec = np.load(file_path, mmap_mode="r")  # mmap_mode nie ładuje całego pliku do RAM
                # total_frames = spec.shape[1]

                # # Obliczamy punkty startowe dla okien
                # for start in range(0, total_frames - segment_width, hop_size):
                #     self.samples.append((song_id, start))
                # Zaczynamy od 15s (DEAM), idziemy skokiem równym spanowi
                start_min = int(round((self.label_start_ms / 1000) * self.sr / self.hop_size))
                # Używamy mmap_mode, żeby tylko sprawdzić rozmiar pliku
                spec_shape = np.load(file_path, mmap_mode="r").shape[1]
                max_start = spec_shape - segment_width
                for start in range(start_min, max_start + 1, self.hop_size_frames):
                    self.samples.append((song_id, start))

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        batch_samples = self.samples[index * self.batch_size : (index + 1) * self.batch_size]

        X = []
        y = []

        for song_id, start in batch_samples:
            mel_spectrogram = np.load(os.path.join(self.data_dir, f"{song_id}.npy"), mmap_mode="r")
            segment = np.asarray(mel_spectrogram[:, start : start + self.segment_width], dtype=np.float32)

            # # normalizacja
            # segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-6)
            if self.normalize_segments:
                segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-6)

            # # mel_spectrogram = # przycinanie lub padding do stałego rozmiaru (np. 128x128)

            # X.append(segment)
            # y.append(self.labels.loc[song_id])

            # Obliczamy czas startu w ms, aby dopasować do słownika
            time_ms = int((start * self.hop_size / self.sr) * 1000)

            # Szukamy najbliższego klucza
            available_times = list(self.labels[song_id].keys())
            closest_time = min(available_times, key=lambda x: abs(x - time_ms))

            X.append(segment)
            y.append(self.labels[song_id][closest_time])

        X = np.array(X)[..., np.newaxis]  # Add channel dimension (Batch, Height, Width, 1)
        y = np.array(y)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)


class DEAMGenerator(Sequence):
    def __init__(
        self, song_ids, labels_dict, data_dir, segment_width=128, hop_size=64, sr=22050, batch_size=32, shuffle=True
    ):
        self.song_ids = song_ids
        self.labels = labels_dict
        self.segment_width = segment_width
        self.hop_size = hop_size
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
