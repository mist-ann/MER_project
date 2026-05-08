"""
Poprawiony DataPipeline — naprawa Bug #2 (label alignment)
==========================================================

Zmiana względem oryginalnego data_pipeline.py:

  PRZED (błąd):
      song_labels = self.labels_df[
          self.labels_df['song_id'] == song_id
      ].sort_values('time')

      for i in range(n_windows):
          label_row = song_labels.iloc[i]   # bierze t=0, 1, 2, ...
          dataset['mel_spec'].append(mel_specs[i])

  PO (naprawa):
      song_labels = self.labels_df[
          (self.labels_df['song_id'] == song_id) &
          (self.labels_df['time'] >= self.start_time)  # filtruj od start_time!
      ].sort_values('time')

  Dlaczego to ważne:
      AudioProcessor.process_single_audio wycina audio:
          audio = audio[start:end]  gdzie start = int(start_time * sr)
      Więc mel_specs[0] reprezentuje sekundę 14-15 w oryginalnym audio.
      Ale labels_df sortowane po 'time' ma iloc[0] → t=0 (sekunda 0-1).
      To powoduje przesunięcie etykiet o 14 sekund → systematyczny błąd.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split


class DataPipeline:
    """
    Zarządzanie pipeliną danych z poprawioną obsługą czasową.

    WAŻNE: start_time musi być identyczny jak AudioProcessor.start_time
    """

    def __init__(self, csv_path: str, mel_spec_folder: str,
                 test_size: float = 0.2, random_state: int = 42,
                 start_time: float = 14.0):
        """
        Args:
            csv_path:        Ścieżka do CSV (song_id, time, valence_mean, arousal_mean, ...)
            mel_spec_folder: Folder ze spektrogramami .npy
            test_size:       Frakcja danych testowych
            random_state:    Seed dla reproducibilności
            start_time:      Sekunda od której AudioProcessor zaczął wycinać audio.
                             Musi być identyczna jak w AudioProcessor.
                             Domyślnie 14.0 (jak w AudioProcessor z kodu projektu).
        """
        self.csv_path = csv_path
        self.mel_spec_folder = mel_spec_folder
        self.test_size = test_size
        self.random_state = random_state
        self.start_time = start_time   # <-- przechowujemy dla label alignment

        self.labels_df = pd.read_csv(csv_path)
        print(f"✓ Loaded CSV: {len(self.labels_df)} rows")
        print(f"  Columns: {list(self.labels_df.columns)}")

        # Weryfikacja: czy CSV zawiera kolumnę 'time'
        if 'time' not in self.labels_df.columns:
            raise ValueError("CSV musi zawierać kolumnę 'time'")

        # Statystyki zakresów etykiet
        print(f"\n  Zakresy etykiet w CSV:")
        for col in ['valence_mean', 'arousal_mean']:
            if col in self.labels_df.columns:
                s = self.labels_df[col]
                print(f"    {col}: [{s.min():.3f}, {s.max():.3f}], "
                      f"mean={s.mean():.3f}")

        self.unique_songs = sorted(self.labels_df['song_id'].unique())
        print(f"\n✓ Found {len(self.unique_songs)} unique songs")
        print(f"  start_time = {self.start_time}s (etykiety filtrowane od tej sekundy)")

        self._verify_spectrograms()

    def _verify_spectrograms(self) -> None:
        missing = [
            sid for sid in self.unique_songs[:10]
            if not os.path.exists(
                os.path.join(self.mel_spec_folder, f"{sid}.npy")
            )
        ]
        if missing:
            print(f"⚠ Missing spectrograms for: {missing}")
        else:
            print(f"✓ Spectrograms verified (checked first 10)")

    def split_train_test(self) -> Tuple[List[str], List[str]]:
        """
        Podział na poziomie całych piosenek — eliminuje data leakage.
        """
        train_songs, test_songs = train_test_split(
            self.unique_songs,
            test_size=self.test_size,
            random_state=self.random_state
        )
        n = len(self.unique_songs)
        print(f"\n✓ Train: {len(train_songs)} songs ({len(train_songs)/n*100:.1f}%)")
        print(f"  Test:  {len(test_songs)} songs ({len(test_songs)/n*100:.1f}%)")
        return list(train_songs), list(test_songs)

    def create_dataset_dict(self, song_ids: List[str]) -> Dict:
        """
        Tworzy słownik danych dla podanej listy piosenek.

        BUG #2 FIX: filtrujemy etykiety CSV do time >= start_time,
        żeby dopasowanie okno-mel[i] ↔ etykieta[i] było poprawne.
        """
        dataset = {
            'song_id': [], 'time': [], 'mel_spec': [],
            'valence': [], 'arousal': [],
            'valence_std': [], 'arousal_std': []
        }

        skipped = 0
        for song_id in song_ids:
            spec_file = os.path.join(self.mel_spec_folder, f"{song_id}.npy")
            if not os.path.exists(spec_file):
                skipped += 1
                continue

            mel_specs = np.load(spec_file)  # (n_windows, n_mels, n_frames)

            # BUG #2 FIX: tylko etykiety od start_time wzwyż
            song_labels = self.labels_df[
                (self.labels_df['song_id'] == song_id) &
                (self.labels_df['time'] >= self.start_time)   # <-- NAPRAWA
            ].sort_values('time').reset_index(drop=True)

            if len(song_labels) == 0:
                print(f"⚠ Brak etykiet od t>={self.start_time} dla: {song_id}")
                skipped += 1
                continue

            n_windows = min(len(mel_specs), len(song_labels))

            for i in range(n_windows):
                row = song_labels.iloc[i]
                dataset['song_id'].append(song_id)
                dataset['time'].append(float(row['time']))
                dataset['mel_spec'].append(mel_specs[i])
                dataset['valence'].append(float(row['valence_mean']))
                dataset['arousal'].append(float(row['arousal_mean']))

                if 'valence_std' in row.index:
                    dataset['valence_std'].append(float(row['valence_std']))
                if 'arousal_std' in row.index:
                    dataset['arousal_std'].append(float(row['arousal_std']))

        print(f"  Loaded {len(dataset['song_id'])} samples "
              f"({skipped} songs skipped)")
        return dataset

    def prepare_pipeline(self, output_dir: str = 'data/processed') -> Tuple[Dict, Dict]:
        """
        Pełny pipeline: podział → dataset → zapis .npz + .pkl
        """
        os.makedirs(output_dir, exist_ok=True)

        train_songs, test_songs = self.split_train_test()

        print("\n📦 Creating Train Dataset...")
        train_dataset = self.create_dataset_dict(train_songs)

        print("\n📦 Creating Test Dataset...")
        test_dataset = self.create_dataset_dict(test_songs)

        print(f"\n💾 Saving to {output_dir}...")

        for split_name, ds, songs in [
            ('train', train_dataset, train_songs),
            ('test',  test_dataset,  test_songs)
        ]:
            np.savez_compressed(
                os.path.join(output_dir, f'{split_name}_dataset.npz'),
                song_id=np.array(ds['song_id']),
                time=np.array(ds['time']),
                valence=np.array(ds['valence'], dtype=np.float32),
                arousal=np.array(ds['arousal'], dtype=np.float32),
                valence_std=np.array(ds.get('valence_std', []), dtype=np.float32),
                arousal_std=np.array(ds.get('arousal_std', []), dtype=np.float32),
            )
            with open(os.path.join(output_dir, f'{split_name}_mel_specs.pkl'), 'wb') as f:
                pickle.dump(ds['mel_spec'], f)

        # Metadane
        meta = {
            'n_train_samples': len(train_dataset['song_id']),
            'n_test_samples':  len(test_dataset['song_id']),
            'n_train_songs':   len(train_songs),
            'n_test_songs':    len(test_songs),
            'train_songs':     train_songs,
            'test_songs':      test_songs,
            'start_time':      self.start_time,     # zapisz dla spójności!
            'mel_spec_shape':  str(train_dataset['mel_spec'][0].shape)
                               if train_dataset['mel_spec'] else 'unknown',
            'valence_range':   [float(np.min(train_dataset['valence'])),
                                float(np.max(train_dataset['valence']))],
            'arousal_range':   [float(np.min(train_dataset['arousal'])),
                                float(np.max(train_dataset['arousal']))],
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"✓ Saved. Zakresy etykiet (train):")
        print(f"  Valence: {meta['valence_range']}")
        print(f"  Arousal: {meta['arousal_range']}")

        return train_dataset, test_dataset


# ============================================================
# Przykład użycia
# ============================================================

if __name__ == "__main__":
    # pipeline = DataPipeline(
    #     csv_path='sp2/labels.csv',
    #     mel_spec_folder='~/Documents/datasets/mel_spec_processed',
    #     test_size=0.2,
    #     random_state=42,
    #     start_time=14.0     # identyczne jak AudioProcessor.start_time!
    # )

    # train_ds, test_ds = pipeline.prepare_pipeline(output_dir='sp2/data/processed')

    import pandas as pd
    df = pd.read_csv('sp2/labels.csv')
    print(df.head(20))
    print(df.dtypes)
    print(df['time'].describe())
    print(df['time'].unique()[:20])