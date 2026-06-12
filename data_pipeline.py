"""
Data Pipeline Module
- Podział piosnek na train/test
- Ładowanie CSV etykiet
- Łączenie spektrogramów z labelami
- Przygotowanie dla PyTorcha/TensorFlow
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
import json


class DataPipeline:
    """
    Zarządzanie całym pipeliną data:
    - Wczytanie CSV etykiet
    - Podział na train/test
    - Łączenie spektrogramów z labelami
    """
    
    def __init__(self, csv_path: str, mel_spec_folder: str, 
                 test_size: float = 0.2, random_state: int = 42):
        """
        Args:
            csv_path: Ścieżka do CSV z (song_id, time, valence_mean, arousal_mean, ...)
            mel_spec_folder: Folder ze spektrogramami (.npy)
            test_size: % danych do test setu (0.2 = 20%)
            random_state: Dla reproducibilności
        """
        self.csv_path = csv_path
        self.mel_spec_folder = mel_spec_folder
        self.test_size = test_size
        self.random_state = random_state
        
        # Wczytaj CSV
        self.labels_df = pd.read_csv(csv_path)

        print(f"✓ Loaded CSV: {len(self.labels_df)} rows")
        print(f"  Columns: {list(self.labels_df.columns)}")
        print(f"  Sample:\n{self.labels_df.head()}\n")
        
        # Identyfikuj unikalne piosenki
        self.unique_songs = sorted(self.labels_df['song_id'].unique())
        print(f"✓ Found {len(self.unique_songs)} unique songs")
        
        # Weryfikuj że istnieją spektrogramy
        self._verify_spectrograms()
    
    def _verify_spectrograms(self) -> None:
        """Sprawdzić czy wszystkie spektrogramy istnieją."""
        missing = []
        for song_id in self.unique_songs[:5]:  # Sprawdź pierwsze 5
            spec_file = os.path.join(self.mel_spec_folder, f"{song_id}.npy")
            if not os.path.exists(spec_file):
                missing.append(song_id)
        
        if missing:
            print(f"⚠ Warning: Missing spectrograms for {missing}")
        else:
            print(f"✓ Spectrograms exist (checked first 5)")
    
    def split_train_test(self) -> Tuple[List[str], List[str]]:
        """
        Podziel piosenki na train/test na poziomie całych piosenek.
        
        WAŻNE: Dzielisz PIOSENKI, nie segmenty!
        Wtedy każda piosenka będzie TYLKO w train albo TYLKO w test.
        To eliminuje data leakage.
        
        Returns:
            (train_song_ids, test_song_ids)
        """
        train_songs, test_songs = train_test_split(
            self.unique_songs,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        print(f"\n✓ Train/Test Split:")
        print(f"  Train: {len(train_songs)} songs ({len(train_songs)/len(self.unique_songs)*100:.1f}%)")
        print(f"  Test:  {len(test_songs)} songs ({len(test_songs)/len(self.unique_songs)*100:.1f}%)")
        
        return train_songs, test_songs
    
    def create_dataset_dict(self, song_ids: List[str]) -> Dict:
        """
        Dla danej listy piosenek, stwórz dictionary z spektrogramami i labelami.
        
        Struktura:
        {
            'song_id': ['song_001', 'song_001', 'song_001', ...],  # identyfikator
            'time': [0.0, 1.0, 2.0, ...],  # sekunda z CSV
            'mel_spec': [array, array, ...],  # kształt: (n_mels, n_frames_per_window)
            'valence': [0.65, 0.67, 0.64, ...],  # etykieta
            'arousal': [0.48, 0.50, 0.47, ...]   # etykieta
        }
        
        Args:
            song_ids: Lista ID piosenek
            
        Returns:
            Dict ze wszystkimi danymi
        """
        dataset = {
            'song_id': [],
            'time': [],
            'mel_spec': [],
            'valence': [],
            'arousal': [],
            'valence_std': [],
            'arousal_std': []
        }
        
        for song_id in song_ids:
            # Załaduj spektrogram
            spec_file = os.path.join(self.mel_spec_folder, f"{song_id}.npy")
            
            if not os.path.exists(spec_file):
                print(f"⚠ Missing: {spec_file}")
                continue
            
            mel_specs = np.load(spec_file)  # shape: (n_windows, n_mels, n_frames)
            
            # Załaduj labele dla tej piosenki
            song_labels = self.labels_df[self.labels_df['song_id'] == song_id].sort_values('time')
            
            # Dopasuj liczę spektrogramów do liczby etykiet
            # (mogą być różne ze względu na padding)
            n_windows = min(len(mel_specs), len(song_labels))
            
            # Dodaj do dataset'u
            for i in range(n_windows):
                label_row = song_labels.iloc[i]
                
                dataset['song_id'].append(song_id)
                dataset['time'].append(label_row['time'])
                dataset['mel_spec'].append(mel_specs[i])  # (n_mels, n_frames)
                dataset['valence'].append(label_row['valence_mean'])
                dataset['arousal'].append(label_row['arousal_mean'])
                
                # Czy masz std w CSV?
                if 'valence_std' in label_row.index:
                    dataset['valence_std'].append(label_row['valence_std'])
                if 'arousal_std' in label_row.index:
                    dataset['arousal_std'].append(label_row['arousal_std'])
        
        print(f"  Loaded {len(dataset['song_id'])} samples from {len(song_ids)} songs")
        
        return dataset

    def to_python(self, obj):
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def prepare_pipeline(self, output_dir: str = 'data/processed') -> Tuple[Dict, Dict]:
        """
        Kompletny pipeline:
        1. Podziel piosenki na train/test
        2. Stwórz dataset dictionaries
        3. Zapisz jako .npz dla szybkiego ładowania
        
        Returns:
            (train_dataset, test_dataset) - dictionaries
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Krok 1: Podział na poziomie piosenek
        train_songs, test_songs = self.split_train_test()
        
        # Krok 2: Stwórz dataset dictionaries
        print("\n📦 Creating Train Dataset...")
        train_dataset = self.create_dataset_dict(train_songs)
        
        print("\n📦 Creating Test Dataset...")
        test_dataset = self.create_dataset_dict(test_songs)
        
        # Krok 3: Zapisz jako .npz (efficient binary format)
        print(f"\n💾 Saving to {output_dir}...")
        
        # Train
        train_path = os.path.join(output_dir, 'train_dataset.npz')
        np.savez_compressed(
            train_path,
            song_id=np.array(train_dataset['song_id']),
            time=np.array(train_dataset['time']),
            valence=np.array(train_dataset['valence']),
            arousal=np.array(train_dataset['arousal']),
            valence_std=np.array(train_dataset.get('valence_std', [])),
            arousal_std=np.array(train_dataset.get('arousal_std', []))
        )
        
        # Spektrogramy osobno (bo różne kształty!)
        # Zapishemy je jako lista pickled arrays
        import pickle
        with open(os.path.join(output_dir, 'train_mel_specs.pkl'), 'wb') as f:
            pickle.dump(train_dataset['mel_spec'], f)
        
        # Test
        test_path = os.path.join(output_dir, 'test_dataset.npz')
        np.savez_compressed(
            test_path,
            song_id=np.array(test_dataset['song_id']),
            time=np.array(test_dataset['time']),
            valence=np.array(test_dataset['valence']),
            arousal=np.array(test_dataset['arousal']),
            valence_std=np.array(test_dataset.get('valence_std', [])),
            arousal_std=np.array(test_dataset.get('arousal_std', []))
        )
        
        with open(os.path.join(output_dir, 'test_mel_specs.pkl'), 'wb') as f:
            pickle.dump(test_dataset['mel_spec'], f)
        
        # Metadane
        metadata = {
            'n_train_samples': int(len(train_dataset['song_id'])),
            'n_test_samples': int(len(test_dataset['song_id'])),
            'n_train_songs': int(len(train_songs)),
            'n_test_songs': int(len(test_songs)),
            'train_songs': list(train_songs),
            'test_songs': list(test_songs),
            'mel_spec_shape_example': str(train_dataset['mel_spec'][0].shape)
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=self.to_python)
        
        print(f"✓ Saved to {output_dir}")
        print(f"  - train_dataset.npz: Labels + metadata")
        print(f"  - train_mel_specs.pkl: Mel-spectrograms")
        print(f"  - test_dataset.npz: Labels + metadata")
        print(f"  - test_mel_specs.pkl: Mel-spectrograms")
        print(f"  - metadata.json: Info o dataset'cie")
        
        return train_dataset, test_dataset


class DataLoader:
    """
    Ładuj train/test dataset ze zbuforowanych plików.
    Przygotowanie batchy dla modeli ML.
    """
    
    def __init__(self, data_dir: str = 'data/processed', split: str = 'train'):
        """
        Args:
            data_dir: Folder z preprocessowanymi danymi
            split: 'train' lub 'test'
        """
        self.split = split
        self.data_dir = data_dir
        
        # Załaduj metadata
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        # Załaduj labels + metadata
        data = np.load(os.path.join(data_dir, f'{split}_dataset.npz'))
        self.song_ids = data['song_id']
        self.times = data['time']
        self.valences = data['valence']
        self.arousals = data['arousal']
        
        # Załaduj spektrogramy
        import pickle
        with open(os.path.join(data_dir, f'{split}_mel_specs.pkl'), 'rb') as f:
            self.mel_specs = pickle.load(f)
        
        print(f"✓ Loaded {split} dataset: {len(self.song_ids)} samples")
    
    def __len__(self):
        return len(self.song_ids)
    
    def __getitem__(self, idx):
        """Zwróć (mel_spec, valence, arousal) dla indeksu."""
        return {
            'mel_spec': self.mel_specs[idx],  # (n_mels, n_frames)
            'valence': self.valences[idx],     # scalar
            'arousal': self.arousals[idx],     # scalar
            'song_id': self.song_ids[idx],
            'time': self.times[idx]
        }
    
    def get_batch(self, indices: List[int]) -> Dict:
        """
        Zwróć batch dla danych indices.
        
        Returns:
            {
                'mel_specs': (batch_size, n_mels, n_frames),
                'valences': (batch_size,),
                'arousals': (batch_size,)
            }
        """
        mel_specs_list = []
        valences_list = []
        arousals_list = []
        
        for idx in indices:
            mel_specs_list.append(self.mel_specs[idx])
            valences_list.append(self.valences[idx])
            arousals_list.append(self.arousals[idx])
        
        # Stack do array
        # ⚠️ Pamiętaj: spektrogramy mogą mieć różne rozmiary!
        # Dla LSTM to ok (zmienne length), ale dla CNN trzeba padding
        
        return {
            'mel_specs': mel_specs_list,  # Lista arrays (różne rozmiary)
            'valences': np.array(valences_list),
            'arousals': np.array(arousals_list)
        }


# ============================================================================
# PRZYKŁAD UŻYCIA
# ============================================================================

if __name__ == "__main__":
    # Krok 1: Przygotuj cały pipeline
    pipeline = DataPipeline(
        csv_path='music/data_scut/labels.csv',
        mel_spec_folder='music/data_scut/mel_spec_processed',
        test_size=0.2,
        random_state=42
    )
    
    train_dataset, test_dataset = pipeline.prepare_pipeline(
        output_dir='data/processed'
    )
    
    # Krok 2: Załaduj dane (w osobnym skrypcie)
    train_loader = DataLoader('data/processed', split='train')
    test_loader = DataLoader('data/processed', split='test')
    
    print(f"\n✓ Train loader: {len(train_loader)} samples")
    print(f"✓ Test loader: {len(test_loader)} samples")
    
    # Krok 3: Sprawdź sample
    sample = train_loader[0]
    print(f"\nSample shape:")
    print(f"  mel_spec: {sample['mel_spec'].shape}")
    print(f"  valence: {sample['valence']}")
    print(f"  arousal: {sample['arousal']}")