"""
Audio Processing Module
Konwersja MP3->WAV, ekstrakcja mel-spektrogramów z windowing
"""

import os
os.environ["NUMBA_DISABLE_COVERAGE"] = "1"
os.environ["NUMBA_DISABLE_JIT"] = "1"
import numpy as np
import librosa
import warnings
from pathlib import Path
from typing import Tuple, List, Optional

warnings.filterwarnings('ignore')



class AudioProcessor:
    """
    Unified audio processor dla całego pipeline'u.
    Obsługuje:
    - MP3 -> WAV konwersja (opcjonalna, masz już WAV)
    - Windowing audio na segmenty 1-sekundowe
    - Ekstrakcja mel-spektrogramów
    """
    
    def __init__(self, sr: int = 22050, n_mels: int = 128, 
                 window_duration: float = 1.0, 
                 start_time: float = 14, end_time: float = 60):
        """
        Args:
            sr: Sample rate (22050 = standard dla librosa)
            n_mels: Liczba mel-bins (128 = standard, mogą być 64 czy 256)
            window_duration: Długość okna w sekundach (1.0 = 1 sec)
            start_time: Pomijaj pierwsze N sekund (Ty masz 14 - noise?)
            end_time: Bierz tylko do sekundy N (Ty masz 60)
        """
        self.sr = sr
        self.n_mels = n_mels
        self.window_duration = window_duration
        self.start_time = start_time
        self.end_time = end_time
        
        # Hop length = ile sampli przesunięcia na sekundę
        self.hop_length = int(sr * window_duration)
        self.window_samples = int(sr * window_duration)
        
    def load_audio(self, wav_path: str) -> Tuple[np.ndarray, int]:
        """
        Załaduj plik WAV. Librosa będzie resamplować do self.sr.
        
        Returns:
            (audio_data, sample_rate)
        """
        audio, sr = librosa.load(wav_path, sr=self.sr, mono=True)
        return audio, sr
    
    def extract_mel_spectrogram(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Ekstrakcja mel-spektrogramu z audio segmentu.
        
        Args:
            audio_segment: 1D numpy array audio samples
            
        Returns:
            2D array (n_mels, n_frames) - surowy mel-spektrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio_segment,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512,
            fmin=0,
            fmax=self.sr/2
        )
        return mel_spec
    
    def normalize_mel_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Normalizacja mel-spektrogramu: log-scale + min-max normalizacja.
        
        Args:
            mel_spec: Raw mel-spectrogram (n_mels, n_frames)
            
        Returns:
            Znormalizowany spektrogram [0, 1]
        """
        # Log scale (dodaj epsilon żeby uniknąć log(0))
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Min-max normalizacja do [0, 1]
        mel_min = mel_spec_db.min()
        mel_max = mel_spec_db.max()
        
        if mel_max > mel_min:
            mel_spec_norm = (mel_spec_db - mel_min) / (mel_max - mel_min)
        else:
            mel_spec_norm = np.zeros_like(mel_spec_db)
            
        return mel_spec_norm
    
    def process_single_audio(self, wav_path: str) -> List[np.ndarray]:
        """
        Przetwórz pojedynczy plik WAV na listę znormalizowanych mel-spektrogramów.
        
        Proces:
        1. Załaduj WAV
        2. Wytnij window [start_time, end_time]
        3. Podziel na 1-sekundowe segmenty
        4. Dla każdego: ekstrakcja mel-spec + normalizacja
        
        Args:
            wav_path: Ścieżka do pliku WAV
            
        Returns:
            Lista mel-spektrogramów, każdy shape: (n_mels, n_frames_per_window)
        """
        audio, sr = self.load_audio(wav_path)
        
        # Wytnij window czasu
        start_frame = int(self.start_time * self.sr)
        end_frame = int(self.end_time * self.sr)
        audio_windowed = audio[start_frame:end_frame]
        
        mel_specs = []
        
        # Iteruj po 1-sekundowych segmentach
        for i in range(0, len(audio_windowed), self.hop_length):
            segment = audio_windowed[i:i + self.window_samples]
            
            # Padding jeśli segment zbyt krótki
            if len(segment) < self.window_samples:
                segment = np.pad(segment, (0, self.window_samples - len(segment)))
            
            # Ekstrakcja + normalizacja
            mel_spec = self.extract_mel_spectrogram(segment)
            mel_spec_norm = self.normalize_mel_spectrogram(mel_spec)
            
            mel_specs.append(mel_spec_norm)
        
        return mel_specs
    
    def process_batch(self, wav_folder: str, output_folder: str, 
                     max_files: Optional[int] = None) -> None:
        """
        Przetwórz cały folder WAV plików. Zapisz mel-spektrogramy jako .npy.
        
        Struktura output:
            output_folder/
            ├── song_001.npy  -> shape (n_windows, n_mels, n_frames)
            ├── song_002.npy
            └── ...
        
        Args:
            wav_folder: Folder ze źródłowymi WAV
            output_folder: Folder do zapisu .npy
            max_files: Limit plików do przetworzenia (dla testów)
        """
        os.makedirs(output_folder, exist_ok=True)
        
        wav_files = sorted([f for f in os.listdir(wav_folder) if f.endswith('.wav')])
        
        if max_files:
            wav_files = wav_files[:max_files]
        
        print(f"Processing {len(wav_files)} files...")
        
        for idx, wav_file in enumerate(wav_files):
            wav_path = os.path.join(wav_folder, wav_file)
            
            try:
                mel_specs = self.process_single_audio(wav_path)
                mel_specs_array = np.array(mel_specs)  # (n_windows, n_mels, n_frames)
                
                # Zapisz
                output_name = wav_file.replace('.wav', '.npy')
                output_path = os.path.join(output_folder, output_name)
                np.save(output_path, mel_specs_array)
                
                if (idx + 1) % 100 == 0:
                    print(f"  ✓ Processed {idx + 1}/{len(wav_files)}")
                    
            except Exception as e:
                print(f"  ✗ Error processing {wav_file}: {e}")
        
        print(f"Done! Saved to {output_folder}")


# ============================================================================
# PRZYKŁAD UŻYCIA
# ============================================================================

if __name__ == "__main__":
    # Inicjalizuj processor
    processor = AudioProcessor(
        sr=22050,
        n_mels=128,
        window_duration=1.0,
        start_time=14,
        end_time=60
    )
    
    # Przetwórz batch
    processor.process_batch(
        wav_folder='music/data_scut/data_scut_clean',
        output_folder='music/data_scut/mel_spec_processed',
        max_files=None  # Zmień na np. 10 dla testów
    )