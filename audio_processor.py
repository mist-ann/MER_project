import os
import numpy as np
import torch
import torchaudio
from typing import Tuple, List, Optional


class AudioProcessor:
    def __init__(
        self,
        sr: int = 22050,
        n_mels: int = 128,
        window_duration: float = 1.0,
        start_time: float = 14,
        end_time: float = 60
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.window_duration = window_duration
        self.start_time = start_time
        self.end_time = end_time

        self.window_samples = int(sr * window_duration)
        self.hop_length = self.window_samples

        # self.mel_transform = None
        # self.resampler = None

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=2048,
            hop_length=512,
            n_mels=self.n_mels
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # transforms
        # self.resampler = torchaudio.transforms.Resample(orig_freq=None, new_freq=sr)
        # self.mel_transform = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=sr,
        #     n_fft=2048,
        #     hop_length=512,
        #     n_mels=n_mels
        # )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def load_audio(self, wav_path: str) -> Tuple[torch.Tensor, int]:
        waveform, sr = torchaudio.load(wav_path)

        # mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)
            self.resampler = torchaudio.transforms.Resample(sr, self.sr)
            sr = self.sr
        else:
            self.resampler = torchaudio.transforms.Resample(sr, self.sr)

        return waveform.squeeze(0), sr

    def extract_mel_spectrogram(self, segment: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(segment)
        mel_db = self.db_transform(mel)

        # normalize [0,1]
        mel_min = mel_db.min()
        mel_max = mel_db.max()
        if mel_max > mel_min:
            mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
        else:
            mel_norm = torch.zeros_like(mel_db)

        return mel_norm

    def process_single_audio(self, wav_path: str) -> List[np.ndarray]:
        audio, sr = self.load_audio(wav_path)

        start = int(self.start_time * self.sr)
        end = int(self.end_time * self.sr)
        audio = audio[start:end]

        mel_specs = []

        for i in range(0, len(audio), self.hop_length):
            segment = audio[i:i + self.window_samples]

            if len(segment) < self.window_samples:
                pad = self.window_samples - len(segment)
                segment = torch.nn.functional.pad(segment, (0, pad))

            mel = self.extract_mel_spectrogram(segment)
            mel_specs.append(mel.numpy())

        return mel_specs

    def process_batch(
        self,
        wav_folder: str,
        output_folder: str,
        max_files: Optional[int] = None
    ):
        os.makedirs(output_folder, exist_ok=True)

        wav_files = sorted([f for f in os.listdir(wav_folder) if f.endswith(".wav")])
        if max_files:
            wav_files = wav_files[:max_files]

        print(f"Processing {len(wav_files)} files...")

        for idx, wav_file in enumerate(wav_files):
            wav_path = os.path.join(wav_folder, wav_file)

            try:
                mel_specs = self.process_single_audio(wav_path)
                mel_specs_array = np.array(mel_specs)

                output_name = wav_file.replace(".wav", ".npy")
                output_path = os.path.join(output_folder, output_name)

                np.save(output_path, mel_specs_array)

                if (idx + 1) % 10 == 0:
                    print(f"  ✓ {idx + 1}/{len(wav_files)}")

            except Exception as e:
                print(f"  ✗ Error processing {wav_file}: {e}")

        print(f"Done! Saved to {output_folder}")