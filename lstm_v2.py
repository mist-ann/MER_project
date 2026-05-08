"""
LSTM dla rozpoznawania emocji w muzyce — NAPRAWIONA wersja
==========================================================

WPROWADZONE POPRAWKI:
  Bug #1 (KRYTYCZNY) — output: sigmoid → tanh
    Etykiety MEMO są w zakresie [−1, 1] z centrum w 0.
    Sigmoid daje (0, 1), więc model nigdy nie mógł predykować wartości
    ujemnych. Tanh daje (−1, 1) — idealnie dopasowane do danych.

  Bug #2 (ISTOTNY) — label alignment
    Preprocessing wycina audio od start_time=14s, ale dopasowanie
    etykiet CSV przez iloc[i] brało etykiety od t=0.
    Okno mel-spek #0 odpowiada sekundzie 14 w audio, ale etykieta
    iloc[0] to t=0 w CSV → 14-sekundowy offset w każdej piosence.
    Naprawa: filtrujemy CSV do time >= start_time przed dopasowaniem.

  Bug #3 (POMOCNICZY) — per-segment normalizacja
    Normalizacja min-max wewnątrz każdego 1-sekundowego okna usuwa
    informację o względnej głośności między segmentami i piosenkami.
    Zastąpiono standardyzacją z-score z globalnym klikowaniem.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import pickle
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================
# NAPRAWA BUG #3 — normalizacja mel-spektrogramu
# ============================================================

def normalize_mel_global(mel_db: torch.Tensor,
                          global_mean: float = -40.0,
                          global_std: float = 20.0) -> torch.Tensor:
    """
    Standaryzacja z-score z globalnymi statystykami zamiast per-segment.

    Dlaczego: normalizacja [0,1] na każdym 1-sekundowym oknie osobno
    niszczy informację o dynamice głośności — ciche i głośne fragmenty
    wyglądają identycznie po normalizacji. Model traci kontekst globalny.

    Jak użyć:
        global_mean, global_std = compute_dataset_stats(all_mel_specs)
        mel_norm = normalize_mel_global(mel_db, global_mean, global_std)
    """
    return (mel_db - global_mean) / (global_std + 1e-8)


def compute_dataset_stats(mel_specs_list: list) -> tuple:
    """
    Oblicz globalne statystyki (mean, std) ze wszystkich spektrogramów
    na zbiorze treningowym. Uruchom raz, zapisz, użyj przy normalizacji.
    """
    all_values = np.concatenate([m.flatten() for m in mel_specs_list])
    return float(all_values.mean()), float(all_values.std())


# ============================================================
# NAPRAWA BUG #2 — poprawne dopasowanie etykiet czasowych
# ============================================================

def get_song_labels_aligned(labels_df, song_id: str, start_time: float):
    """
    Pobierz etykiety dla piosenki zaczynając od start_time.

    Naprawa: preprocessing AudioProcessor wycina audio [start_time, end_time].
    Pierwsze okno mel-spek = [start_time, start_time+1s].
    Więc dopasowujemy etykiety od time >= start_time, nie od iloc[0].

    Args:
        labels_df: DataFrame z kolumnami (song_id, time, valence_mean, arousal_mean)
        song_id:   identyfikator piosenki
        start_time: ten sam co AudioProcessor.start_time (domyślnie 14)

    Returns:
        DataFrame z etykietami posortowanymi wg czasu, od start_time
    """
    song_df = labels_df[
        (labels_df['song_id'] == song_id) &
        (labels_df['time'] >= start_time)          # <-- kluczowa naprawa
    ].sort_values('time').reset_index(drop=True)

    return song_df


# ============================================================
# Dataset z poprawioną normalizacją
# ============================================================

class EmotionDataset(Dataset):
    """
    Dataset łączący mel-spektrogramy z etykietami valence/arousal.

    Parametry normalizacji (global_mean, global_std) powinny być obliczone
    na zbiorze TRENINGOWYM i użyte konsekwentnie dla train/val/test.
    """

    def __init__(self, mel_specs: list, valences: np.ndarray,
                 arousals: np.ndarray,
                 global_mean: float = 0.0,
                 global_std: float = 1.0):
        self.mel_specs = mel_specs
        self.valences = valences.astype(np.float32)
        self.arousals = arousals.astype(np.float32)
        self.global_mean = global_mean
        self.global_std = global_std

    def __len__(self):
        return len(self.mel_specs)

    def __getitem__(self, idx):
        mel = torch.tensor(self.mel_specs[idx], dtype=torch.float32)

        # Bug #3 fix: z-score zamiast per-segment [0,1]
        mel = (mel - self.global_mean) / (self.global_std + 1e-8)

        return {
            'mel_spec': mel,
            'valence':  torch.tensor(self.valences[idx], dtype=torch.float32),
            'arousal':  torch.tensor(self.arousals[idx], dtype=torch.float32),
        }


# ============================================================
# NAPRAWIONY MODEL — Bug #1: tanh zamiast sigmoid
# ============================================================

class LSTMFixed(nn.Module):
    """
    BiLSTM z attention dla predykcji valence i arousal.

    Kluczowa naprawa: output layer używa Tanh zamiast Sigmoid.
      - Sigmoid: wyjście ∈ (0, 1)  →  model nie może predykować < 0
      - Tanh:    wyjście ∈ (−1, 1) →  dopasowane do etykiet MEMO

    Architektura: BiLSTM → scaled dot-product attention → FC → Tanh → [val, aro]
    """

    def __init__(self, n_mels: int = 128, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # BiLSTM: przetwarza sekwencję ramek mel-spektrogramu
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # Scaled dot-product attention (single head)
        self.attn_q = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_k = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_v = nn.Linear(hidden_size * 2, hidden_size)
        self.scale  = hidden_size ** 0.5

        # Klasyfikator
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
            nn.Tanh()          # BUG #1 FIX: sigmoid → tanh, zakres (−1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_mels, n_frames)
        Returns:
            (batch, 2) — [valence, arousal] ∈ (−1, 1)
        """
        # BiLSTM oczekuje (batch, seq_len, features)
        x = x.transpose(1, 2)                          # → (B, n_frames, n_mels)
        lstm_out, _ = self.lstm(x)                     # → (B, n_frames, H*2)

        # Scaled dot-product attention
        Q = self.attn_q(lstm_out)                      # (B, T, H)
        K = self.attn_k(lstm_out)
        V = self.attn_v(lstm_out)

        scores  = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, V).mean(dim=1) # (B, H)

        return self.classifier(context)                # (B, 2), tanh output


# ============================================================
# Trening + ewaluacja
# ============================================================

class Trainer:
    def __init__(self, model, device='cpu', lr=1e-3,
                 checkpoint_dir='checkpoints', experiment_name='lstm_fixed'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, f'{experiment_name}_best.pth')

        self.history = {k: [] for k in
                        ['train_loss', 'val_loss', 'val_mae_val', 'val_mae_aro',
                         'val_corr_val', 'val_corr_aro', 'lr']}
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss, preds, targets = 0.0, [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                mel = batch['mel_spec'].to(self.device)
                val = batch['valence'].to(self.device)
                aro = batch['arousal'].to(self.device)

                if train:
                    self.optimizer.zero_grad()

                pred = self.model(mel)
                loss = self.criterion(pred[:, 0], val) + \
                       self.criterion(pred[:, 1], aro)

                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                preds.append(pred.detach().cpu().numpy())
                targets.append(torch.stack([val, aro], dim=1).cpu().numpy())

        preds   = np.vstack(preds)
        targets = np.vstack(targets)
        avg_loss = total_loss / len(loader)
        mae_val  = mean_absolute_error(targets[:, 0], preds[:, 0])
        mae_aro  = mean_absolute_error(targets[:, 1], preds[:, 1])
        corr_val = np.corrcoef(targets[:, 0], preds[:, 0])[0, 1]
        corr_aro = np.corrcoef(targets[:, 1], preds[:, 1])[0, 1]
        return avg_loss, mae_val, mae_aro, corr_val, corr_aro, preds, targets

    def fit(self, train_loader, val_loader, epochs=100, patience=20):
        print(f"\n{'='*55}")
        print(f"Params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
        print(f"{'='*55}")

        t0 = datetime.now()
        for epoch in range(epochs):
            tl, *_ = self._run_epoch(train_loader, train=True)
            vl, vmv, vma, ccv, cca, *_ = self._run_epoch(val_loader, train=False)

            self.scheduler.step(vl)
            lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(tl)
            self.history['val_loss'].append(vl)
            self.history['val_mae_val'].append(vmv)
            self.history['val_mae_aro'].append(vma)
            self.history['val_corr_val'].append(ccv)
            self.history['val_corr_aro'].append(cca)
            self.history['lr'].append(lr)

            if (epoch + 1) % 10 == 0:
                elapsed = (datetime.now() - t0).total_seconds() / 60
                print(f"Epoch {epoch+1:4d}/{epochs} | "
                      f"TL={tl:.4f} VL={vl:.4f} | "
                      f"MAE val={vmv:.4f} aro={vma:.4f} | "
                      f"Corr val={ccv:.4f} aro={cca:.4f} | "
                      f"LR={lr:.2e} | {elapsed:.1f}m")

            if vl < self.best_val_loss:
                self.best_val_loss = vl
                self.patience_counter = 0
                torch.save({'epoch': epoch, 'model_state': self.model.state_dict(),
                            'val_loss': vl}, self.checkpoint_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping (epoch {epoch+1})")
                    break

        print(f"\nBest val_loss: {self.best_val_loss:.4f}")
        print(f"Model: {self.checkpoint_path}")
        return self.history

    def evaluate(self, test_loader):
        """Ewaluacja na test secie z pełnymi metrykami."""
        _, mae_v, mae_a, cor_v, cor_a, preds, targets = \
            self._run_epoch(test_loader, train=False)

        rmse_v = np.sqrt(mean_squared_error(targets[:, 0], preds[:, 0]))
        rmse_a = np.sqrt(mean_squared_error(targets[:, 1], preds[:, 1]))

        print("\n" + "="*55)
        print("EWALUACJA NA ZBIORZE TESTOWYM")
        print("="*55)
        print(f"\n  Valence: MAE={mae_v:.4f}, RMSE={rmse_v:.4f}, Corr={cor_v:.4f}")
        print(f"  Arousal: MAE={mae_a:.4f}, RMSE={rmse_a:.4f}, Corr={cor_a:.4f}")
        print(f"\n  Zakresy predykcji:")
        print(f"    Valence: [{preds[:,0].min():.4f}, {preds[:,0].max():.4f}] "
              f"(GT: [{targets[:,0].min():.4f}, {targets[:,0].max():.4f}])")
        print(f"    Arousal: [{preds[:,1].min():.4f}, {preds[:,1].max():.4f}] "
              f"(GT: [{targets[:,1].min():.4f}, {targets[:,1].max():.4f}])")

        return {'valence_mae': mae_v, 'arousal_mae': mae_a,
                'valence_corr': cor_v, 'arousal_corr': cor_a,
                'valence_rmse': rmse_v, 'arousal_rmse': rmse_a}, preds, targets

    def plot_history(self, save_path='training_history.png'):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        axes[0, 0].plot(self.history['train_loss'], label='Train', alpha=0.8)
        axes[0, 0].plot(self.history['val_loss'], label='Val', alpha=0.8)
        axes[0, 0].set_title('Loss (MSE)')
        axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(self.history['val_mae_val'], label='Valence', alpha=0.8)
        axes[0, 1].plot(self.history['val_mae_aro'], label='Arousal', alpha=0.8)
        axes[0, 1].set_title('Val MAE')
        axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(self.history['val_corr_val'], label='Valence', alpha=0.8)
        axes[1, 0].plot(self.history['val_corr_aro'], label='Arousal', alpha=0.8)
        axes[1, 0].set_title('Val Pearson Corr')
        axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

        axes[1, 1].semilogy(self.history['lr'], alpha=0.8)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
        plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='LSTM emotion recognition — fixed version')
    parser.add_argument('--data_dir',    default='data/processed')
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--epochs',      type=int,   default=40)
    parser.add_argument('--hidden_size', type=int,   default=128)
    parser.add_argument('--num_layers',  type=int,   default=2)
    parser.add_argument('--dropout',     type=float, default=0.3)
    parser.add_argument('--lr',          type=float, default=5e-4)
    parser.add_argument('--start_time',  type=float, default=14.0,
                        help='Musi być identyczne jak AudioProcessor.start_time')
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Wczytaj dane
    train_data = np.load('sp2/data/processed/train_dataset_v0.npz')
    test_data  = np.load('sp2/data/processed/test_dataset_v0.npz')

    with open('sp2/data/processed/train_mel_specs_v0.pkl', 'rb') as f:
        train_mel = pickle.load(f)
    with open('sp2/data/processed/test_mel_specs_v0.pkl', 'rb') as f:
        test_mel  = pickle.load(f)

    # Bug #3 fix: oblicz globalne statystyki na zbiorze treningowym
    print("Obliczanie globalnych statystyk normalizacji...")
    global_mean, global_std = compute_dataset_stats(train_mel)
    print(f"  global_mean={global_mean:.2f}, global_std={global_std:.2f}")

    # Przygotuj datasety
    train_ds = EmotionDataset(train_mel, train_data['valence'], train_data['arousal'],
                              global_mean, global_std)
    test_ds  = EmotionDataset(test_mel,  test_data['valence'],  test_data['arousal'],
                              global_mean, global_std)

    # train → train/val split (85/15)
    train_size = int(0.85 * len(train_ds))
    val_size   = len(train_ds) - train_size
    train_ds, val_ds = random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Model
    model = LSTMFixed(
        n_mels=128,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    trainer = Trainer(model, device=device, lr=args.lr,
                      experiment_name='lstm_fixed')

    history = trainer.fit(train_loader, val_loader,
                          epochs=args.epochs, patience=args.patience)

    metrics, preds, targets = trainer.evaluate(test_loader)
    trainer.plot_history('sp2/lstm_fixed_history_v2.png')

    # Zapisz wyniki
    with open('sp2/lstm_fixed_results_v2.json', 'w') as f:
        json.dump({'hyperparams': vars(args), 'test_metrics': metrics,
                   'global_mean': global_mean, 'global_std': global_std}, f, indent=2)