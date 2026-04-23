"""
Baseline & Starter Models
- CNN shallow (diagnostic)
- Random Forest (handcrafted features)
- LSTM (main model)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import librosa


# ============================================================================
# HANDCRAFTED FEATURES - dla Random Forest baseline
# ============================================================================

def extract_features_from_melspec(mel_spec: np.ndarray) -> np.ndarray:
    """
    Ekstrakcja handcrafted features z mel-spektrogramu.
    
    Args:
        mel_spec: (n_mels, n_frames)
        
    Returns:
        Features vector (n_features,)
    """
    features = []
    
    # 1. Mean energy per frequency band
    mean_energy = mel_spec.mean(axis=1)  # (n_mels,)
    features.extend(mean_energy)
    
    # 2. Std energy per frequency band
    std_energy = mel_spec.std(axis=1)
    features.extend(std_energy)
    
    # 3. Overall energy (mean across all bins)
    overall_energy = mel_spec.mean()
    features.append(overall_energy)
    
    # 4. Spectral centroid (weighted average frequency)
    # Bardziej wysokie częstotliwości = wyższa wartość
    n_mels = mel_spec.shape[0]
    freqs = np.linspace(0, 1, n_mels)
    spectral_centroid = np.sum(mel_spec * freqs[:, np.newaxis]) / mel_spec.sum()
    features.append(spectral_centroid)
    
    # 5. Spectral spread (jak rozproszone energie)
    spectral_spread = np.sqrt(
        np.sum(mel_spec * (freqs[:, np.newaxis] - spectral_centroid)**2) / mel_spec.sum()
    )
    features.append(spectral_spread)
    
    # 6. Temporal features (zmiana energii w czasie)
    energy_over_time = mel_spec.mean(axis=0)  # (n_frames,)
    features.append(energy_over_time.mean())
    features.append(energy_over_time.std())
    
    return np.array(features)


# ============================================================================
# PYTORCH MODELS
# ============================================================================

class MelSpectrogramDataset(Dataset):
    """PyTorch Dataset dla mel-spektrogramów."""
    
    def __init__(self, mel_specs: list, valences: np.ndarray, 
                 arousals: np.ndarray, pad_to: int = None):
        """
        Args:
            mel_specs: Lista mel-spektrogramów (różne rozmiary)
            valences: (n_samples,) array
            arousals: (n_samples,) array
            pad_to: Jeśli nie None, pad wszystkie spektrogramy do tego rozmiaru
        """
        self.mel_specs = mel_specs
        self.valences = torch.FloatTensor(valences)
        self.arousals = torch.FloatTensor(arousals)
        self.pad_to = pad_to
    
    def __len__(self):
        return len(self.mel_specs)
    
    def __getitem__(self, idx):
        mel_spec = self.mel_specs[idx]  # (n_mels, n_frames)
        
        # Convert to tensor
        mel_spec = torch.FloatTensor(mel_spec)
        
        # Padding jeśli trzeba (dla CNN)
        if self.pad_to is not None:
            if mel_spec.shape[1] < self.pad_to:
                pad_amount = self.pad_to - mel_spec.shape[1]
                mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_amount))
            else:
                mel_spec = mel_spec[:, :self.pad_to]
        
        return {
            'mel_spec': mel_spec,
            'valence': self.valences[idx],
            'arousal': self.arousals[idx]
        }


class CNNBaseline(nn.Module):
    """
    Prosty CNN - diagnostic baseline.
    Input: (batch, n_mels, n_frames) e.g., (32, 128, 128)
    Output: (batch, 2) - [valence, arousal]
    """
    
    def __init__(self, n_mels: int = 128):
        super().__init__()
        
        # Conv layers
        self.conv1 = nn.Conv1d(n_mels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Global average pooling + FC
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)  # valence + arousal
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_mels, n_frames)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        # Global average pooling
        x = x.mean(dim=2)  # (batch, 256)
        
        # FC
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x  # (batch, 2)


class LSTMBaseline(nn.Module):
    """
    BiLSTM model dla emotion recognition.
    
    Obsługuje zmienne długości input (sequence length może się różnić).
    Input: (batch, n_mels, n_frames)
    Output: (batch, 2) - [valence, arousal]
    """
    
    def __init__(self, n_mels: int = 128, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM - n_mels to n_features for LSTM
        # Traktujemy każdy mel-bin jako feature
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention (opcjonalne, ale pomaga)
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # FC layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 2)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_mels, n_frames)
        
        Returns:
            (batch, 2)
        """
        # Transpose dla LSTM: (batch, n_frames, n_mels)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, n_frames, hidden_size*2)
        # h_n: (num_layers*2, batch, hidden_size)
        
        # Attention-weighted sum over frames
        attention_weights = self.attention(lstm_out)  # (batch, n_frames, 1)
        attention_weights = self.softmax(attention_weights)  # Normalize
        context = (lstm_out * attention_weights).sum(dim=1)  # (batch, hidden_size*2)
        
        # FC
        x = self.fc1(context)
        x = self.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)  # (batch, 2)
        
        return x


class RandomForestBaseline:
    """
    Random Forest baseline - handcrafted features.
    Dla szybkiego diagnostyka i porównania.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 20):
        self.rf_valence = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.rf_arousal = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
    
    def extract_all_features(self, mel_specs: list) -> np.ndarray:
        """
        Ekstrakcja features dla wszystkich spektrogramów.
        
        Returns:
            (n_samples, n_features)
        """
        features = []
        for mel_spec in mel_specs:
            feat = extract_features_from_melspec(mel_spec)
            features.append(feat)
        
        return np.array(features)
    
    def fit(self, mel_specs: list, valences: np.ndarray, 
            arousals: np.ndarray) -> None:
        """Trenuj RF modele."""
        X = self.extract_all_features(mel_specs)
        X = self.scaler.fit_transform(X)
        
        self.rf_valence.fit(X, valences)
        self.rf_arousal.fit(X, arousals)
        
        print("✓ Random Forest models trained")
        print(f"  Valence R² = {self.rf_valence.score(X, valences):.3f}")
        print(f"  Arousal R² = {self.rf_arousal.score(X, arousals):.3f}")
    
    def predict(self, mel_specs: list):
        """
        Returns:
            (valences, arousals) każdy shape (n_samples,)
        """
        X = self.extract_all_features(mel_specs)
        X = self.scaler.transform(X)
        
        valences = self.rf_valence.predict(X)
        arousals = self.rf_arousal.predict(X)
        
        return valences, arousals


# ============================================================================
# TRAINING UTILS
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Jeden epoch treningu."""
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        mel_specs = batch['mel_spec'].to(device)
        valences = batch['valence'].to(device)
        arousals = batch['arousal'].to(device)
        
        # Forward
        optimizer.zero_grad()
        predictions = model(mel_specs)
        
        # Loss (MSE dla obu outputs)
        loss_val = criterion(predictions[:, 0], valences)
        loss_arou = criterion(predictions[:, 1], arousals)
        loss = loss_val + loss_arou
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def eval_epoch(model, val_loader, criterion, device):
    """Ewaluacja."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            mel_specs = batch['mel_spec'].to(device)
            valences = batch['valence'].to(device)
            arousals = batch['arousal'].to(device)
            
            predictions = model(mel_specs)
            
            loss_val = criterion(predictions[:, 0], valences)
            loss_arou = criterion(predictions[:, 1], arousals)
            loss = loss_val + loss_arou
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


# ============================================================================
# PRZYKŁAD UŻYCIA
# ============================================================================

if __name__ == "__main__":
    # Pseudo-data dla testów
    n_samples = 100
    mel_specs = [np.random.randn(128, 100) for _ in range(n_samples)]
    valences = np.random.uniform(0, 1, n_samples)
    arousals = np.random.uniform(0, 1, n_samples)
    
    # Test Random Forest
    print("=" * 60)
    print("RANDOM FOREST BASELINE")
    print("=" * 60)
    rf = RandomForestBaseline()
    rf.fit(mel_specs[:80], valences[:80], arousals[:80])
    pred_val, pred_arou = rf.predict(mel_specs[80:])
    print(f"Predictions shape: {pred_val.shape}")
    
    # Test CNN
    print("\n" + "=" * 60)
    print("CNN BASELINE")
    print("=" * 60)
    dataset = MelSpectrogramDataset(mel_specs, valences, arousals, pad_to=100)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = CNNBaseline(n_mels=128).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    
    train_loss = train_epoch(cnn, dataloader, optimizer, criterion, device)
    print(f"✓ CNN: 1 epoch loss = {train_loss:.4f}")
    
    # Test LSTM
    print("\n" + "=" * 60)
    print("LSTM BASELINE")
    print("=" * 60)
    lstm = LSTMBaseline(n_mels=128, hidden_size=128).to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    
    train_loss = train_epoch(lstm, dataloader, optimizer, criterion, device)
    print(f"✓ LSTM: 1 epoch loss = {train_loss:.4f}")