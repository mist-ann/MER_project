"""
LSTM Training v2 - Fixed version
- Brak sigmoid w output (pure regression)
- L1Loss (lepszy dla regression niż MSE)
- Proper learning rate scheduling
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from lstm_model_v2 import ImprovedLSTMModel
from baseline_models import MelSpectrogramDataset


class LSTMTrainerV2:
    """Training z L1Loss zamiast MSE."""
    
    def __init__(self, model, device='cpu', lr=1e-3, checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # L1Loss (MAE) zamiast MSE
        # Lepszy dla regression na continuous targets
        self.criterion = nn.L1Loss()
        
        # Learning rate scheduling
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae_val': [],
            'train_mae_arou': [],
            'val_mae_val': [],
            'val_mae_arou': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        """Jeden epoch."""
        self.model.train()
        total_loss = 0
        all_pred = []
        all_target = []
        
        for batch in train_loader:
            mel_specs = batch['mel_spec'].to(self.device)
            valences = batch['valence'].to(self.device)
            arousals = batch['arousal'].to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(mel_specs)
            
            # L1Loss (MAE) na both outputs
            loss = self.criterion(predictions[:, 0], valences) + \
                   self.criterion(predictions[:, 1], arousals)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            all_pred.append(predictions.detach().cpu().numpy())
            all_target.append(torch.stack([valences, arousals], dim=1).cpu().numpy())
        
        train_loss = total_loss / len(train_loader)
        
        all_pred = np.vstack(all_pred)
        all_target = np.vstack(all_target)
        
        # Normalizuj predictions before computing MAE
        from scipy.special import expit
        all_pred_norm = expit(all_pred)  # sigmoid
        
        mae_val = mean_absolute_error(all_target[:, 0], all_pred_norm[:, 0])
        mae_arou = mean_absolute_error(all_target[:, 1], all_pred_norm[:, 1])
        
        return train_loss, mae_val, mae_arou
    
    def validate(self, val_loader):
        """Validation."""
        self.model.eval()
        total_loss = 0
        all_pred = []
        all_target = []
        
        with torch.no_grad():
            for batch in val_loader:
                mel_specs = batch['mel_spec'].to(self.device)
                valences = batch['valence'].to(self.device)
                arousals = batch['arousal'].to(self.device)
                
                predictions = self.model(mel_specs)
                
                loss = self.criterion(predictions[:, 0], valences) + \
                       self.criterion(predictions[:, 1], arousals)
                
                total_loss += loss.item()
                
                all_pred.append(predictions.cpu().numpy())
                all_target.append(torch.stack([valences, arousals], dim=1).cpu().numpy())
        
        val_loss = total_loss / len(val_loader)
        
        all_pred = np.vstack(all_pred)
        all_target = np.vstack(all_target)
        
        # Normalizuj
        from scipy.special import expit
        all_pred_norm = expit(all_pred)
        
        mae_val = mean_absolute_error(all_target[:, 0], all_pred_norm[:, 0])
        mae_arou = mean_absolute_error(all_target[:, 1], all_pred_norm[:, 1])
        
        return val_loss, mae_val, mae_arou
    
    def fit(self, train_loader, val_loader, epochs=100, early_stopping_patience=15):
        """Full training."""
        print(f"\nLSTM Training v2 (L1Loss + Cosine Annealing)")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}\n")
        
        start_time = datetime.now()
        
        for epoch in range(epochs):
            train_loss, train_mae_val, train_mae_arou = self.train_epoch(train_loader)
            val_loss, val_mae_val, val_mae_arou = self.validate(val_loader)
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae_val'].append(train_mae_val)
            self.history['train_mae_arou'].append(train_mae_arou)
            self.history['val_mae_val'].append(val_mae_val)
            self.history['val_mae_arou'].append(val_mae_arou)
            self.history['learning_rates'].append(current_lr)
            
            # Print status every epoch (concise) and a more detailed log every 10 epochs
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

            if (epoch + 1) % 10 == 0:
                print(f"  ({elapsed:.1f} min) Train MAE (val/arou): {train_mae_val:.4f} / {train_mae_arou:.4f}")
                print(f"  Val MAE (val/arou): {val_mae_val:.4f} / {val_mae_arou:.4f}")
                print("\n")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, 'lstm_v2_best.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'val_loss': val_loss
                }, checkpoint_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    print(f"\n⚠️ Early stopping at epoch {epoch + 1}")
                    break
        
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        print(f"\n✓ Training complete! ({elapsed:.1f} min total)")
        
        return self.history


def main(data_dir='data/processed', batch_size=32, epochs=40, lr=1e-3):
    """Main."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    train_data = np.load(f'{data_dir}/train_dataset.npz')
    with open(f'{data_dir}/train_mel_specs.pkl', 'rb') as f:
        train_mel_specs = pickle.load(f)
    
    test_data = np.load(f'{data_dir}/test_dataset.npz')
    with open(f'{data_dir}/test_mel_specs.pkl', 'rb') as f:
        test_mel_specs = pickle.load(f)
    
    print(f"✓ Train: {len(train_mel_specs)}, Test: {len(test_mel_specs)}\n")
    
    # Create dataset
    train_dataset = MelSpectrogramDataset(
        train_mel_specs,
        train_data['valence'],
        train_data['arousal']
    )
    
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_dataset = MelSpectrogramDataset(
        test_mel_specs,
        test_data['valence'],
        test_data['arousal']
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = ImprovedLSTMModel(n_mels=128, hidden_size=128, num_layers=2, dropout=0.5)
    
    # Train
    trainer = LSTMTrainerV2(model, device=device, lr=lr)
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    
    # Test
    print(f"\n{'='*70}")
    print("TEST EVALUATION")
    print(f"{'='*70}\n")
    
    model.eval()
    all_pred = []
    all_target = []
    
    with torch.no_grad():
        for batch in test_loader:
            mel_specs = batch['mel_spec'].to(device)
            valences = batch['valence'].to(device)
            arousals = batch['arousal'].to(device)
            
            predictions = model(mel_specs)
            
            all_pred.append(predictions.cpu().numpy())
            all_target.append(torch.stack([valences, arousals], dim=1).cpu().numpy())
    
    all_pred = np.vstack(all_pred)
    all_target = np.vstack(all_target)
    
    # Normalize predictions
    from scipy.special import expit
    all_pred_norm = expit(all_pred)
    
    # Metrics
    mae_val = mean_absolute_error(all_target[:, 0], all_pred_norm[:, 0])
    mae_arou = mean_absolute_error(all_target[:, 1], all_pred_norm[:, 1])
    rmse_val = np.sqrt(mean_squared_error(all_target[:, 0], all_pred_norm[:, 0]))
    rmse_arou = np.sqrt(mean_squared_error(all_target[:, 1], all_pred_norm[:, 1]))
    
    print(f"Valence: MAE={mae_val:.4f}, RMSE={rmse_val:.4f}")
    print(f"Arousal: MAE={mae_arou:.4f}, RMSE={rmse_arou:.4f}")
    
    # Save
    results = {
        'model': 'LSTM v2 (L1Loss)',
        'valence_mae': float(mae_val),
        'arousal_mae': float(mae_arou),
        'valence_rmse': float(rmse_val),
        'arousal_rmse': float(rmse_arou),
    }
    
    with open('lstm_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to lstm_v2_results.json")


if __name__ == "__main__":
    main(epochs=40, lr=1e-3)