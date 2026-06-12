"""
Complete Training Pipeline
- Przepływ treningu z validation setami
- Metric computation (MAE, MSE, correlation)
- Model checkpointing
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path


class Trainer:
    """
    Unified trainer dla modeli PyTorch.
    """
    
    def __init__(self, model, device='cuda', lr=1e-3, checkpoint_dir='checkpoints'):
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # History do monitorowania
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def train_epoch(self, train_loader):
        """Jeden epoch treningu."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            mel_specs = batch['mel_spec'].to(self.device)
            valences = batch['valence'].to(self.device)
            arousals = batch['arousal'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            predictions = self.model(mel_specs)
            
            # Loss - ważona suma dla dwóch outputs
            loss_val = self.criterion(predictions[:, 0], valences)
            loss_arou = self.criterion(predictions[:, 1], arousals)
            loss = loss_val + loss_arou
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Ewaluacja na validation secie."""
        self.model.eval()
        total_loss = 0
        predictions_all = []
        targets_all = []
        
        with torch.no_grad():
            for batch in val_loader:
                mel_specs = batch['mel_spec'].to(self.device)
                valences = batch['valence'].to(self.device)
                arousals = batch['arousal'].to(self.device)
                
                predictions = self.model(mel_specs)
                
                loss_val = self.criterion(predictions[:, 0], valences)
                loss_arou = self.criterion(predictions[:, 1], arousals)
                loss = loss_val + loss_arou
                
                total_loss += loss.item()
                
                # Zbierz predykcje dla metryk
                predictions_all.append(predictions.cpu().numpy())
                targets_all.append(torch.stack([valences, arousals], dim=1).cpu().numpy())
        
        predictions_all = np.vstack(predictions_all)
        targets_all = np.vstack(targets_all)
        
        val_loss = total_loss / len(val_loader)
        metrics = self._compute_metrics(predictions_all, targets_all)
        
        return val_loss, metrics, predictions_all, targets_all
    
    def _compute_metrics(self, predictions, targets):
        """
        Compute detailed metrics.
        
        Args:
            predictions: (n_samples, 2) - [valence, arousal]
            targets: (n_samples, 2)
        
        Returns:
            Dict z metrykami
        """
        metrics = {}
        
        # Valence
        mae_val = mean_absolute_error(targets[:, 0], predictions[:, 0])
        rmse_val = np.sqrt(mean_squared_error(targets[:, 0], predictions[:, 0]))
        corr_val = np.corrcoef(targets[:, 0], predictions[:, 0])[0, 1]
        
        metrics['valence_mae'] = mae_val
        metrics['valence_rmse'] = rmse_val
        metrics['valence_corr'] = corr_val
        
        # Arousal
        mae_arou = mean_absolute_error(targets[:, 1], predictions[:, 1])
        rmse_arou = np.sqrt(mean_squared_error(targets[:, 1], predictions[:, 1]))
        corr_arou = np.corrcoef(targets[:, 1], predictions[:, 1])[0, 1]
        
        metrics['arousal_mae'] = mae_arou
        metrics['arousal_rmse'] = rmse_arou
        metrics['arousal_corr'] = corr_arou
        
        # Average
        metrics['avg_mae'] = (mae_val + mae_arou) / 2
        metrics['avg_rmse'] = (rmse_val + rmse_arou) / 2
        metrics['avg_corr'] = (corr_val + corr_arou) / 2
        
        return metrics
    
    def fit(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        """
        Pełny training loop.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics, _, _ = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Log
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  Val Metrics:")
                print(f"    - Valence MAE:  {val_metrics['valence_mae']:.4f}")
                print(f"    - Arousal MAE:  {val_metrics['arousal_mae']:.4f}")
                print(f"    - Avg Corr:     {val_metrics['avg_corr']:.4f}")
            
            # Early stopping + checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Zapisz best model
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n⚠️  Early stopping at epoch {epoch + 1}")
                    break
        
        return self.history
    
    def evaluate_on_test(self, test_loader):
        """
        Ewaluacja na test secie i Return detailed metrics.
        """
        self.model.eval()
        predictions_all = []
        targets_all = []
        
        with torch.no_grad():
            for batch in test_loader:
                mel_specs = batch['mel_spec'].to(self.device)
                valences = batch['valence'].to(self.device)
                arousals = batch['arousal'].to(self.device)
                
                predictions = self.model(mel_specs)
                
                predictions_all.append(predictions.cpu().numpy())
                targets_all.append(torch.stack([valences, arousals], dim=1).cpu().numpy())
        
        predictions_all = np.vstack(predictions_all)
        targets_all = np.vstack(targets_all)
        
        metrics = self._compute_metrics(predictions_all, targets_all)
        
        return metrics, predictions_all, targets_all
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot loss i metrics during training."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss Over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid()
        
        # Valence MAE
        val_maes = [m['valence_mae'] for m in self.history['val_metrics']]
        axes[0, 1].plot(val_maes, label='Valence MAE')
        axes[0, 1].set_title('Valence MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].grid()
        
        # Arousal MAE
        arou_maes = [m['arousal_mae'] for m in self.history['val_metrics']]
        axes[1, 0].plot(arou_maes, label='Arousal MAE')
        axes[1, 0].set_title('Arousal MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].grid()
        
        # Correlation
        avg_corrs = [m['avg_corr'] for m in self.history['val_metrics']]
        axes[1, 1].plot(avg_corrs, label='Avg Correlation')
        axes[1, 1].set_title('Average Correlation')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].grid()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training plot to {save_path}")
        plt.close()


def create_train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.15):
    """
    Podziel dataset na train/val/test.
    
    Args:
        dataset: PyTorch Dataset
        train_ratio: 0.7 = 70% train
        val_ratio: 0.15 = 15% val, 15% test
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    n = len(dataset)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    test_size = n - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset, test_dataset


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main(
    train_dataset_path='data/processed/train_dataset.npz',
    test_dataset_path='data/processed/test_dataset.npz',
    model_type='lstm',  # 'lstm', 'cnn', or 'rf'
    batch_size=32,
    epochs=100,
    lr=1e-3
):
    """
    Kompletny training pipeline.
    
    Usage:
        python train.py --model_type lstm --batch_size 32 --epochs 100
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Załaduj dane
    print("Loading datasets...")
    import pickle
    
    # Train data
    train_data = np.load(train_dataset_path)
    with open('data/processed/train_mel_specs.pkl', 'rb') as f:
        train_mel_specs = pickle.load(f)
    
    # Test data
    test_data = np.load(test_dataset_path)
    with open('data/processed/test_mel_specs.pkl', 'rb') as f:
        test_mel_specs = pickle.load(f)
    
    print(f"✓ Train: {len(train_mel_specs)} samples")
    print(f"✓ Test:  {len(test_mel_specs)} samples\n")
    
    # Stwórz Dataset
    from baseline_models import MelSpectrogramDataset
    
    train_dataset = MelSpectrogramDataset(
        train_mel_specs,
        train_data['valence'],
        train_data['arousal']
    )
    
    test_dataset = MelSpectrogramDataset(
        test_mel_specs,
        test_data['valence'],
        test_data['arousal']
    )
    
    # Split train na train/val
    train_dataset, val_dataset, _ = create_train_val_test_split(
        train_dataset, train_ratio=0.85, val_ratio=0.15
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model selection
    if model_type.lower() == 'lstm':
        from baseline_models import LSTMBaseline
        model = LSTMBaseline(n_mels=128, hidden_size=128, num_layers=2)
        print(f"Using LSTM model\n")
    elif model_type.lower() == 'cnn':
        from baseline_models import CNNBaseline
        model = CNNBaseline(n_mels=128)
        print(f"Using CNN model\n")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Train
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    trainer = Trainer(model, device=device, lr=lr, checkpoint_dir='checkpoints')
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    
    # Evaluate na test secie
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    test_metrics, test_pred, test_targets = trainer.evaluate_on_test(test_loader)
    
    print(f"\nTest Metrics:")
    print(f"  Valence: MAE={test_metrics['valence_mae']:.4f}, "
          f"RMSE={test_metrics['valence_rmse']:.4f}, "
          f"Corr={test_metrics['valence_corr']:.4f}")
    print(f"  Arousal: MAE={test_metrics['arousal_mae']:.4f}, "
          f"RMSE={test_metrics['arousal_rmse']:.4f}, "
          f"Corr={test_metrics['arousal_corr']:.4f}")
    
    # Zapisz wyniki
    results = {
        'model_type': model_type,
        'test_metrics': test_metrics,
        'hyperparams': {
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr
        }
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot
    trainer.plot_training_history('training_history.png')
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='lstm', 
                       choices=['lstm', 'cnn'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    
    main(
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )