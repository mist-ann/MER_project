"""
Full LSTM Training with Validation, Early Stopping & Model Saving
Optimized dla uczenia przez noc
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
from pathlib import Path


class ImprovedLSTMModel(nn.Module):
    """
    Better LSTM architecture z:
    - Wyższą capacity (więcej paramtrów)
    - Dropout dla regularizacji
    - BatchNorm gdzie sensowne
    - Multi-head attention (simplified)
    """
    
    def __init__(self, n_mels: int = 128, hidden_size: int = 256, 
                 num_layers: int = 3, dropout: float = 0.4):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM - dwa directiony
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-head attention (simplified - single head)
        self.attention_query = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_key = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_value = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_scale = hidden_size ** 0.5
        
        # FC layers z dropout
        self.fc1 = nn.Linear(hidden_size, 512)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout)
        
        # Output layer - 2 values (valence, arousal)
        self.output = nn.Linear(128, 2)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_mels, n_frames)
        Returns:
            (batch, 2)
        """
        # Transpose dla LSTM: (batch, n_frames, n_mels)
        x = x.transpose(1, 2)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, n_frames, hidden_size*2)
        
        # Scaled dot-product attention
        Q = self.attention_query(lstm_out)  # (batch, n_frames, hidden_size)
        K = self.attention_key(lstm_out)
        V = self.attention_value(lstm_out)
        
        # Attention weights
        scores = torch.matmul(Q, K.transpose(1, 2)) / self.attention_scale
        # scores: (batch, n_frames, n_frames)
        
        attention_weights = torch.softmax(scores, dim=-1)
        # (batch, n_frames, n_frames)
        
        context = torch.matmul(attention_weights, V)
        # (batch, n_frames, hidden_size)
        
        # Pool: average over time
        context = context.mean(dim=1)  # (batch, hidden_size)
        
        # FC layers
        x = self.fc1(context)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.output(x)  # (batch, 2)
        x = self.sigmoid(x)
        return x


class LSTMTrainer:
    """
    Full training loop z:
    - Validation set
    - Early stopping
    - Model checkpointing
    - Metrics logging
    """
    
    def __init__(self, model, device='cpu', lr=1e-3, 
                 checkpoint_dir='checkpoints', experiment_name='lstm'):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Learning rate scheduler - zmniejsz LR gdy się plateau
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        # )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae_val': [],
            'train_mae_arou': [],
            'val_mae_val': [],
            'val_mae_arou': [],
            'val_corr_val': [],
            'val_corr_arou': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        """Jeden epoch treningu."""
        self.model.train()
        total_loss = 0
        all_pred = []
        all_target = []
        
        for batch_idx, batch in enumerate(train_loader):
            mel_specs = batch['mel_spec'].to(self.device)
            valences = batch['valence'].to(self.device)
            arousals = batch['arousal'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            predictions = self.model(mel_specs)
            
            # Loss - suma obu outputs
            loss = self.criterion(predictions[:, 0], valences) + \
                   self.criterion(predictions[:, 1], arousals)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            all_pred.append(predictions.detach().cpu().numpy())
            all_target.append(torch.stack([valences, arousals], dim=1).cpu().numpy())
        
        train_loss = total_loss / len(train_loader)
        
        # Metrics
        all_pred = np.vstack(all_pred)
        all_target = np.vstack(all_target)
        
        mae_val = mean_absolute_error(all_target[:, 0], all_pred[:, 0])
        mae_arou = mean_absolute_error(all_target[:, 1], all_pred[:, 1])
        
        return train_loss, mae_val, mae_arou
    
    def validate(self, val_loader):
        """Ewaluacja na validation secie."""
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
        
        # Metrics
        all_pred = np.vstack(all_pred)
        all_target = np.vstack(all_target)
        
        mae_val = mean_absolute_error(all_target[:, 0], all_pred[:, 0])
        mae_arou = mean_absolute_error(all_target[:, 1], all_pred[:, 1])
        
        corr_val = np.corrcoef(all_target[:, 0], all_pred[:, 0])[0, 1]
        corr_arou = np.corrcoef(all_target[:, 1], all_pred[:, 1])[0, 1]
        
        return val_loss, mae_val, mae_arou, corr_val, corr_arou
    
    def fit(self, train_loader, val_loader, epochs=200, early_stopping_patience=20):
        """
        Full training loop.
        """
        print(f"\n{'='*60}")
        print(f"LSTM Training - {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"{'='*60}\n")
        
        start_time = datetime.now()
        
        for epoch in range(epochs):
            # Training
            train_loss, train_mae_val, train_mae_arou = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_mae_val, val_mae_arou, val_corr_val, val_corr_arou = \
                self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # History
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae_val'].append(train_mae_val)
            self.history['train_mae_arou'].append(train_mae_arou)
            self.history['val_mae_val'].append(val_mae_val)
            self.history['val_mae_arou'].append(val_mae_arou)
            self.history['val_corr_val'].append(val_corr_val)
            self.history['val_corr_arou'].append(val_corr_arou)
            self.history['learning_rates'].append(current_lr)
            
            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                print(f"\nEpoch {epoch + 1}/{epochs} ({elapsed:.1f} min)")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  Train MAE (val/arou): {train_mae_val:.4f} / {train_mae_arou:.4f}")
                print(f"  Val MAE (val/arou):   {val_mae_val:.4f} / {val_mae_arou:.4f}")
                print(f"  Val Corr (val/arou):  {val_corr_val:.4f} / {val_corr_arou:.4f}")
                print(f"  LR: {current_lr:.6f}")
            
            # Early stopping + checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, 
                    f'{self.experiment_name}_best.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"  → Saved best model: {checkpoint_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    print(f"\n⚠️ Early stopping at epoch {epoch + 1}")
                    break
        
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        print(f"\n✓ Training complete! ({elapsed:.1f} min total)")
        
        return self.history
    
    def evaluate_test(self, test_loader):
        """Ewaluacja na test secie."""
        self.model.eval()
        all_pred = []
        all_target = []
        
        with torch.no_grad():
            for batch in test_loader:
                mel_specs = batch['mel_spec'].to(self.device)
                valences = batch['valence'].to(self.device)
                arousals = batch['arousal'].to(self.device)
                
                predictions = self.model(mel_specs)
                
                all_pred.append(predictions.cpu().numpy())
                all_target.append(torch.stack([valences, arousals], dim=1).cpu().numpy())
        
        all_pred = np.vstack(all_pred)
        all_target = np.vstack(all_target)
        
        # Compute metrics
        metrics = {
            'valence_mae': mean_absolute_error(all_target[:, 0], all_pred[:, 0]),
            'arousal_mae': mean_absolute_error(all_target[:, 1], all_pred[:, 1]),
            'valence_rmse': np.sqrt(mean_squared_error(all_target[:, 0], all_pred[:, 0])),
            'arousal_rmse': np.sqrt(mean_squared_error(all_target[:, 1], all_pred[:, 1])),
            'valence_corr': np.corrcoef(all_target[:, 0], all_pred[:, 0])[0, 1],
            'arousal_corr': np.corrcoef(all_target[:, 1], all_pred[:, 1])[0, 1],
        }
        
        return metrics, all_pred, all_target
    
    def plot_history(self, save_path='lstm_training_history.png'):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train', alpha=0.7)
        axes[0, 0].plot(self.history['val_loss'], label='Val', alpha=0.7)
        axes[0, 0].set_title('Loss Over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # MAE
        axes[0, 1].plot(self.history['val_mae_val'], label='Valence', alpha=0.7)
        axes[0, 1].plot(self.history['val_mae_arou'], label='Arousal', alpha=0.7)
        axes[0, 1].set_title('Validation MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Correlation
        axes[1, 0].plot(self.history['val_corr_val'], label='Valence', alpha=0.7)
        axes[1, 0].plot(self.history['val_corr_arou'], label='Arousal', alpha=0.7)
        axes[1, 0].set_title('Validation Correlation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Learning rate
        axes[1, 1].semilogy(self.history['learning_rates'], alpha=0.7)
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot: {save_path}")
        plt.close()


def main(data_dir='data/processed', batch_size=32, epochs=200, 
         hidden_size=256, num_layers=3, dropout=0.4, lr=1e-3):
    """Main training script."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading datasets...")
    train_data = np.load(f'{data_dir}/train_dataset.npz')
    with open(f'{data_dir}/train_mel_specs.pkl', 'rb') as f:
        train_mel_specs = pickle.load(f)
    
    test_data = np.load(f'{data_dir}/test_dataset.npz')
    with open(f'{data_dir}/test_mel_specs.pkl', 'rb') as f:
        test_mel_specs = pickle.load(f)
    
    print(f"✓ Train: {len(train_mel_specs)} samples")
    print(f"✓ Test:  {len(test_mel_specs)} samples\n")
    
    # Create dataset
    from baseline_models import MelSpectrogramDataset
    
    train_dataset = MelSpectrogramDataset(
        train_mel_specs,
        train_data['valence'],
        train_data['arousal']
    )
    
    # Split train→train/val (85/15)
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_dataset = MelSpectrogramDataset(
        test_mel_specs,
        test_data['valence'],
        test_data['arousal']
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model
    model = ImprovedLSTMModel(
        n_mels=128,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Train
    trainer = LSTMTrainer(
        model, 
        device=device, 
        lr=lr,
        checkpoint_dir='checkpoints',
        experiment_name='lstm_improved'
    )
    
    history = trainer.fit(train_loader, val_loader, epochs=epochs, early_stopping_patience=20)
    
    # Evaluate
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}")
    
    test_metrics, test_pred, test_target = trainer.evaluate_test(test_loader)
    
    print(f"\nTest Metrics:")
    print(f"  Valence: MAE={test_metrics['valence_mae']:.4f}, "
          f"RMSE={test_metrics['valence_rmse']:.4f}, "
          f"Corr={test_metrics['valence_corr']:.4f}")
    print(f"  Arousal: MAE={test_metrics['arousal_mae']:.4f}, "
          f"RMSE={test_metrics['arousal_rmse']:.4f}, "
          f"Corr={test_metrics['arousal_corr']:.4f}")
    
    # Save results
    results = {
        'model_type': 'lstm_improved',
        'hyperparameters': {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': lr,
            'batch_size': batch_size,
            'epochs_trained': len(history['train_loss'])
        },
        'test_metrics': test_metrics
    }
    
    with open('lstm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot
    trainer.plot_history('lstm_training_history.png')
    
    print("\n✓ Training complete!")
    print("  - Model saved: checkpoints/lstm_improved_best.pth")
    print("  - Results: lstm_results.json")
    print("  - Plot: lstm_training_history.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr
    )