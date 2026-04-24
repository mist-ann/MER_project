"""
Hyperparameter Tuning - Grid Search
Automatycznie testuje różne kombinacje i zapisuje najlepsze modele
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import pickle
from itertools import product
from datetime import datetime
import csv

# Import z poprzedniego skryptu
from lstm_full_training import ImprovedLSTMModel, LSTMTrainer
from baseline_models import MelSpectrogramDataset


def grid_search_lstm(
    data_dir='data/processed',
    hidden_sizes=[128, 256],
    num_layers=[2],
    dropouts=[0.3, 0.5],
    learning_rates=[5e-4, 1e-3],
    batch_sizes=[32],
    epochs=40,
    early_stopping_patience=5,
    results_file='hyperparameter_search_results.csv'
):
    """
    Grid search po hyperparametrach.
    
    Dla każdej kombinacji:
    1. Trenuje model
    2. Ewaluuje na validation secie
    3. Zapisuje najlepszy model
    4. Loguje wyniki
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data (raz)
    print("Loading datasets...")
    train_data = np.load(f'{data_dir}/train_dataset.npz')
    with open(f'{data_dir}/train_mel_specs.pkl', 'rb') as f:
        train_mel_specs = pickle.load(f)
    
    test_data = np.load(f'{data_dir}/test_dataset.npz')
    with open(f'{data_dir}/test_mel_specs.pkl', 'rb') as f:
        test_mel_specs = pickle.load(f)
    
    print(f"✓ Train: {len(train_mel_specs)} samples")
    print(f"✓ Test:  {len(test_mel_specs)} samples\n")
    
    # Create test dataset (constant)
    test_dataset = MelSpectrogramDataset(
        test_mel_specs,
        test_data['valence'],
        test_data['arousal']
    )
    
    # Grid
    param_combinations = list(product(
        hidden_sizes,
        num_layers,
        dropouts,
        learning_rates,
        batch_sizes
    ))
    
    print(f"Total combinations to test: {len(param_combinations)}")
    print(f"Estimated time: ~{len(param_combinations) * epochs / 60:.1f} GPU-hours\n")
    
    # Results
    results = []
    best_val_loss = float('inf')
    best_params = None
    
    # CSV header
    csv_columns = [
        'rank', 'hidden_size', 'num_layers', 'dropout', 'lr', 'batch_size',
        'best_epoch', 'val_loss', 'val_mae_val', 'val_mae_arou',
        'test_mae_val', 'test_mae_arou', 'test_corr_val', 'test_corr_arou',
        'training_time_min', 'checkpoint'
    ]
    
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
    
    # Grid search
    for combo_idx, (hidden_size, num_layers_val, dropout, lr, batch_size) in enumerate(param_combinations):
        
        print(f"\n{'='*70}")
        print(f"Combination {combo_idx + 1}/{len(param_combinations)}")
        print(f"{'='*70}")
        print(f"Params: hidden={hidden_size}, layers={num_layers_val}, "
              f"dropout={dropout}, lr={lr}, batch={batch_size}")
        
        start_time = datetime.now()
        
        # Prepare data
        train_dataset = MelSpectrogramDataset(
            train_mel_specs,
            train_data['valence'],
            train_data['arousal']
        )
        
        train_size = int(0.85 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_ds, val_ds = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Model
        model = ImprovedLSTMModel(
            n_mels=128,
            hidden_size=hidden_size,
            num_layers=num_layers_val,
            dropout=dropout
        )
        
        experiment_name = f"lstm_hs{hidden_size}_l{num_layers_val}_d{dropout:.1f}_lr{lr:.0e}"
        
        # Train
        trainer = LSTMTrainer(
            model,
            device=device,
            lr=lr,
            checkpoint_dir='checkpoints',
            experiment_name=experiment_name
        )
        
        try:
            history = trainer.fit(
                train_loader,
                val_loader,
                epochs=epochs,
                early_stopping_patience=early_stopping_patience
            )
            
            # Best epoch
            best_epoch = np.argmin(history['val_loss'])
            best_val_loss_combo = history['val_loss'][best_epoch]
            best_val_mae_val = history['val_mae_val'][best_epoch]
            best_val_mae_arou = history['val_mae_arou'][best_epoch]
            
            # Test evaluation
            test_metrics, _, _ = trainer.evaluate_test(test_loader)
            
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            
            # Checkpoint path (trainer saves best model there)
            checkpoint_path = os.path.join('checkpoints', f'{experiment_name}_best.pth')

            # Save result
            result = {
                'hidden_size': hidden_size,
                'num_layers': num_layers_val,
                'dropout': dropout,
                'lr': lr,
                'batch_size': batch_size,
                'best_epoch': best_epoch,
                'val_loss': best_val_loss_combo,
                'val_mae_val': best_val_mae_val,
                'val_mae_arou': best_val_mae_arou,
                'test_mae_val': test_metrics['valence_mae'],
                'test_mae_arou': test_metrics['arousal_mae'],
                'test_corr_val': test_metrics['valence_corr'],
                'test_corr_arou': test_metrics['arousal_corr'],
                'training_time_min': elapsed,
                'checkpoint': checkpoint_path
            }
            
            results.append(result)
            
            # Update best
            if best_val_loss_combo < best_val_loss:
                best_val_loss = best_val_loss_combo
                best_params = result
            
            # Print
            print(f"\n✓ Best epoch: {best_epoch}")
            print(f"  Val Loss: {best_val_loss_combo:.4f}")
            print(f"  Val MAE (val/arou): {best_val_mae_val:.4f} / {best_val_mae_arou:.4f}")
            print(f"  Test MAE (val/arou): {test_metrics['valence_mae']:.4f} / {test_metrics['arousal_mae']:.4f}")
            print(f"  Test Corr (val/arou): {test_metrics['valence_corr']:.4f} / {test_metrics['arousal_corr']:.4f}")
            print(f"  Training time: {elapsed:.1f} min")
            
            # Append to CSV
            with open(results_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                row = {
                    'rank': len(results),
                    'hidden_size': hidden_size,
                    'num_layers': num_layers_val,
                    'dropout': dropout,
                    'lr': lr,
                    'batch_size': batch_size,
                    'best_epoch': best_epoch,
                    'val_loss': f"{best_val_loss_combo:.4f}",
                    'val_mae_val': f"{best_val_mae_val:.4f}",
                    'val_mae_arou': f"{best_val_mae_arou:.4f}",
                    'test_mae_val': f"{test_metrics['valence_mae']:.4f}",
                    'test_mae_arou': f"{test_metrics['arousal_mae']:.4f}",
                    'test_corr_val': f"{test_metrics['valence_corr']:.4f}",
                    'test_corr_arou': f"{test_metrics['arousal_corr']:.4f}",
                    'training_time_min': f"{elapsed:.1f}",
                    'checkpoint': checkpoint_path
                }
                writer.writerow(row)
        
        except Exception as e:
            print(f"\n✗ Error: {e}")
            continue
    
    # Summary
    print(f"\n\n{'='*70}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*70}")
    
    # Sort by val_loss
    results_sorted = sorted(results, key=lambda x: x['val_loss'])
    
    print(f"\nTop 10 configurations:")
    for i, res in enumerate(results_sorted[:10]):
        print(f"\n{i+1}. hidden={res['hidden_size']}, layers={res['num_layers']}, "
              f"dropout={res['dropout']}, lr={res['lr']}, batch={res['batch_size']}")
        print(f"   Val Loss: {res['val_loss']:.4f} (epoch {res['best_epoch']})")
        print(f"   Test MAE: {res['test_mae_val']:.4f} (val) / {res['test_mae_arou']:.4f} (arou)")
        print(f"   Test Corr: {res['test_corr_val']:.4f} (val) / {res['test_corr_arou']:.4f} (arou)")
        print(f"   Time: {res['training_time_min']:.1f} min")
    
    # Save best params
    if best_params:
        print(f"\n\nBest Parameters (by validation loss):")
        print(f"  Hidden size: {best_params['hidden_size']}")
        print(f"  Num layers: {best_params['num_layers']}")
        print(f"  Dropout: {best_params['dropout']}")
        print(f"  Learning rate: {best_params['lr']}")
        print(f"  Batch size: {best_params['batch_size']}")
        print(f"  Model: {best_params['checkpoint']}")
        
        # Save as JSON
        with open('best_hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\n✓ Best params saved: best_hyperparameters.json")
        print(f"✓ All results saved: {results_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128])
    parser.add_argument('--num_layers', type=int, nargs='+', default=[2])
    parser.add_argument('--dropouts', type=float, nargs='+', default=[0.5])
    parser.add_argument('--lrs', type=float, nargs='+', default=[1e-3])
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[32])
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=5)
    
    args = parser.parse_args()
    
    grid_search_lstm(
        hidden_sizes=args.hidden_sizes,
        num_layers=args.num_layers,
        dropouts=args.dropouts,
        learning_rates=args.lrs,
        batch_sizes=args.batch_sizes,
        epochs=args.epochs,
        early_stopping_patience=args.patience
    )