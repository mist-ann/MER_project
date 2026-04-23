#!/usr/bin/env python3
"""
Quick Start / Testing Script
Testuj całą pipeline na małym subsecie (10 piosenek = szybko)
"""

import os
import sys
import numpy as np
from audio_processor import AudioProcessor
from data_pipeline import DataPipeline, DataLoader
from baseline_models import RandomForestBaseline, MelSpectrogramDataset, LSTMBaseline
import torch
from torch.utils.data import DataLoader as TorchDataLoader

# os.environ["NUMBA_DISABLE_COVERAGE"] = "1"


def step_1_audio_processing():
    """STEP 1: WAV → Mel-spectrograms (TEST on 10 files)"""
    print("\n" + "="*60)
    print("STEP 1: Audio Processing (WAV → Mel-Specs)")
    print("="*60)
    
    processor = AudioProcessor(
        sr=22050,
        n_mels=128,
        window_duration=1.0,
        start_time=14,
        end_time=60
    )
    
    print("\nProcessing first 10 WAV files for testing...")
    processor.process_batch(
        wav_folder='/home/mist/Documents/datasets/MusicRawData',
        output_folder='/home/mist/Documents/datasets/mel_spec_processed',
        max_files= None
    )
    
    print("✓ STEP 1 Complete!")


def step_2_data_pipeline():
    """STEP 2: Create train/test split + windowing"""
    print("\n" + "="*60)
    print("STEP 2: Data Pipeline (Split + Windowing)")
    print("="*60)
    
    pipeline = DataPipeline(
        csv_path='sp2/labels.csv',
        mel_spec_folder='/home/mist/Documents/datasets/mel_spec_processed',
        test_size=0.2,
        random_state=42
    )
    
    print("\nPreparing dataset...")
    train_dataset, test_dataset = pipeline.prepare_pipeline(
        output_dir='sp2/data/processed'
    )
    
    print("✓ STEP 2 Complete!")
    
    return train_dataset, test_dataset


def step_3_random_forest_baseline():
    """STEP 3a: Random Forest Baseline (quick & dirty)"""
    print("\n" + "="*60)
    print("STEP 3a: Random Forest Baseline")
    print("="*60)
    
    # Załaduj dane
    train_loader = DataLoader('sp2/data/processed', split='train')
    test_loader = DataLoader('sp2/data/processed', split='test')
    
    print(f"\nTrain: {len(train_loader)} samples")
    print(f"Test: {len(test_loader)} samples")
    
    # Trenuj
    print("\nTraining Random Forest...")
    rf = RandomForestBaseline(n_estimators=50, max_depth=15)  # Mniej estimators dla szybkości
    rf.fit(
        train_loader.mel_specs,
        train_loader.valences,
        train_loader.arousals
    )
    
    # Predykcja
    print("\nPredicting on test set...")
    pred_val, pred_arou = rf.predict(test_loader.mel_specs)
    
    # Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae_val = mean_absolute_error(test_loader.valences, pred_val)
    mae_arou = mean_absolute_error(test_loader.arousals, pred_arou)
    rmse_val = np.sqrt(mean_squared_error(test_loader.valences, pred_val))
    rmse_arou = np.sqrt(mean_squared_error(test_loader.arousals, pred_arou))
    
    print(f"\n✓ Random Forest Results:")
    print(f"  Valence: MAE={mae_val:.4f}, RMSE={rmse_val:.4f}")
    print(f"  Arousal: MAE={mae_arou:.4f}, RMSE={rmse_arou:.4f}")
    
    return {'val_mae': mae_val, 'arou_mae': mae_arou}


def step_4_lstm_training():
    """STEP 3b: LSTM Training (more complex, educational)"""
    print("\n" + "="*60)
    print("STEP 3b: LSTM Training")
    print("="*60)
    
    import pickle
    
    # Załaduj dane
    print("\nLoading data...")
    train_data = np.load('sp2/data/processed/train_dataset.npz')
    with open('sp2/data/processed/train_mel_specs.pkl', 'rb') as f:
        train_mel_specs = pickle.load(f)
    
    test_data = np.load('sp2/data/processed/test_dataset.npz')
    with open('sp2/data/processed/test_mel_specs.pkl', 'rb') as f:
        test_mel_specs = pickle.load(f)
    
    print(f"Train: {len(train_mel_specs)} samples")
    print(f"Test: {len(test_mel_specs)} samples")
    
    # Stwórz Dataset
    print("\nCreating PyTorch datasets...")
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
    
    # DataLoaders
    train_loader = TorchDataLoader(
        train_dataset, 
        batch_size=16,  # Mniejszy batch dla szybkości
        shuffle=True
    )
    
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LSTMBaseline(n_mels=128, hidden_size=64, num_layers=1).to(device)  # Mniejszy model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    # Mini training (tylko 5 epochs dla testów)
    print("\nTraining for 5 epochs (demo)...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            mel_specs = batch['mel_spec'].to(device)
            valences = batch['valence'].to(device)
            arousals = batch['arousal'].to(device)
            
            optimizer.zero_grad()
            predictions = model(mel_specs)
            
            loss = criterion(predictions[:, 0], valences) + \
                   criterion(predictions[:, 1], arousals)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/5: Loss={avg_loss:.4f}")
    
    # Test inference
    print("\nRunning inference on test set...")
    model.eval()
    predictions_all = []
    targets_all = []
    
    with torch.no_grad():
        for batch in test_loader:
            mel_specs = batch['mel_spec'].to(device)
            valences = batch['valence'].to(device)
            arousals = batch['arousal'].to(device)
            
            predictions = model(mel_specs)
            predictions_all.append(predictions.cpu().numpy())
            targets_all.append(torch.stack([valences, arousals], dim=1).cpu().numpy())
    
    predictions_all = np.vstack(predictions_all)
    targets_all = np.vstack(targets_all)
    
    # Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae_val = mean_absolute_error(targets_all[:, 0], predictions_all[:, 0])
    mae_arou = mean_absolute_error(targets_all[:, 1], predictions_all[:, 1])
    rmse_val = np.sqrt(mean_squared_error(targets_all[:, 0], predictions_all[:, 0]))
    rmse_arou = np.sqrt(mean_squared_error(targets_all[:, 1], predictions_all[:, 1]))
    
    print(f"\n✓ LSTM Results (5 epochs demo):")
    print(f"  Valence: MAE={mae_val:.4f}, RMSE={rmse_val:.4f}")
    print(f"  Arousal: MAE={mae_arou:.4f}, RMSE={rmse_arou:.4f}")
    
    return {'val_mae': mae_val, 'arou_mae': mae_arou}


def main():
    """Run complete pipeline test"""
    
    # Sprawdzenie czy mamy dane
    if not os.path.exists('/home/mist/Documents/datasets/MusicRawData'):
        print(" ERROR: Could not find '/home/mist/Documents/datasets/MusicRawData'")
        print(" Please ensure WAV files are in this directory")
        return
    
    if not os.path.exists('sp2/labels.csv'):
        print(" ERROR: Could not find 'sp2/labels.csv'")
        print(" Please ensure CSV labels are in this directory")
        return
    

    
    try:
        # Step 1: Audio processing
        # step_1_audio_processing()
        
        # Step 2: Data pipeline
        step_2_data_pipeline()
        
        # Step 3a: Random Forest
        rf_results = step_3_random_forest_baseline()
        
        # Step 3b: LSTM
        lstm_results = step_4_lstm_training()
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\nRandom Forest Results:")
        print(f"  Valence MAE: {rf_results['val_mae']:.4f}")
        print(f"  Arousal MAE: {rf_results['arou_mae']:.4f}")
        print(f"\nLSTM Results (5 epochs):")
        print(f"  Valence MAE: {lstm_results['val_mae']:.4f}")
        print(f"  Arousal MAE: {lstm_results['arou_mae']:.4f}")
        
        print("\n✓ Quick Start Complete!")
        print("\nNext steps:")
        print("  1. Full training: python train.py --model_type lstm --epochs 100")
        print("  2. Tune hyperparameters based on validation metrics")
        print("  3. Analyze predictions (look for patterns/failures)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()