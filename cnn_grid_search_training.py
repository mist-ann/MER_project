import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D,
    Dropout, Dense, Normalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from itertools import product
from datetime import datetime
import csv
import pickle


class EpochProgressCallback(Callback):
    """Custom callback to print epoch progress during training."""
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n  Epoch {epoch + 1}/{self.params['epochs']}", end="")
    
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', 0)
        val_mae = logs.get('val_mae', 0)
        print(f" | val_loss: {val_loss:.4f}, val_mae: {val_mae:.4f}")


class MER_CNN_Model:
    """Base class for MER CNN models."""
    def __init__(self):
        self._model = None
    
    @property
    def model(self):
        return self._model
    
    def fit(self, train_data, validation_data, epochs=50, callbacks=None):
        return self._model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks or [],
            verbose=0  # We handle verbose output with EpochProgressCallback
        )
    
    def evaluate(self, test_data):
        return self._model.evaluate(test_data, verbose=0)


class MER_CNN_Simple(MER_CNN_Model):
    """Simple CNN for Music Emotion Recognition."""
    def __init__(self, dropout=0.5, learning_rate=0.001, filters_base=32):
        self._model = Sequential([
            Input(shape=(128, 128, 1)),
            Normalization(),
            Conv2D(filters_base, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(filters_base * 2, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(filters_base * 4, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dropout(dropout),
            Dense(64, activation="relu"),
            Dense(2, activation="linear"),
        ])
        
        self._model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mean_squared_error",
            metrics=["mae"],
        )
        
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.filters_base = filters_base


def load_mel_spectrograms(data_dir='data/processed'):
    """Load mel spectrograms from pickle files."""
    print("Loading mel spectrograms...")
    
    with open(f'{data_dir}/train_mel_specs.pkl', 'rb') as f:
        train_mel_specs = pickle.load(f)
    
    with open(f'{data_dir}/test_mel_specs.pkl', 'rb') as f:
        test_mel_specs = pickle.load(f)
    
    # Load arousal/valence labels
    train_data = np.load(f'{data_dir}/train_dataset.npz')
    test_data = np.load(f'{data_dir}/test_dataset.npz')

    train_mask = np.isfinite(train_data['valence']) & np.isfinite(train_data['arousal'])
    test_mask = np.isfinite(test_data['valence']) & np.isfinite(test_data['arousal'])

    train_mel_specs = [mel_spec for mel_spec, keep in zip(train_mel_specs, train_mask) if keep]
    test_mel_specs = [mel_spec for mel_spec, keep in zip(test_mel_specs, test_mask) if keep]

    train_valence = train_data['valence'][train_mask]
    train_arousal = train_data['arousal'][train_mask]
    test_valence = test_data['valence'][test_mask]
    test_arousal = test_data['arousal'][test_mask]

    print(f"  Removed invalid rows: train={int((~train_mask).sum())}, test={int((~test_mask).sum())}")
    
    print(f"✓ Train: {len(train_mel_specs)} samples")
    print(f"✓ Test:  {len(test_mel_specs)} samples\n")
    
    return (
        np.array(train_mel_specs)[..., np.newaxis],  # Add channel dimension
        train_valence,
        train_arousal,
        np.array(test_mel_specs)[..., np.newaxis],
        test_valence,
        test_arousal
    )


def create_data_generators(X_train, y_train_val, y_train_arousal, X_val, y_val_val, y_val_arousal, batch_size=32):
    """Create data generators for augmentation."""
    # Simple generator for training (with some augmentation)
    train_gen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    val_gen = ImageDataGenerator()  # No augmentation for validation
    
    # Stack valence and arousal as multi-output
    y_train_stacked = np.stack([y_train_val, y_train_arousal], axis=1)
    y_val_stacked = np.stack([y_val_val, y_val_arousal], axis=1)
    
    train_generator = train_gen.flow(X_train, y_train_stacked, batch_size=batch_size)
    val_generator = val_gen.flow(X_val, y_val_stacked, batch_size=batch_size)
    
    return train_generator, val_generator


def grid_search_cnn(
    data_dir='data/processed',
    dropouts=[0.3, 0.5],
    learning_rates=[1e-3, 5e-4],
    batch_sizes=[32],
    filters_bases=[32],
    epochs=50,
    patience=7,
    results_file='cnn_hyperparameter_search_results.csv'
):
    """
    Grid search over CNN hyperparameters.
    
    For each combination:
    1. Trains the model
    2. Evaluates on validation set
    3. Saves best model checkpoint
    4. Logs results to CSV
    """
    
    # GPU setup
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    device = 'GPU' if len(tf.config.list_physical_devices('GPU')) > 0 else 'CPU'
    print(f"Using device: {device}\n")
    
    # Load data once
    X_train, y_train_val, y_train_arou, X_test, y_test_val, y_test_arou = load_mel_spectrograms(data_dir)
    
    # Split training into train/val
    X_train_split, X_val, y_train_val_split, y_val_val, y_train_arou_split, y_val_arou = train_test_split(
        X_train, y_train_val, y_train_arou,
        test_size=0.15,
        random_state=42
    )
    
    # Stack outputs for multi-task learning
    y_train_split = np.stack([y_train_val_split, y_train_arou_split], axis=1)
    y_val = np.stack([y_val_val, y_val_arou], axis=1)
    y_test = np.stack([y_test_val, y_test_arou], axis=1)
    
    # Parameter combinations
    param_combinations = list(product(
        dropouts,
        learning_rates,
        batch_sizes,
        filters_bases
    ))
    
    print(f"Total combinations to test: {len(param_combinations)}")
    print(f"Estimated time: ~{len(param_combinations) * epochs / 10:.1f} GPU-hours\n")
    
    # CSV setup
    csv_columns = [
        'rank', 'dropout', 'lr', 'batch_size', 'filters_base',
        'best_epoch', 'val_loss', 'val_mae',
        'test_loss', 'test_mae',
        'training_time_min', 'checkpoint'
    ]
    
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
    
    # Results tracking
    results = []
    best_val_loss = float('inf')
    best_params = None
    
    # Create checkpoints directory
    os.makedirs('checkpoints3', exist_ok=True)
    
    # Grid search
    for combo_idx, (dropout, lr, batch_size, filters_base) in enumerate(param_combinations):
        
        print(f"\n{'='*75}")
        print(f"Combination {combo_idx + 1}/{len(param_combinations)}")
        print(f"{'='*75}")
        print(f"Params: dropout={dropout}, lr={lr}, batch_size={batch_size}, filters_base={filters_base}")
        
        start_time = datetime.now()
        
        # Create data generators
        train_gen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        val_gen = ImageDataGenerator()
        
        train_generator = train_gen.flow(
            X_train_split, y_train_split,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_generator = val_gen.flow(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False
        )
        
        test_generator = val_gen.flow(
            X_test, y_test,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Model
        model = MER_CNN_Simple(
            dropout=dropout,
            learning_rate=lr,
            filters_base=filters_base
        )
        
        experiment_name = f"cnn_do{dropout:.2f}_lr{lr:.0e}_bs{batch_size}_fb{filters_base}"
        checkpoint_path = os.path.join('checkpoints3', f'{experiment_name}_best.h5')
        
        # Callbacks
        callbacks = [
            EpochProgressCallback(),
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            ),
        ]
        
        try:
            print(f"\nTraining:")
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks
            )
            
            # Get best epoch and metrics
            best_epoch = int(np.argmin(history.history['val_loss']))
            best_val_loss_combo = float(history.history['val_loss'][best_epoch])
            best_val_mae = float(history.history['val_mae'][best_epoch])
            
            # Test evaluation
            print(f"\nEvaluating on test set...")
            test_loss, test_mae = model.evaluate(test_generator)
            
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            
            # Save result
            result = {
                'dropout': float(dropout),
                'lr': float(lr),
                'batch_size': int(batch_size),
                'filters_base': int(filters_base),
                'best_epoch': best_epoch,
                'val_loss': best_val_loss_combo,
                'val_mae': best_val_mae,
                'test_loss': float(test_loss),
                'test_mae': float(test_mae),
                'training_time_min': float(elapsed),
                'checkpoint': checkpoint_path
            }
            
            results.append(result)
            
            # Update best
            if best_val_loss_combo < best_val_loss:
                best_val_loss = best_val_loss_combo
                best_params = result
            
            # Print results
            print(f"\n✓ Results:")
            print(f"  Best epoch: {best_epoch}")
            print(f"  Val Loss: {best_val_loss_combo:.4f}")
            print(f"  Val MAE: {best_val_mae:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Test MAE: {test_mae:.4f}")
            print(f"  Training time: {elapsed:.1f} min")
            print(f"  Checkpoint: {checkpoint_path}")
            
            # Append to CSV
            with open(results_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                row = {
                    'rank': len(results),
                    'dropout': dropout,
                    'lr': lr,
                    'batch_size': batch_size,
                    'filters_base': filters_base,
                    'best_epoch': best_epoch,
                    'val_loss': f"{best_val_loss_combo:.4f}",
                    'val_mae': f"{best_val_mae:.4f}",
                    'test_loss': f"{test_loss:.4f}",
                    'test_mae': f"{test_mae:.4f}",
                    'training_time_min': f"{elapsed:.1f}",
                    'checkpoint': checkpoint_path
                }
                writer.writerow(row)
        
        except Exception as e:
            print(f"\n✗ Error during training: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n\n{'='*75}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*75}")
    
    # Sort by val_loss
    results_sorted = sorted(results, key=lambda x: x['val_loss'])
    
    print(f"\nTop 10 configurations:")
    for i, res in enumerate(results_sorted[:10]):
        print(f"\n{i+1}. dropout={res['dropout']}, lr={res['lr']}, "
              f"batch_size={res['batch_size']}, filters_base={res['filters_base']}")
        print(f"   Val Loss: {res['val_loss']:.4f} (epoch {res['best_epoch']})")
        print(f"   Test Loss: {res['test_loss']:.4f}, Test MAE: {res['test_mae']:.4f}")
        print(f"   Training time: {res['training_time_min']:.1f} min")
    
    # Save best params
    if best_params:
        print(f"\n\nBest Parameters (by validation loss):")
        print(f"  Dropout: {best_params['dropout']}")
        print(f"  Learning rate: {best_params['lr']}")
        print(f"  Batch size: {best_params['batch_size']}")
        print(f"  Filters base: {best_params['filters_base']}")
        print(f"  Model: {best_params['checkpoint']}")
        
        # Save as JSON
        with open('best_cnn_hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\n✓ Best params saved: best_cnn_hyperparameters.json")
        print(f"✓ All results saved: {results_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropouts', type=float, nargs='+', default=[0.3, 0.5])
    parser.add_argument('--lrs', type=float, nargs='+', default=[1e-3, 5e-4])
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[32])
    parser.add_argument('--filters_bases', type=int, nargs='+', default=[32])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=7)
    
    args = parser.parse_args()
    
    grid_search_cnn(
        dropouts=args.dropouts,
        learning_rates=args.lrs,
        batch_sizes=args.batch_sizes,
        filters_bases=args.filters_bases,
        epochs=args.epochs,
        patience=args.patience
    )