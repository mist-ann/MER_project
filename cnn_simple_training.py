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
from datetime import datetime
import pickle
import matplotlib.pyplot as plt


class EpochProgressCallback(Callback):
    """Custom callback to print epoch progress during training."""
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n  [{epoch + 1:3d}/{self.params['epochs']}]", end=" | ")
    
    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss', 0)
        train_mae = logs.get('mae', 0)
        val_loss = logs.get('val_loss', 0)
        val_mae = logs.get('val_mae', 0)
        print(f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_mae: {val_mae:.4f}")


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
    def __init__(self, dropout=0.5, learning_rate=0.001):
        self._model = Sequential([
            Input(shape=(128, 128, 1)),
            Normalization(),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
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


def load_mel_spectrograms(data_dir='data/processed'):
    """Load mel spectrograms from pickle files."""
    print("Loading mel spectrograms...")
    
    with open(f'sp2/{data_dir}/train_mel_specs_v0.pkl', 'rb') as f:
        train_mel_specs = pickle.load(f)
    
    with open(f'sp2/{data_dir}/test_mel_specs_v0.pkl', 'rb') as f:
        test_mel_specs = pickle.load(f)
    
    # Load arousal/valence labels
    train_data = np.load(f'sp2/{data_dir}/train_dataset_v0.npz')
    test_data = np.load(f'sp2/{data_dir}/test_dataset_v0.npz')
    
    print(f"✓ Train: {len(train_mel_specs)} samples")
    print(f"✓ Test:  {len(test_mel_specs)} samples\n")
    
    return (
        np.array(train_mel_specs)[..., np.newaxis],  # Add channel dimension
        train_data['valence'],
        train_data['arousal'],
        np.array(test_mel_specs)[..., np.newaxis],
        test_data['valence'],
        test_data['arousal']
    )


def train_cnn(
    data_dir='data/processed',
    dropout=0.5,
    learning_rate=0.001,
    batch_size=32,
    epochs=50,
    patience=10,
    model_name='cnn_mer'
):
    """
    Train a CNN model for Music Emotion Recognition.
    
    Args:
        data_dir: Path to processed data directory
        dropout: Dropout rate
        learning_rate: Learning rate for Adam optimizer
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        patience: Early stopping patience
        model_name: Name for saving checkpoints
    """
    
    # GPU setup
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    device = 'GPU' if len(tf.config.list_physical_devices('GPU')) > 0 else 'CPU'
    print(f"Using device: {device}\n")
    
    # Load data
    X_train, y_train_val, y_train_arou, X_test, y_test_val, y_test_arou = load_mel_spectrograms(data_dir)
    
    # Split training into train/val
    X_train_split, X_val, y_train_val_split, y_val_val, y_train_arou_split, y_val_arou = train_test_split(
        X_train, y_train_val, y_train_arou,
        test_size=0.15,
        random_state=42
    )
    
    # Stack outputs for multi-task learning (valence + arousal)
    y_train = np.stack([y_train_val_split, y_train_arou_split], axis=1)
    y_val = np.stack([y_val_val, y_val_arou], axis=1)
    y_test = np.stack([y_test_val, y_test_arou], axis=1)
    
    print(f"Data shapes:")
    print(f"  Train: {X_train_split.shape}, Labels: {y_train.shape}")
    print(f"  Val:   {X_val.shape}, Labels: {y_val.shape}")
    print(f"  Test:  {X_test.shape}, Labels: {y_test.shape}\n")
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Data augmentation
    print("Setting up data generators...")
    train_gen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    val_gen = ImageDataGenerator()  # No augmentation for validation
    
    train_generator = train_gen.flow(
        X_train_split, y_train,
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
    print("Creating model...")
    model = MER_CNN_Simple(
        dropout=dropout,
        learning_rate=learning_rate
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.model.summary()
    
    checkpoint_path = os.path.join('checkpoints', f'{model_name}_best.h5')
    
    # Callbacks
    callbacks = [
        EpochProgressCallback(),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
    ]
    
    # Training
    print(f"\n{'='*75}")
    print(f"Training Configuration:")
    print(f"  Dropout: {dropout}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Early stopping patience: {patience}")
    print(f"{'='*75}\n")
    
    start_time = datetime.now()
    
    print("Training:\n")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    
    # Test evaluation
    print(f"\n{'='*75}")
    print("Evaluating on test set...")
    test_loss, test_mae = model.evaluate(test_generator)
    
    print(f"\n✓ Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Training time: {elapsed:.1f} min")
    print(f"  Best model saved: {checkpoint_path}")
    
    # Save training history
    best_epoch = int(np.argmin(history.history['val_loss']))
    best_val_loss = float(history.history['val_loss'][best_epoch])
    
    history_dict = {
        'hyperparameters': {
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs_trained': len(history.history['loss']),
            'best_epoch': best_epoch,
        },
        'metrics': {
            'best_val_loss': best_val_loss,
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'training_time_min': elapsed,
        },
        'history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'mae': [float(x) for x in history.history['mae']],
            'val_mae': [float(x) for x in history.history['val_mae']],
        }
    }
    
    history_file = os.path.join('checkpoints', f'{model_name}_history.json')
    with open(history_file, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"  Training history saved: {history_file}")
    
    # Plot training curves
    plot_training_history(history, model_name)
    
    print(f"\n{'='*75}\n")


def plot_training_history(history, model_name='cnn_mer'):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Training and Validation MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join('checkpoints', f'{model_name}_training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Training curves saved: {plot_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--model_name', type=str, default='cnn_mer')
    
    args = parser.parse_args()
    
    train_cnn(
        dropout=args.dropout,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        model_name=args.model_name
    )