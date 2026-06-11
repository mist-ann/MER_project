from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from audio_processing import DEAMSegmentGenerator, get_song_ids_and_labels, load_valence_arousal
from model import MER_LSTM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--audio-npy-dir", type=Path, default=PROJECT_ROOT / "data" / "DEAM" / "audio_npy")
    parser.add_argument("--span", type=float, default=3.0)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--segment-width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "notebooks" / "models" / "MER_LSTM")
    args = parser.parse_args()

    labels_df = load_valence_arousal(str(args.data_root), span=args.span)
    dynamic_labels = get_song_ids_and_labels(labels_df)

    existing_song_ids = [
        song_id
        for song_id in dynamic_labels.keys()
        if (args.audio_npy_dir / f"{song_id}.npy").exists()
    ]

    if not existing_song_ids:
        raise RuntimeError(
            "No preprocessed .npy files were found. Run preprocessing first with "
            "audio_processing.preprocess_deam(...)."
        )

    print(f"Found {len(existing_song_ids)} songs with audio_npy and labels.")

    train_ids, val_ids = train_test_split(
        existing_song_ids,
        test_size=args.validation_size,
        random_state=42,
        shuffle=True,
    )

    train_generator = DEAMSegmentGenerator(
        song_ids=train_ids,
        labels_dict=dynamic_labels,
        data_dir=str(args.audio_npy_dir),
        segment_width=args.segment_width,
        hop_size=args.hop_length,
        sr=args.sr,
        batch_size=args.batch_size,
        shuffle=True,
        window_seconds=args.span,
    )

    valid_generator = DEAMSegmentGenerator(
        song_ids=val_ids,
        labels_dict=dynamic_labels,
        data_dir=str(args.audio_npy_dir),
        segment_width=args.segment_width,
        hop_size=args.hop_length,
        sr=args.sr,
        batch_size=args.batch_size,
        shuffle=False,
        window_seconds=args.span,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = MER_LSTM(input_shape=(128, args.segment_width, 1))
    model.summary()

    model.fit(
        train_generator,
        validation_generator=valid_generator,
        epochs=args.epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(str(args.output_dir / "best_model.h5"), save_best_only=True),
        ],
    )

    model._model.save(args.output_dir / "final_model.h5")
    print(f"Saved best model to: {args.output_dir / 'best_model.h5'}")
    print(f"Saved final model to: {args.output_dir / 'final_model.h5'}")


if __name__ == "__main__":
    main()
