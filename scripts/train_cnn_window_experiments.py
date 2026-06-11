from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from audio_processing import DEAMSegmentGenerator, get_song_ids_and_labels, load_valence_arousal
from evaluator import EmotionEvaluator, EmotionTagger
from model import MER_CNN_Simple, MER_CNN_VGG_Style

DEFAULT_WINDOWS = [0.5, 1.0, 3.0]


def frames_for_seconds(seconds: float, sr: int, hop_length: int) -> int:
    return max(1, int(round(seconds * sr / hop_length)))


def window_label(seconds: float) -> str:
    if float(seconds).is_integer():
        return f"{int(seconds)}s"
    return f"{seconds:g}".replace(".", "p") + "s"


def build_model(model_name: str, input_shape: tuple[int, int, int]):
    if model_name == "simple":
        return MER_CNN_Simple(input_shape=input_shape)
    if model_name == "vgg":
        return MER_CNN_VGG_Style(input_shape=input_shape)

    raise ValueError(f"Unsupported model: {model_name}")


def normalize_window(seconds: float) -> float:
    for allowed in DEFAULT_WINDOWS:
        if np.isclose(seconds, allowed):
            return allowed

    allowed_values = ", ".join(str(window) for window in DEFAULT_WINDOWS)
    raise ValueError(f"Unsupported window {seconds}. Use one of: {allowed_values}.")


def selected_windows(args) -> list[float]:
    if args.all_windows:
        return DEFAULT_WINDOWS
    if args.window is not None:
        return [normalize_window(args.window)]
    if args.windows is not None:
        return [normalize_window(window) for window in args.windows]

    raise ValueError("Choose --window 0.5, --window 1.0, --window 3.0, or --all-windows.")


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def split_song_ids(
    song_ids: list[str],
    validation_size: float,
    test_size: float,
    random_state: int,
) -> tuple[list[str], list[str], list[str]]:
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1.")

    train_val_ids, test_ids = train_test_split(
        song_ids,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    validation_fraction = validation_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=validation_fraction,
        random_state=random_state,
        shuffle=True,
    )
    return train_ids, val_ids, test_ids


def validate_split_integrity(train_ids: list[str], val_ids: list[str], test_ids: list[str]) -> None:
    split_sets = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }
    overlaps = {
        "train_val": split_sets["train"].intersection(split_sets["val"]),
        "train_test": split_sets["train"].intersection(split_sets["test"]),
        "val_test": split_sets["val"].intersection(split_sets["test"]),
    }
    bad_overlaps = {name: sorted(values)[:10] for name, values in overlaps.items() if values}
    if bad_overlaps:
        raise RuntimeError(f"Song-level split leakage detected: {bad_overlaps}")


def validate_generator_split(generator: DEAMSegmentGenerator, expected_song_ids: list[str], split_name: str) -> None:
    expected_ids = set(str(song_id) for song_id in expected_song_ids)
    generated_ids = set(str(song_id) for song_id, _ in generator.samples)
    unexpected_ids = sorted(generated_ids.difference(expected_ids))

    if unexpected_ids:
        preview = unexpected_ids[:10]
        raise RuntimeError(
            f"{split_name} generator contains songs outside its split. "
            f"First unexpected song ids: {preview}"
        )


def make_generator(
    song_ids: list[str],
    labels_dict: dict,
    audio_npy_dir: Path,
    segment_width: int,
    window_seconds: float,
    sr: int,
    hop_length: int,
    batch_size: int,
    shuffle: bool,
) -> DEAMSegmentGenerator:
    return DEAMSegmentGenerator(
        song_ids=song_ids,
        labels_dict=labels_dict,
        data_dir=str(audio_npy_dir),
        segment_width=segment_width,
        hop_size=hop_length,
        sr=sr,
        batch_size=batch_size,
        shuffle=shuffle,
        window_seconds=window_seconds,
    )


def collect_predictions(model, generator: DEAMSegmentGenerator) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = []
    y_true_batches = []
    y_pred_batches = []

    for batch_idx in range(len(generator)):
        x_batch, y_batch = generator[batch_idx]
        y_pred = model.predict(x_batch, verbose=0)

        batch_start = batch_idx * generator.batch_size
        batch_end = batch_start + len(y_batch)
        batch_samples = generator.samples[batch_start:batch_end]

        for (song_id, start_frame), true_pair, pred_pair in zip(batch_samples, y_batch, y_pred):
            start_ms = int(round(start_frame * generator.hop_size / generator.sr * 1000))
            rows.append(
                {
                    "song_id": song_id,
                    "start_frame": int(start_frame),
                    "start_ms": start_ms,
                    "true_valence": float(true_pair[0]),
                    "true_arousal": float(true_pair[1]),
                    "pred_valence": float(pred_pair[0]),
                    "pred_arousal": float(pred_pair[1]),
                }
            )

        y_true_batches.append(y_batch)
        y_pred_batches.append(y_pred)

    y_true = np.vstack(y_true_batches)
    y_pred = np.vstack(y_pred_batches)
    return pd.DataFrame(rows), y_true, y_pred


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    valence_true = y_true[:, 0]
    arousal_true = y_true[:, 1]
    valence_pred = y_pred[:, 0]
    arousal_pred = y_pred[:, 1]

    mae_valence = mean_absolute_error(valence_true, valence_pred)
    mae_arousal = mean_absolute_error(arousal_true, arousal_pred)
    rmse_valence = np.sqrt(mean_squared_error(valence_true, valence_pred))
    rmse_arousal = np.sqrt(mean_squared_error(arousal_true, arousal_pred))
    r2_valence = r2_score(valence_true, valence_pred)
    r2_arousal = r2_score(arousal_true, arousal_pred)

    return {
        "mae_valence": float(mae_valence),
        "mae_arousal": float(mae_arousal),
        "mae_mean": float(np.mean([mae_valence, mae_arousal])),
        "rmse_valence": float(rmse_valence),
        "rmse_arousal": float(rmse_arousal),
        "rmse_mean": float(np.mean([rmse_valence, rmse_arousal])),
        "r2_valence": float(r2_valence),
        "r2_arousal": float(r2_arousal),
        "r2_mean": float(np.mean([r2_valence, r2_arousal])),
    }


def add_emotion_columns(predictions_df: pd.DataFrame, tagger: EmotionTagger) -> pd.DataFrame:
    predictions_df = predictions_df.copy()
    predictions_df["true_emotion"] = tagger.tag_batch(
        predictions_df["true_valence"].to_numpy(),
        predictions_df["true_arousal"].to_numpy(),
    )
    predictions_df["pred_emotion"] = tagger.tag_batch(
        predictions_df["pred_valence"].to_numpy(),
        predictions_df["pred_arousal"].to_numpy(),
    )
    return predictions_df


def run_experiment(
    window_seconds: float,
    args,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> dict:
    label = window_label(window_seconds)
    model_dir = args.output_root / f"model_{label}"
    result_dir = args.results_dir / f"model_{label}"

    labels_df = load_valence_arousal(str(args.data_root), span=window_seconds)
    labels_dict = get_song_ids_and_labels(labels_df)
    segment_width = frames_for_seconds(window_seconds, args.sr, args.hop_length)
    input_shape = (args.n_mels, segment_width, 1)

    train_generator = make_generator(
        train_ids,
        labels_dict,
        args.audio_npy_dir,
        segment_width,
        window_seconds,
        args.sr,
        args.hop_length,
        args.batch_size,
        shuffle=True,
    )
    val_generator = make_generator(
        val_ids,
        labels_dict,
        args.audio_npy_dir,
        segment_width,
        window_seconds,
        args.sr,
        args.hop_length,
        args.batch_size,
        shuffle=False,
    )
    test_generator = make_generator(
        test_ids,
        labels_dict,
        args.audio_npy_dir,
        segment_width,
        window_seconds,
        args.sr,
        args.hop_length,
        args.batch_size,
        shuffle=False,
    )

    validate_generator_split(train_generator, train_ids, "train")
    validate_generator_split(val_generator, val_ids, "validation")
    validate_generator_split(test_generator, test_ids, "test")

    print(
        f"[{label}] input_shape={input_shape}, "
        f"train_segments={len(train_generator.samples)}, "
        f"val_segments={len(val_generator.samples)}, "
        f"test_segments={len(test_generator.samples)}"
    )

    if args.dry_run:
        if len(train_generator) > 0:
            x_batch, y_batch = train_generator[0]
            print(f"[{label}] first_batch X={x_batch.shape}, y={y_batch.shape}")
        return {
            "window_seconds": window_seconds,
            "model_name": f"model_{label}",
            "segment_width": segment_width,
            "input_shape": input_shape,
            "train_segments": len(train_generator.samples),
            "val_segments": len(val_generator.samples),
            "test_segments": len(test_generator.samples),
        }

    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    split_info = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }
    with (result_dir / "split_song_ids.json").open("w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    if len(train_generator) == 0 or len(val_generator) == 0 or len(test_generator) == 0:
        raise RuntimeError(f"[{label}] One of the generators is empty. Check split sizes and preprocessed audio.")

    model = build_model(args.model, input_shape=input_shape)
    model.summary()

    history = model.fit(
        train_generator,
        validation_generator=val_generator,
        epochs=args.epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=args.lr_patience, min_lr=1e-6),
            ModelCheckpoint(str(model_dir / "best_model.keras"), monitor="val_loss", save_best_only=True),
        ],
    )

    model._model.save(model_dir / "final_model.keras")
    with (model_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(history.history), f, indent=2)

    predictions_df, y_true, y_pred = collect_predictions(model, test_generator)

    tagger = EmotionTagger()
    predictions_df = add_emotion_columns(predictions_df, tagger)
    predictions_df.to_csv(result_dir / "test_predictions.csv", index=False)

    regression_metrics = compute_regression_metrics(y_true, y_pred)
    emotion_evaluator = EmotionEvaluator(tagger)
    emotion_results = emotion_evaluator.evaluate(
        y_true[:, 0],
        y_true[:, 1],
        y_pred[:, 0],
        y_pred[:, 1],
    )

    if not args.skip_plots:
        emotion_evaluator.plot_confusion_matrix(emotion_results, result_dir / "emotion_confusion_matrix.png")
        emotion_evaluator.plot_distributions(emotion_results, result_dir / "emotion_distributions.png")
        emotion_evaluator.plot_valence_arousal_scatter(
            y_true[:, 0],
            y_true[:, 1],
            y_pred[:, 0],
            y_pred[:, 1],
            result_dir / "emotion_scatter.png",
        )

    metrics = {
        "window_seconds": window_seconds,
        "model_name": f"model_{label}",
        "segment_width": segment_width,
        "input_shape": input_shape,
        "train_songs": len(train_ids),
        "val_songs": len(val_ids),
        "test_songs": len(test_ids),
        "train_segments": len(train_generator.samples),
        "val_segments": len(val_generator.samples),
        "test_segments": len(test_generator.samples),
        **regression_metrics,
        "emotion_accuracy": emotion_results["overall_accuracy"],
    }

    with (result_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(metrics), f, indent=2)
    with (result_dir / "emotion_results.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(emotion_results), f, indent=2)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--audio-npy-dir", type=Path, default=PROJECT_ROOT / "data" / "DEAM" / "audio_npy")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "notebooks" / "models")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "notebooks" / "results" / "window_experiments",
    )
    window_group = parser.add_mutually_exclusive_group(required=True)
    window_group.add_argument("--window", type=float, default=None, help="Train a single model for one window.")
    window_group.add_argument(
        "--windows",
        type=float,
        nargs="+",
        default=None,
        help="Train selected windows in one process. Prefer --window for separate runs.",
    )
    window_group.add_argument("--all-windows", action="store_true", help="Train 0.5, 1.0, and 3.0 seconds.")
    parser.add_argument("--model", choices=["simple", "vgg"], default="simple")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--limit-songs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows_to_run = selected_windows(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    base_labels = get_song_ids_and_labels(load_valence_arousal(str(args.data_root), span=0.5))
    song_ids = sorted(
        song_id for song_id in base_labels.keys() if (args.audio_npy_dir / f"{song_id}.npy").exists()
    )
    if args.limit_songs is not None:
        song_ids = song_ids[: args.limit_songs]

    if not song_ids:
        raise RuntimeError(
            "No matching .npy files and dynamic labels were found. Run audio preprocessing first."
        )

    train_ids, val_ids, test_ids = split_song_ids(
        song_ids,
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    validate_split_integrity(train_ids, val_ids, test_ids)
    print(
        f"Songs: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)} "
        f"(total={len(song_ids)})"
    )

    all_metrics = []
    for window_seconds in windows_to_run:
        metrics = run_experiment(window_seconds, args, train_ids, val_ids, test_ids)
        all_metrics.append(metrics)

    summary_df = pd.DataFrame(all_metrics)
    if len(windows_to_run) == 1:
        suffix = window_label(windows_to_run[0])
        summary_name = f"dry_run_summary_{suffix}.csv" if args.dry_run else f"metrics_summary_{suffix}.csv"
    else:
        summary_name = "dry_run_summary.csv" if args.dry_run else "metrics_summary.csv"
    summary_path = args.results_dir / summary_name
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
