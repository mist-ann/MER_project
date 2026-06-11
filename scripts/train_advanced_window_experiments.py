from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from audio_processing import DEAMSegmentGenerator, get_song_ids_and_labels, load_valence_arousal
from evaluator import EmotionEvaluator, EmotionTagger
from model import (
    MER_CNN_Attention,
    MER_CNN_LSTM,
    MER_CNN_MobileNet,
    MER_CNN_Residual,
    MER_CNN_VGG_Style,
    MER_CRNN_Temporal,
    MER_LSTM,
)
from train_cnn_window_experiments import (
    DEFAULT_WINDOWS,
    add_emotion_columns,
    collect_predictions,
    compute_regression_metrics,
    frames_for_seconds,
    jsonable,
    normalize_window,
    split_song_ids,
    validate_generator_split,
    validate_split_integrity,
    window_label,
)

ARCHITECTURES = {
    "residual_cnn": MER_CNN_Residual,
    "temporal_crnn": MER_CRNN_Temporal,
    "attention_cnn": MER_CNN_Attention,
    "vgg_cnn": MER_CNN_VGG_Style,
    "cnn_lstm": MER_CNN_LSTM,
    "lstm": MER_LSTM,
    "mobilenet_cnn": MER_CNN_MobileNet,
}


def selected_windows(args) -> list[float]:
    if args.all_windows:
        return DEFAULT_WINDOWS
    if args.window is not None:
        return [normalize_window(args.window)]
    if args.windows is not None:
        return [normalize_window(window) for window in args.windows]

    raise ValueError("Choose --window 0.5, --window 1.0, --window 3.0, or --all-windows.")


def selected_architectures(args) -> list[str]:
    if args.architecture == "all":
        return list(ARCHITECTURES.keys())
    return [args.architecture]


def build_model(architecture: str, input_shape: tuple[int, int, int], learning_rate: float):
    try:
        model_cls = ARCHITECTURES[architecture]
    except KeyError as exc:
        supported = ", ".join(ARCHITECTURES.keys())
        raise ValueError(f"Unsupported architecture {architecture}. Use one of: {supported}.") from exc

    return model_cls(input_shape=input_shape, learning_rate=learning_rate)


def make_advanced_generator(
    song_ids: list[str],
    labels_dict: dict,
    audio_npy_dir: Path,
    segment_width: int,
    window_seconds: float,
    sr: int,
    hop_length: int,
    batch_size: int,
    shuffle: bool,
    normalize_segments: bool,
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
        normalize_segments=normalize_segments,
    )


def run_advanced_experiment(
    architecture: str,
    window_seconds: float,
    args,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> dict:
    label = window_label(window_seconds)
    run_name = f"{architecture}_{label}"
    if args.run_suffix:
        run_name = f"{run_name}_{args.run_suffix}"
    model_dir = args.output_root / run_name
    result_dir = args.results_dir / run_name

    labels_df = load_valence_arousal(str(args.data_root), span=window_seconds)
    labels_dict = get_song_ids_and_labels(labels_df)
    segment_width = frames_for_seconds(window_seconds, args.sr, args.hop_length)
    input_shape = (args.n_mels, segment_width, 1)

    train_generator = make_advanced_generator(
        train_ids,
        labels_dict,
        args.audio_npy_dir,
        segment_width,
        window_seconds,
        args.sr,
        args.hop_length,
        args.batch_size,
        shuffle=True,
        normalize_segments=args.normalize_segments,
    )
    val_generator = make_advanced_generator(
        val_ids,
        labels_dict,
        args.audio_npy_dir,
        segment_width,
        window_seconds,
        args.sr,
        args.hop_length,
        args.batch_size,
        shuffle=False,
        normalize_segments=args.normalize_segments,
    )
    test_generator = make_advanced_generator(
        test_ids,
        labels_dict,
        args.audio_npy_dir,
        segment_width,
        window_seconds,
        args.sr,
        args.hop_length,
        args.batch_size,
        shuffle=False,
        normalize_segments=args.normalize_segments,
    )

    validate_generator_split(train_generator, train_ids, "train")
    validate_generator_split(val_generator, val_ids, "validation")
    validate_generator_split(test_generator, test_ids, "test")

    model = build_model(architecture, input_shape=input_shape, learning_rate=args.learning_rate)
    model_params = model._model.count_params()

    print(
        f"[{run_name}] input_shape={input_shape}, params={model_params}, "
        f"train_segments={len(train_generator.samples)}, "
        f"val_segments={len(val_generator.samples)}, "
        f"test_segments={len(test_generator.samples)}, "
        f"normalize_segments={args.normalize_segments}"
    )

    if args.dry_run:
        if len(train_generator) > 0:
            x_batch, y_batch = train_generator[0]
            print(f"[{run_name}] first_batch X={x_batch.shape}, y={y_batch.shape}")
        return {
            "architecture": architecture,
            "window_seconds": window_seconds,
            "model_name": run_name,
            "model_params": model_params,
            "segment_width": segment_width,
            "input_shape": input_shape,
            "train_segments": len(train_generator.samples),
            "val_segments": len(val_generator.samples),
            "test_segments": len(test_generator.samples),
            "normalize_segments": args.normalize_segments,
            "run_suffix": args.run_suffix,
        }

    if len(train_generator) == 0 or len(val_generator) == 0 or len(test_generator) == 0:
        raise RuntimeError(f"[{run_name}] One of the generators is empty.")

    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    with (result_dir / "split_song_ids.json").open("w", encoding="utf-8") as f:
        json.dump({"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}, f, indent=2)

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
        "architecture": architecture,
        "window_seconds": window_seconds,
        "model_name": run_name,
        "model_params": model_params,
        "segment_width": segment_width,
        "input_shape": input_shape,
        "train_songs": len(train_ids),
        "val_songs": len(val_ids),
        "test_songs": len(test_ids),
        "train_segments": len(train_generator.samples),
        "val_segments": len(val_generator.samples),
        "test_segments": len(test_generator.samples),
        "normalize_segments": args.normalize_segments,
        "run_suffix": args.run_suffix,
        "learning_rate": args.learning_rate,
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
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "notebooks" / "models" / "advanced")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "notebooks" / "results" / "advanced_window_experiments",
    )

    window_group = parser.add_mutually_exclusive_group(required=True)
    window_group.add_argument("--window", type=float, default=None)
    window_group.add_argument("--windows", type=float, nargs="+", default=None)
    window_group.add_argument("--all-windows", action="store_true")

    parser.add_argument(
        "--architecture",
        choices=list(ARCHITECTURES.keys()) + ["all"],
        default="residual_cnn",
    )
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--limit-songs", type=int, default=None)
    parser.add_argument("--run-suffix", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--normalize-segments", dest="normalize_segments", action="store_true")
    parser.add_argument("--no-normalize-segments", dest="normalize_segments", action="store_false")
    parser.set_defaults(normalize_segments=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows_to_run = selected_windows(args)
    architectures_to_run = selected_architectures(args)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    base_labels = get_song_ids_and_labels(load_valence_arousal(str(args.data_root), span=0.5))
    song_ids = sorted(
        song_id for song_id in base_labels.keys() if (args.audio_npy_dir / f"{song_id}.npy").exists()
    )
    if args.limit_songs is not None:
        song_ids = song_ids[: args.limit_songs]

    if not song_ids:
        raise RuntimeError("No matching .npy files and dynamic labels were found.")

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
    for architecture in architectures_to_run:
        for window_seconds in windows_to_run:
            metrics = run_advanced_experiment(architecture, window_seconds, args, train_ids, val_ids, test_ids)
            all_metrics.append(metrics)

    summary_df = pd.DataFrame(all_metrics)
    if len(architectures_to_run) == 1 and len(windows_to_run) == 1:
        summary_suffix = f"{architectures_to_run[0]}_{window_label(windows_to_run[0])}"
        if args.run_suffix:
            summary_suffix = f"{summary_suffix}_{args.run_suffix}"
    else:
        summary_suffix = "advanced"

    summary_name = f"dry_run_summary_{summary_suffix}.csv" if args.dry_run else f"metrics_summary_{summary_suffix}.csv"
    summary_path = args.results_dir / summary_name
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
