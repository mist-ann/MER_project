from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from evaluator import EmotionTagger

EMOTION_COLORS = {
    "NEUTRAL": "#9ca3af",
    "HAPPY": "#facc15",
    "CALM": "#7dd3fc",
    "ANGRY": "#fb7185",
    "SAD": "#60a5fa",
}


def frames_for_seconds(seconds: float, sr: int, hop_length: int) -> int:
    return max(1, int(round(seconds * sr / hop_length)))


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


def deam_to_spotify_scale(values: np.ndarray) -> np.ndarray:
    return (values + 1.0) / 2.0


def spotify_to_deam_scale(value: float) -> float:
    return value * 2.0 - 1.0


def load_mel_spectrogram(audio_path: Path, sr: int, n_mels: int, hop_length: int) -> tuple[np.ndarray, float]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Audio file is empty or unreadable: {audio_path}")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32), float(librosa.get_duration(y=y, sr=sr))


def build_segments(
    mel_spectrogram: np.ndarray,
    segment_width: int,
    hop_frames: int,
    sr: int,
    hop_length: int,
    normalize_segments: bool,
) -> tuple[np.ndarray, list[dict]]:
    total_frames = mel_spectrogram.shape[1]
    starts = list(range(0, total_frames, hop_frames))
    if not starts:
        starts = [0]

    fill_value = float(np.min(mel_spectrogram)) if mel_spectrogram.size else -80.0
    segments = []
    rows = []

    for start_frame in starts:
        end_frame = start_frame + segment_width
        segment = mel_spectrogram[:, start_frame:end_frame]
        if segment.shape[1] < segment_width:
            pad_width = segment_width - segment.shape[1]
            segment = np.pad(
                segment,
                ((0, 0), (0, pad_width)),
                mode="constant",
                constant_values=fill_value,
            )

        segment = np.asarray(segment, dtype=np.float32)
        if normalize_segments:
            segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-6)

        start_s = start_frame * hop_length / sr
        end_s = min(end_frame, total_frames) * hop_length / sr
        rows.append(
            {
                "segment_index": len(rows) + 1,
                "start_frame": int(start_frame),
                "start_s": round(float(start_s), 3),
                "end_s": round(float(end_s), 3),
            }
        )
        segments.append(segment)

    return np.asarray(segments, dtype=np.float32)[..., np.newaxis], rows


def predict_track(args) -> tuple[pd.DataFrame, dict]:
    model_path = args.model_path
    mel_spectrogram, duration_seconds = load_mel_spectrogram(
        args.audio,
        sr=args.sr,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
    )

    segment_width = frames_for_seconds(args.window_seconds, args.sr, args.hop_length)
    hop_frames = frames_for_seconds(args.step_seconds, args.sr, args.hop_length)
    x, rows = build_segments(
        mel_spectrogram,
        segment_width=segment_width,
        hop_frames=hop_frames,
        sr=args.sr,
        hop_length=args.hop_length,
        normalize_segments=args.normalize_segments,
    )

    model = load_model(model_path, compile=False)
    predictions = model.predict(x, batch_size=args.batch_size, verbose=0)
    predictions = np.asarray(predictions, dtype=np.float32)

    tagger = EmotionTagger()
    pred_valence = predictions[:, 0]
    pred_arousal = predictions[:, 1]
    pred_valence_spotify = deam_to_spotify_scale(pred_valence)
    pred_arousal_spotify = deam_to_spotify_scale(pred_arousal)
    pred_emotions = tagger.tag_batch(pred_valence, pred_arousal)

    predictions_df = pd.DataFrame(rows)
    predictions_df["pred_valence_deam_scale"] = pred_valence
    predictions_df["pred_arousal_deam_scale"] = pred_arousal
    predictions_df["pred_valence_0_1_scale"] = pred_valence_spotify
    predictions_df["pred_arousal_0_1_scale"] = pred_arousal_spotify
    predictions_df["pred_emotion"] = pred_emotions

    mean_valence_deam = float(np.mean(pred_valence))
    mean_arousal_deam = float(np.mean(pred_arousal))
    mean_valence_spotify = float(np.mean(pred_valence_spotify))
    mean_arousal_spotify = float(np.mean(pred_arousal_spotify))
    mean_emotion = tagger.tag_emotion(mean_valence_deam, mean_arousal_deam)

    target_valence = args.target_valence
    target_arousal = args.target_arousal
    abs_error_valence = abs(mean_valence_spotify - target_valence)
    abs_error_arousal = abs(mean_arousal_spotify - target_arousal)

    summary = {
        "track_title": args.title,
        "artist": args.artist,
        "audio_path": str(args.audio),
        "model_path": str(model_path),
        "sr": args.sr,
        "n_mels": args.n_mels,
        "hop_length": args.hop_length,
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "segment_width_frames": segment_width,
        "hop_frames": hop_frames,
        "duration_seconds": duration_seconds,
        "segment_count": int(len(predictions_df)),
        "normalize_segments": bool(args.normalize_segments),
        "mean_pred_valence_deam_scale": mean_valence_deam,
        "mean_pred_arousal_deam_scale": mean_arousal_deam,
        "mean_pred_valence_0_1_scale": mean_valence_spotify,
        "mean_pred_arousal_0_1_scale": mean_arousal_spotify,
        "mean_pred_emotion": mean_emotion,
        "target_spotify_valence": target_valence,
        "target_spotify_arousal": target_arousal,
        "abs_error_valence_0_1_scale": float(abs_error_valence),
        "abs_error_arousal_0_1_scale": float(abs_error_arousal),
        "mean_abs_error_0_1_scale": float(np.mean([abs_error_valence, abs_error_arousal])),
        "emotion_distribution": predictions_df["pred_emotion"].value_counts().to_dict(),
        "scale_note": "Model predicts DEAM-style values on roughly [-1, 1]. Spotify comparison uses (x + 1) / 2.",
    }
    return predictions_df, summary


def render_external_visualization(
    predictions_df: pd.DataFrame,
    summary: dict,
    audio_path: Path,
    output_path: Path,
) -> str:
    rows = []
    for row in predictions_df.to_dict(orient="records"):
        rows.append(
            {
                "time_s": round(float(row["start_s"]), 3),
                "pred_valence": round(float(row["pred_valence_deam_scale"]), 6),
                "pred_arousal": round(float(row["pred_arousal_deam_scale"]), 6),
                "pred_emotion": str(row["pred_emotion"]),
            }
        )

    payload = {
        "track_title": summary["track_title"],
        "artist": summary["artist"],
        "rows": rows,
        "segment_count": len(rows),
        "duration_seconds": summary["duration_seconds"],
        "audio_src": os.path.relpath(audio_path, start=output_path.parent).replace("\\", "/"),
        "emotion_colors": EMOTION_COLORS,
        "target_valence_spotify": summary["target_spotify_valence"],
        "target_arousal_spotify": summary["target_spotify_arousal"],
        "target_valence_deam": spotify_to_deam_scale(summary["target_spotify_valence"]),
        "target_arousal_deam": spotify_to_deam_scale(summary["target_spotify_arousal"]),
        "mean_pred_valence_spotify": summary["mean_pred_valence_0_1_scale"],
        "mean_pred_arousal_spotify": summary["mean_pred_arousal_0_1_scale"],
        "mean_pred_valence_deam": summary["mean_pred_valence_deam_scale"],
        "mean_pred_arousal_deam": summary["mean_pred_arousal_deam_scale"],
        "mean_abs_error": summary["mean_abs_error_0_1_scale"],
    }
    data_json = json.dumps(payload, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>External Track MER Replay</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --panel: #ffffff;
      --line: #dbe3ee;
      --text: #111827;
      --muted: #64748b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }}
    header {{
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 720;
      letter-spacing: 0;
    }}
    main {{
      width: min(1440px, 100%);
      margin: 0 auto;
      padding: 18px 20px 28px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 120px minmax(180px, 1fr) 100px 150px;
      gap: 12px;
      align-items: center;
      margin-bottom: 14px;
      padding: 12px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    select, button, input[type="range"] {{
      width: 100%;
    }}
    audio {{
      display: none;
    }}
    select, button {{
      min-height: 38px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }}
    button {{
      cursor: pointer;
      font-weight: 650;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      min-height: 72px;
    }}
    .stat label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .stat strong {{
      font-size: 20px;
      line-height: 1.05;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(360px, 0.85fr) minmax(520px, 1.15fr);
      gap: 14px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .panel h2 {{
      margin: 0 0 10px;
      font-size: 16px;
      font-weight: 700;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 6px;
      background: #fff;
    }}
    .emotion-strip {{
      display: grid;
      grid-template-columns: 78px 1fr;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    .strip-track {{
      display: flex;
      height: 20px;
      border-radius: 5px;
      overflow: hidden;
      border: 1px solid #e2e8f0;
      background: #f1f5f9;
    }}
    .strip-segment {{
      min-width: 2px;
      height: 100%;
      transition: background-color 120ms linear, box-shadow 120ms linear;
    }}
    .strip-segment.current {{
      box-shadow: inset 0 0 0 2px #111827;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }}
    .swatch {{
      width: 11px;
      height: 11px;
      border-radius: 50%;
      border: 1px solid rgba(0,0,0,0.18);
    }}
    @media (max-width: 980px) {{
      .controls {{ grid-template-columns: 1fr; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>External track emotion replay</h1>
  </header>
  <main>
    <section class="controls">
      <button id="playBtn" type="button">Play</button>
      <input id="frameSlider" type="range" min="0" max="0" value="0" />
      <select id="speedSelect" aria-label="Playback speed">
        <option value="1">1x</option>
        <option value="2">2x</option>
        <option value="5">5x</option>
      </select>
      <select id="smoothingSelect" aria-label="Chart smoothing">
        <option value="1">bez wygladzania</option>
        <option value="3">avg 3 seg.</option>
        <option value="5">avg 5 seg.</option>
      </select>
    </section>
    <audio id="audioPlayer" preload="metadata"></audio>

    <section class="stats">
      <div class="stat"><label>Segment</label><strong id="segmentText">-</strong></div>
      <div class="stat"><label>Czas</label><strong id="timeText">-</strong></div>
      <div class="stat"><label>Pred emotion</label><strong id="predEmotionText">-</strong></div>
      <div class="stat"><label>Mean pred final</label><strong id="meanPredText">-</strong></div>
      <div class="stat"><label>Spotify target / MAE</label><strong id="targetText">-</strong></div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Valence-Arousal live</h2>
        <canvas id="planeCanvas" width="620" height="620"></canvas>
        <div class="legend" id="emotionLegend"></div>
      </div>
      <div class="panel">
        <h2>Przebieg w czasie</h2>
        <canvas id="timelineCanvas" width="920" height="430"></canvas>
        <div class="emotion-strip"><div>Pred</div><div id="predStrip" class="strip-track"></div></div>
        <div class="legend">
          <span><i class="swatch" style="background:#ef4444"></i>pred valence</span>
          <span><i class="swatch" style="background:#fb923c"></i>pred arousal</span>
          <span><i class="swatch" style="background:#111827"></i>Spotify avg target</span>
        </div>
      </div>
    </section>
  </main>

  <script>
    const payload = {data_json};
    const colors = payload.emotion_colors;
    const rows = payload.rows;
    const playBtn = document.getElementById('playBtn');
    const frameSlider = document.getElementById('frameSlider');
    const speedSelect = document.getElementById('speedSelect');
    const smoothingSelect = document.getElementById('smoothingSelect');
    const audioPlayer = document.getElementById('audioPlayer');
    const planeCanvas = document.getElementById('planeCanvas');
    const timelineCanvas = document.getElementById('timelineCanvas');
    const planeCtx = planeCanvas.getContext('2d');
    const timelineCtx = timelineCanvas.getContext('2d');
    let currentIndex = 0;
    let timer = null;
    let trackCompleted = false;
    let smoothedRows = [];

    function init() {{
      audioPlayer.src = payload.audio_src;
      frameSlider.max = Math.max(0, rows.length - 1);
      renderEmotionLegend();
      computeSmoothedRows();
      renderStrip();
      draw();
    }}

    function renderEmotionLegend() {{
      const root = document.getElementById('emotionLegend');
      root.innerHTML = '';
      Object.entries(colors).forEach(([name, color]) => {{
        const item = document.createElement('span');
        item.innerHTML = `<i class="swatch" style="background:${{color}}"></i>${{name}}`;
        root.appendChild(item);
      }});
    }}

    function computeSmoothedRows() {{
      const windowSize = Math.max(1, Number(smoothingSelect.value) || 1);
      smoothedRows = rows.map((row, index) => {{
        const start = Math.max(0, index - windowSize + 1);
        const slice = rows.slice(start, index + 1);
        const mean = (key) => slice.reduce((sum, item) => sum + item[key], 0) / slice.length;
        return {{
          ...row,
          pred_valence: mean('pred_valence'),
          pred_arousal: mean('pred_arousal'),
        }};
      }});
    }}

    function displayValue(index, key) {{
      return (smoothedRows[index] || rows[index])[key];
    }}

    function emotionColor(name) {{
      return colors[name] || '#94a3b8';
    }}

    function stop() {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
      }}
      audioPlayer.pause();
      playBtn.textContent = 'Play';
    }}

    function play() {{
      stop();
      if (currentIndex >= rows.length - 1) {{
        currentIndex = 0;
        trackCompleted = false;
        frameSlider.value = currentIndex;
      }}
      playBtn.textContent = 'Pause';
      syncAudio();
      const speed = Number(speedSelect.value);
      audioPlayer.playbackRate = speed;
      audioPlayer.play().catch(() => {{}});
      timer = setInterval(() => {{
        currentIndex += 1;
        if (currentIndex >= rows.length) {{
          currentIndex = rows.length - 1;
          trackCompleted = true;
          frameSlider.value = currentIndex;
          draw();
          stop();
          return;
        }}
        frameSlider.value = currentIndex;
        draw();
      }}, 1000 / speed);
    }}

    function syncAudio() {{
      const targetTime = rows[currentIndex]?.time_s ?? 0;
      if (Number.isFinite(targetTime) && Math.abs(audioPlayer.currentTime - targetTime) > 0.35) {{
        audioPlayer.currentTime = targetTime;
      }}
    }}

    function xyToPlane(v, a, size) {{
      const pad = 46;
      const x = pad + ((v + 1) / 2) * (size - pad * 2);
      const y = size - pad - ((a + 1) / 2) * (size - pad * 2);
      return [x, y];
    }}

    function drawPlane() {{
      const ctx = planeCtx;
      const size = planeCanvas.width;
      const pad = 46;
      const plot = size - pad * 2;
      ctx.clearRect(0, 0, size, size);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, size, size);

      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      for (let i = -1; i <= 1; i += 0.5) {{
        const [x] = xyToPlane(i, 0, size);
        const [, y] = xyToPlane(0, i, size);
        ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, size - pad); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(size - pad, y); ctx.stroke();
      }}
      const [zeroX, zeroY] = xyToPlane(0, 0, size);
      ctx.strokeStyle = '#94a3b8';
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(zeroX, pad); ctx.lineTo(zeroX, size - pad); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad, zeroY); ctx.lineTo(size - pad, zeroY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = '#cbd5e1';
      ctx.strokeRect(pad, pad, plot, plot);

      for (let i = 0; i <= currentIndex; i++) {{
        const row = rows[i];
        const [x, y] = xyToPlane(displayValue(i, 'pred_valence'), displayValue(i, 'pred_arousal'), size);
        ctx.globalAlpha = i === currentIndex ? 0.95 : 0.2;
        ctx.fillStyle = emotionColor(row.pred_emotion);
        ctx.beginPath(); ctx.arc(x, y, i === currentIndex ? 8 : 3, 0, Math.PI * 2); ctx.fill();
      }}
      ctx.globalAlpha = 1;

      const [targetX, targetY] = xyToPlane(payload.target_valence_deam, payload.target_arousal_deam, size);
      ctx.strokeStyle = '#111827';
      ctx.fillStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(targetX, targetY, 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#111827';
      ctx.font = '13px Segoe UI, Arial';
      ctx.fillText('Valence', size / 2 - 24, size - 12);
      ctx.save();
      ctx.translate(16, size / 2 + 24);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Arousal', 0, 0);
      ctx.restore();
    }}

    function timelinePoint(index, value) {{
      const pad = {{left: 62, right: 24, top: 24, bottom: 52}};
      const w = timelineCanvas.width - pad.left - pad.right;
      const h = timelineCanvas.height - pad.top - pad.bottom;
      const x = pad.left + (index / Math.max(1, rows.length - 1)) * w;
      const y = pad.top + ((1 - value) / 2) * h;
      return [x, y];
    }}

    function drawSeries(ctx, key, color) {{
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      rows.forEach((row, i) => {{
        if (i > currentIndex) return;
        const [x, y] = timelinePoint(i, displayValue(i, key));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.stroke();
    }}

    function drawTargetLine(value, color) {{
      const ctx = timelineCtx;
      const pad = {{left: 62, right: 24, top: 24, bottom: 52}};
      const w = timelineCanvas.width - pad.left - pad.right;
      const [, y] = timelinePoint(0, value);
      ctx.strokeStyle = color;
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + w, y); ctx.stroke();
      ctx.setLineDash([]);
    }}

    function drawTimeline() {{
      const ctx = timelineCtx;
      const pad = {{left: 62, right: 24, top: 24, bottom: 52}};
      const w = timelineCanvas.width - pad.left - pad.right;
      const h = timelineCanvas.height - pad.top - pad.bottom;
      ctx.clearRect(0, 0, timelineCanvas.width, timelineCanvas.height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, timelineCanvas.width, timelineCanvas.height);

      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      for (let v = -1; v <= 1; v += 0.5) {{
        const [, y] = timelinePoint(0, v);
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + w, y); ctx.stroke();
      }}
      drawTargetLine(payload.target_valence_deam, '#111827');
      drawTargetLine(payload.target_arousal_deam, '#111827');
      drawSeries(ctx, 'pred_valence', '#ef4444');
      drawSeries(ctx, 'pred_arousal', '#fb923c');

      const [cursorX] = timelinePoint(currentIndex, 0);
      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(cursorX, pad.top); ctx.lineTo(cursorX, pad.top + h); ctx.stroke();

      const [, zeroY] = timelinePoint(0, 0);
      ctx.fillStyle = '#111827';
      ctx.font = '13px Segoe UI, Arial';
      ctx.fillText('-1', 28, pad.top + h + 4);
      ctx.fillText('0', 38, zeroY + 4);
      ctx.fillText('1', 38, pad.top + 4);

      const currentTime = rows[currentIndex]?.time_s ?? 0;
      const currentLabel = `${{currentTime.toFixed(1)}} s`;
      const labelWidth = ctx.measureText(currentLabel).width + 10;
      const labelX = Math.min(Math.max(cursorX, pad.left + labelWidth / 2), pad.left + w - labelWidth / 2);
      const labelY = pad.top + h - 8;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(labelX - labelWidth / 2, labelY - 13, labelWidth, 18);
      ctx.strokeStyle = '#111827';
      ctx.strokeRect(labelX - labelWidth / 2, labelY - 13, labelWidth, 18);
      ctx.fillStyle = '#111827';
      ctx.textAlign = 'center';
      ctx.fillText(currentLabel, labelX, labelY);
      ctx.textAlign = 'start';

      ctx.fillStyle = '#111827';
      ctx.font = '13px Segoe UI, Arial';
      ctx.fillText('Czas segmentu [s]', timelineCanvas.width / 2 - 55, timelineCanvas.height - 10);
      ctx.save();
      ctx.translate(16, timelineCanvas.height / 2 + 48);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Wartosc valence/arousal [-1, 1]', 0, 0);
      ctx.restore();
    }}

    function renderStrip() {{
      const root = document.getElementById('predStrip');
      root.innerHTML = '';
      rows.forEach((row, index) => {{
        const div = document.createElement('div');
        div.className = 'strip-segment';
        div.dataset.color = emotionColor(row.pred_emotion);
        div.style.width = `${{100 / rows.length}}%`;
        div.style.background = '#e2e8f0';
        div.title = `${{index + 1}}: ${{row.pred_emotion}}`;
        root.appendChild(div);
      }});
      updateStrip();
    }}

    function updateStrip() {{
      const segments = document.getElementById('predStrip').children;
      Array.from(segments).forEach((segment, index) => {{
        segment.style.background = index <= currentIndex ? segment.dataset.color : '#e2e8f0';
        segment.classList.toggle('current', index === currentIndex);
      }});
    }}

    function draw() {{
      const row = rows[currentIndex];
      document.getElementById('segmentText').textContent = `${{currentIndex + 1}} / ${{rows.length}}`;
      document.getElementById('timeText').textContent = `${{row.time_s.toFixed(1)}} s`;
      document.getElementById('predEmotionText').textContent = row.pred_emotion;
      document.getElementById('meanPredText').textContent = trackCompleted
        ? `${{payload.mean_pred_valence_spotify.toFixed(3)}} / ${{payload.mean_pred_arousal_spotify.toFixed(3)}}`
        : '-';
      document.getElementById('targetText').textContent = trackCompleted
        ? `${{payload.target_valence_spotify.toFixed(2)}} / ${{payload.target_arousal_spotify.toFixed(2)}} | ${{payload.mean_abs_error.toFixed(3)}}`
        : `${{payload.target_valence_spotify.toFixed(2)}} / ${{payload.target_arousal_spotify.toFixed(2)}}`;
      updateStrip();
      drawPlane();
      drawTimeline();
    }}

    playBtn.addEventListener('click', () => timer ? stop() : play());
    frameSlider.addEventListener('input', () => {{
      currentIndex = Number(frameSlider.value);
      trackCompleted = false;
      syncAudio();
      draw();
    }});
    speedSelect.addEventListener('change', () => {{
      if (timer) play();
    }});
    smoothingSelect.addEventListener('change', () => {{
      computeSmoothedRows();
      draw();
    }});

    init();
  </script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True, help="Path to an external audio file.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "notebooks" / "models" / "model_1s" / "best_model.keras",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "notebooks" / "results" / "external_tracks" / "never_gonna_give_you_up_model_1s",
    )
    parser.add_argument("--title", type=str, default="Never Gonna Give You Up")
    parser.add_argument("--artist", type=str, default="Rick Astley")
    parser.add_argument("--target-valence", type=float, default=0.91)
    parser.add_argument("--target-arousal", type=float, default=0.94)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--normalize-segments", action="store_true")
    parser.add_argument(
        "--common-visualization-output",
        type=Path,
        default=PROJECT_ROOT / "notebooks" / "results" / "live_visualization" / "live_emotion_visualization.html",
    )
    parser.add_argument(
        "--deam-predictions",
        type=Path,
        default=PROJECT_ROOT / "notebooks" / "results" / "window_experiments" / "model_1s" / "test_predictions.csv",
    )
    parser.add_argument("--deam-audio-dir", type=Path, default=PROJECT_ROOT / "data" / "DEAM" / "MEMD_audio")
    parser.add_argument("--skip-common-visualization", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_df, summary = predict_track(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "external_track_predictions.csv"
    summary_path = args.output_dir / "external_track_summary.json"
    predictions_df.to_csv(predictions_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(summary), f, indent=2, ensure_ascii=False)

    if not args.skip_common_visualization:
        from create_live_emotion_visualization import (
            add_external_track,
            build_payload,
            load_predictions,
            render_html,
        )

        deam_df = load_predictions(args.deam_predictions, song_id=None, max_songs=None)
        payload = build_payload(deam_df, args.deam_audio_dir, args.common_visualization_output)
        add_external_track(payload, predictions_path, summary_path, args.audio, args.common_visualization_output)
        args.common_visualization_output.parent.mkdir(parents=True, exist_ok=True)
        args.common_visualization_output.write_text(
            render_html(payload, args.deam_predictions),
            encoding="utf-8",
        )

    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved summary to: {summary_path}")
    if not args.skip_common_visualization:
        print(f"Updated common visualization: {args.common_visualization_output}")
    print(
        "Mean prediction on 0..1 scale: "
        f"valence={summary['mean_pred_valence_0_1_scale']:.4f}, "
        f"arousal={summary['mean_pred_arousal_0_1_scale']:.4f}"
    )
    print(
        "Spotify target: "
        f"valence={summary['target_spotify_valence']:.4f}, "
        f"arousal={summary['target_spotify_arousal']:.4f}"
    )
    print(
        "Absolute error: "
        f"valence={summary['abs_error_valence_0_1_scale']:.4f}, "
        f"arousal={summary['abs_error_arousal_0_1_scale']:.4f}, "
        f"mean={summary['mean_abs_error_0_1_scale']:.4f}"
    )


if __name__ == "__main__":
    main()
