from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMOTION_COLORS = {
    "NEUTRAL": "#9ca3af",
    "HAPPY": "#facc15",
    "CALM": "#7dd3fc",
    "ANGRY": "#fb7185",
    "SAD": "#60a5fa",
}


def song_sort_key(song_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(song_id))
    except ValueError:
        return (1, song_id)


def spotify_to_deam_scale(value: float) -> float:
    return value * 2.0 - 1.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT
        / "notebooks"
        / "results"
        / "window_experiments"
        / "model_1s"
        / "test_predictions.csv",
    )
    parser.add_argument("--song-id", type=str, default=None)
    parser.add_argument("--max-songs", type=int, default=None)
    parser.add_argument("--audio-dir", type=Path, default=PROJECT_ROOT / "data" / "DEAM" / "MEMD_audio")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "notebooks"
        / "results"
        / "live_visualization"
        / "live_emotion_visualization.html",
    )
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument(
        "--external-predictions",
        type=Path,
        default=PROJECT_ROOT
        / "notebooks"
        / "results"
        / "external_tracks"
        / "never_gonna_give_you_up_model_1s"
        / "external_track_predictions.csv",
    )
    parser.add_argument(
        "--external-summary",
        type=Path,
        default=PROJECT_ROOT
        / "notebooks"
        / "results"
        / "external_tracks"
        / "never_gonna_give_you_up_model_1s"
        / "external_track_summary.json",
    )
    parser.add_argument("--external-audio", type=Path, default=PROJECT_ROOT / "never-gonna-give-you-up.mp3")
    parser.add_argument("--no-external", action="store_true")
    return parser.parse_args()


def load_predictions(path: Path, song_id: str | None, max_songs: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file does not exist: {path}")

    df = pd.read_csv(path)
    required_columns = {
        "song_id",
        "start_ms",
        "true_valence",
        "true_arousal",
        "pred_valence",
        "pred_arousal",
        "true_emotion",
        "pred_emotion",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in predictions CSV: {sorted(missing)}")

    df["song_id"] = df["song_id"].astype(str)

    if song_id is not None:
        song_id = str(song_id)
        df = df[df["song_id"] == song_id]
        if df.empty:
            raise ValueError(f"No rows found for song_id={song_id}")
    elif max_songs is not None:
        selected = df.groupby("song_id").size().sort_values(ascending=False).head(max_songs).index
        df = df[df["song_id"].isin(selected)]

    sort_columns = ["song_id", "start_ms"]
    if "start_frame" in df.columns:
        sort_columns = ["song_id", "start_frame", "start_ms"]
    return df.sort_values(sort_columns).reset_index(drop=True)


def build_payload(df: pd.DataFrame, audio_dir: Path | None, output_path: Path) -> dict:
    songs = {}
    for song_id in sorted(df["song_id"].unique(), key=song_sort_key):
        song_df = df[df["song_id"] == song_id]
        song_df = song_df.sort_values("start_ms")
        rows = []
        for row in song_df.to_dict(orient="records"):
            rows.append(
                {
                    "time_s": round(float(row["start_ms"]) / 1000.0, 3),
                    "true_valence": round(float(row["true_valence"]), 6),
                    "true_arousal": round(float(row["true_arousal"]), 6),
                    "pred_valence": round(float(row["pred_valence"]), 6),
                    "pred_arousal": round(float(row["pred_arousal"]), 6),
                    "true_emotion": str(row["true_emotion"]),
                    "pred_emotion": str(row["pred_emotion"]),
                }
            )
        correct = sum(1 for row in rows if row["true_emotion"] == row["pred_emotion"])
        audio_src = None
        if audio_dir is not None:
            audio_path = audio_dir / f"{song_id}.mp3"
            if audio_path.exists():
                audio_src = os.path.relpath(audio_path, start=output_path.parent).replace("\\", "/")

        songs[str(song_id)] = {
            "song_id": str(song_id),
            "kind": "deam",
            "title": f"DEAM song {song_id}",
            "has_true": True,
            "rows": rows,
            "segment_count": len(rows),
            "emotion_accuracy": correct / len(rows) if rows else 0.0,
            "audio_src": audio_src,
        }

    return {
        "songs": songs,
        "song_ids": list(songs.keys()),
        "emotion_colors": EMOTION_COLORS,
    }


def add_external_track(payload: dict, predictions_path: Path, summary_path: Path, audio_path: Path, output_path: Path):
    if not predictions_path.exists() or not summary_path.exists():
        return

    predictions_df = pd.read_csv(predictions_path)
    required_columns = {
        "start_s",
        "pred_valence_deam_scale",
        "pred_arousal_deam_scale",
        "pred_emotion",
    }
    missing = required_columns.difference(predictions_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in external predictions CSV: {sorted(missing)}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

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

    audio_src = None
    if audio_path.exists():
        audio_src = os.path.relpath(audio_path, start=output_path.parent).replace("\\", "/")

    external_id = "external_never_gonna_give_you_up"
    payload["songs"][external_id] = {
        "song_id": external_id,
        "kind": "external",
        "title": f"External: {summary.get('artist', 'Unknown')} - {summary.get('track_title', audio_path.stem)}",
        "has_true": False,
        "rows": rows,
        "segment_count": len(rows),
        "audio_src": audio_src,
        "target_valence_spotify": float(summary["target_spotify_valence"]),
        "target_arousal_spotify": float(summary["target_spotify_arousal"]),
        "target_valence_deam": spotify_to_deam_scale(float(summary["target_spotify_valence"])),
        "target_arousal_deam": spotify_to_deam_scale(float(summary["target_spotify_arousal"])),
        "mean_pred_valence_spotify": float(summary["mean_pred_valence_0_1_scale"]),
        "mean_pred_arousal_spotify": float(summary["mean_pred_arousal_0_1_scale"]),
        "mean_abs_error": float(summary["mean_abs_error_0_1_scale"]),
    }
    payload["song_ids"].append(external_id)


def render_html(payload: dict, source_path: Path) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live MER Replay</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --panel: #ffffff;
      --line: #dbe3ee;
      --text: #111827;
      --muted: #64748b;
      --pred: #ef4444;
      --true: #2563eb;
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
      grid-template-columns: minmax(260px, 360px) 120px minmax(180px, 1fr) 100px 150px;
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
      min-width: 0;
    }}
    select {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
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
      font-size: 22px;
      line-height: 1;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(520px, 1.05fr) minmax(520px, 0.95fr);
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
    .plane-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .plane-grid.single {{
      grid-template-columns: minmax(0, 1fr);
    }}
    .plane-grid.single .plane-card {{
      width: min(620px, 100%);
      margin: 0 auto;
    }}
    .plane-card h3 {{
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
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
      .plane-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Live replay rozpoznawania emocji</h1>
  </header>
  <main>
    <section class="controls">
      <select id="songSelect" aria-label="Song id"></select>
      <button id="playBtn" type="button">Play</button>
      <input id="frameSlider" type="range" min="0" max="0" value="0" />
      <select id="speedSelect" aria-label="Playback speed">
        <option value="900">1x</option>
        <option value="450">2x</option>
        <option value="180">5x</option>
      </select>
      <select id="smoothingSelect" aria-label="Chart smoothing">
        <option value="1">bez wygładzania</option>
        <option value="3">avg 3 seg.</option>
        <option value="5">avg 5 seg.</option>
      </select>
    </section>
    <audio id="audioPlayer" preload="metadata"></audio>

    <section class="stats">
      <div class="stat"><label>Segment</label><strong id="segmentText">-</strong></div>
      <div class="stat"><label>Czas</label><strong id="timeText">-</strong></div>
      <div class="stat"><label id="trueStatLabel">True emotion</label><strong id="trueEmotionText">-</strong></div>
      <div class="stat"><label>Pred emotion</label><strong id="predEmotionText">-</strong></div>
      <div class="stat"><label id="finalStatLabel">Accuracy final</label><strong id="accuracyText">-</strong></div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Valence-Arousal live</h2>
        <div class="plane-grid" id="planeGrid">
          <div class="plane-card" id="truePlaneCard">
            <h3>True values</h3>
            <canvas id="truePlaneCanvas" width="540" height="540"></canvas>
          </div>
          <div class="plane-card" id="predPlaneCard">
            <h3>Predicted values</h3>
            <canvas id="predPlaneCanvas" width="540" height="540"></canvas>
          </div>
        </div>
        <div class="legend" id="legend"></div>
      </div>
      <div class="panel">
        <h2>Przebieg w czasie</h2>
        <canvas id="timelineCanvas" width="920" height="430"></canvas>
        <div class="emotion-strip" id="trueStripRow"><div>True</div><div id="trueStrip" class="strip-track"></div></div>
        <div class="emotion-strip"><div>Pred</div><div id="predStrip" class="strip-track"></div></div>
        <div class="legend">
          <span><i class="swatch" style="background:#2563eb"></i>true valence</span>
          <span><i class="swatch" style="background:#60a5fa"></i>true arousal</span>
          <span><i class="swatch" style="background:#ef4444"></i>pred valence</span>
          <span><i class="swatch" style="background:#fb923c"></i>pred arousal</span>
        </div>
      </div>
    </section>
  </main>

  <script>
    const payload = {data_json};
    const colors = payload.emotion_colors;
    const songSelect = document.getElementById('songSelect');
    const playBtn = document.getElementById('playBtn');
    const frameSlider = document.getElementById('frameSlider');
    const speedSelect = document.getElementById('speedSelect');
    const smoothingSelect = document.getElementById('smoothingSelect');
    const planeGrid = document.getElementById('planeGrid');
    const truePlaneCard = document.getElementById('truePlaneCard');
    const truePlaneCanvas = document.getElementById('truePlaneCanvas');
    const predPlaneCanvas = document.getElementById('predPlaneCanvas');
    const timelineCanvas = document.getElementById('timelineCanvas');
    const audioPlayer = document.getElementById('audioPlayer');
    const truePlaneCtx = truePlaneCanvas.getContext('2d');
    const predPlaneCtx = predPlaneCanvas.getContext('2d');
    const timelineCtx = timelineCanvas.getContext('2d');
    let timer = null;
    let currentSong = null;
    let currentIndex = 0;
    let smoothedRows = [];
    let trackCompleted = false;

    function init() {{
      payload.song_ids.forEach((id) => {{
        const option = document.createElement('option');
        option.value = id;
        option.textContent = `${{payload.songs[id].title || id}} (${{payload.songs[id].segment_count}} seg.)`;
        songSelect.appendChild(option);
      }});
      renderLegend();
      loadSong(payload.song_ids[0]);
    }}

    function renderLegend() {{
      const legend = document.getElementById('legend');
      legend.innerHTML = '';
      Object.entries(colors).forEach(([name, color]) => {{
        const item = document.createElement('span');
        item.innerHTML = `<i class="swatch" style="background:${{color}}"></i>${{name}}`;
        legend.appendChild(item);
      }});
    }}

    function loadSong(songId) {{
      stop();
      currentSong = payload.songs[songId];
      currentIndex = 0;
      trackCompleted = false;
      document.getElementById('trueStripRow').style.display = currentSong.has_true ? 'grid' : 'none';
      truePlaneCard.style.display = currentSong.has_true ? 'block' : 'none';
      planeGrid.classList.toggle('single', !currentSong.has_true);
      frameSlider.max = Math.max(0, currentSong.rows.length - 1);
      frameSlider.value = 0;
      if (currentSong.audio_src) {{
        audioPlayer.src = currentSong.audio_src;
        audioPlayer.style.visibility = 'visible';
        audioPlayer.currentTime = currentSong.rows[0]?.time_s || 0;
      }} else {{
        audioPlayer.removeAttribute('src');
      }}
      computeSmoothedRows();
      renderStrips();
      draw();
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
      if (currentIndex >= currentSong.rows.length - 1) {{
        currentIndex = 0;
        trackCompleted = false;
        frameSlider.value = currentIndex;
        draw();
      }}
      playBtn.textContent = 'Pause';
      syncAudioToCurrentSegment();
      if (audioPlayer.src) {{
        audioPlayer.playbackRate = 900 / Number(speedSelect.value);
        audioPlayer.play().catch(() => {{}});
      }}
      timer = setInterval(() => {{
        currentIndex += 1;
        if (currentIndex >= currentSong.rows.length) {{
          currentIndex = currentSong.rows.length - 1;
          trackCompleted = true;
          frameSlider.value = currentIndex;
          draw();
          stop();
          return;
        }}
        frameSlider.value = currentIndex;
        draw();
      }}, Number(speedSelect.value));
    }}

    function syncAudioToCurrentSegment() {{
      if (!audioPlayer.src || !currentSong?.rows?.length) return;
      const targetTime = currentSong.rows[currentIndex].time_s;
      if (Number.isFinite(targetTime) && Math.abs(audioPlayer.currentTime - targetTime) > 0.35) {{
        audioPlayer.currentTime = targetTime;
      }}
    }}

    function emotionColor(name) {{
      return colors[name] || '#94a3b8';
    }}

    function computeSmoothedRows() {{
      if (!currentSong?.rows?.length) {{
        smoothedRows = [];
        return;
      }}

      const windowSize = Math.max(1, Number(smoothingSelect.value) || 1);
      smoothedRows = currentSong.rows.map((row, index) => {{
        const start = Math.max(0, index - windowSize + 1);
        const slice = currentSong.rows.slice(start, index + 1);
        const mean = (key) => slice.reduce((sum, item) => sum + item[key], 0) / slice.length;
        const result = {{ ...row }};
        ['true_valence', 'true_arousal', 'pred_valence', 'pred_arousal'].forEach((key) => {{
          if (typeof row[key] === 'number') {{
            result[key] = mean(key);
          }}
        }});
        return result;
      }});
    }}

    function displayValue(index, key) {{
      return (smoothedRows[index] || currentSong.rows[index])[key];
    }}

    function xyToPlane(v, a, size) {{
      const pad = 46;
      const x = pad + ((v + 1) / 2) * (size - pad * 2);
      const y = size - pad - ((a + 1) / 2) * (size - pad * 2);
      return [x, y];
    }}

    function drawEmotionRegionLabels(ctx, size) {{
      const labels = [
        ['ANGRY', -0.72, 0.84],
        ['HAPPY', 0.72, 0.84],
        ['SAD', -0.72, -0.84],
        ['CALM', 0.72, -0.84],
        ['NEUTRAL', 0, 0.12],
      ];

      ctx.save();
      ctx.globalAlpha = 0.72;
      ctx.font = '12px Segoe UI, Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      labels.forEach(([label, valence, arousal]) => {{
        const [x, y] = xyToPlane(valence, arousal, size);
        ctx.fillStyle = emotionColor(label);
        ctx.fillText(label, x, y);
      }});
      ctx.restore();
    }}

    function drawPlaneBase(ctx, canvas) {{
      const size = canvas.width;
      ctx.clearRect(0, 0, size, size);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, size, size);
      const pad = 46;
      const plot = size - pad * 2;

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
      drawEmotionRegionLabels(ctx, size);

      ctx.fillStyle = '#111827';
      ctx.font = '14px Segoe UI, Arial';
      ctx.fillText('Valence', size / 2 - 24, size - 12);
      ctx.save();
      ctx.translate(16, size / 2 + 24);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Arousal', 0, 0);
      ctx.restore();

      return size;
    }}

    function drawPlanePoints(ctx, canvas, valenceKey, arousalKey, emotionKey) {{
      const size = drawPlaneBase(ctx, canvas);
      const rows = currentSong.rows;
      for (let i = 0; i <= currentIndex; i++) {{
        const r = rows[i];
        const valence = displayValue(i, valenceKey);
        const arousal = displayValue(i, arousalKey);
        if (!Number.isFinite(valence) || !Number.isFinite(arousal)) continue;

        const [x, y] = xyToPlane(valence, arousal, size);
        const isCurrent = i === currentIndex;
        ctx.globalAlpha = isCurrent ? 1 : 0.42;
        ctx.fillStyle = emotionColor(r[emotionKey]);
        ctx.beginPath(); ctx.arc(x, y, isCurrent ? 10 : 4.8, 0, Math.PI * 2); ctx.fill();
        if (isCurrent) {{
          ctx.globalAlpha = 1;
          ctx.strokeStyle = '#111827';
          ctx.lineWidth = 2.5;
          ctx.beginPath(); ctx.arc(x, y, 12.5, 0, Math.PI * 2); ctx.stroke();
        }}
      }}
      ctx.globalAlpha = 1;
      return size;
    }}

    function drawExternalTarget(ctx, canvas) {{
      if (typeof currentSong.target_valence_deam === 'number') {{
        const size = canvas.width;
        const [targetX, targetY] = xyToPlane(currentSong.target_valence_deam, currentSong.target_arousal_deam, size);
        ctx.strokeStyle = '#111827';
        ctx.fillStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(targetX, targetY, 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }}
    }}

    function drawPlane() {{
      if (currentSong.has_true) {{
        drawPlanePoints(truePlaneCtx, truePlaneCanvas, 'true_valence', 'true_arousal', 'true_emotion');
      }}
      drawPlanePoints(predPlaneCtx, predPlaneCanvas, 'pred_valence', 'pred_arousal', 'pred_emotion');
      if (!currentSong.has_true) {{
        drawExternalTarget(predPlaneCtx, predPlaneCanvas);
      }}
    }}

    function timelinePoint(index, value) {{
      const pad = {{left: 62, right: 24, top: 24, bottom: 52}};
      const w = timelineCanvas.width - pad.left - pad.right;
      const h = timelineCanvas.height - pad.top - pad.bottom;
      const x = pad.left + (index / Math.max(1, currentSong.rows.length - 1)) * w;
      const y = pad.top + ((1 - value) / 2) * h;
      return [x, y];
    }}

    function drawSeries(ctx, key, color, untilIndex) {{
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      currentSong.rows.forEach((row, i) => {{
        if (i > untilIndex) return;
        const [x, y] = timelinePoint(i, displayValue(i, key));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.stroke();
    }}

    function drawTargetLine(ctx, value, color) {{
      const pad = {{left: 62, right: 24, top: 24, bottom: 52}};
      const w = timelineCanvas.width - pad.left - pad.right;
      const [, y] = timelinePoint(0, value);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + w, y); ctx.stroke();
      ctx.setLineDash([]);
    }}

    function drawTimeline() {{
      const ctx = timelineCtx;
      ctx.clearRect(0, 0, timelineCanvas.width, timelineCanvas.height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, timelineCanvas.width, timelineCanvas.height);
      const pad = {{left: 62, right: 24, top: 24, bottom: 52}};
      const w = timelineCanvas.width - pad.left - pad.right;
      const h = timelineCanvas.height - pad.top - pad.bottom;

      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      for (let v = -1; v <= 1; v += 0.5) {{
        const [, y] = timelinePoint(0, v);
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + w, y); ctx.stroke();
      }}

      const [, zeroY] = timelinePoint(0, 0);
      ctx.strokeStyle = '#94a3b8';
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(pad.left, zeroY); ctx.lineTo(pad.left + w, zeroY); ctx.stroke();
      ctx.setLineDash([]);

      if (currentSong.has_true) {{
        drawSeries(ctx, 'true_valence', '#2563eb', currentIndex);
        drawSeries(ctx, 'true_arousal', '#60a5fa', currentIndex);
      }} else if (typeof currentSong.target_valence_deam === 'number') {{
        drawTargetLine(ctx, currentSong.target_valence_deam, '#111827');
        drawTargetLine(ctx, currentSong.target_arousal_deam, '#111827');
      }}
      drawSeries(ctx, 'pred_valence', '#ef4444', currentIndex);
      drawSeries(ctx, 'pred_arousal', '#fb923c', currentIndex);

      const [cursorX] = timelinePoint(currentIndex, 0);
      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(cursorX, pad.top); ctx.lineTo(cursorX, pad.top + h); ctx.stroke();

      ctx.fillStyle = '#111827';
      ctx.font = '13px Segoe UI, Arial';
      ctx.fillText('-1', 28, pad.top + h + 4);
      ctx.fillText('0', 38, zeroY + 4);
      ctx.fillText('1', 38, pad.top + 4);

      const firstTime = currentSong.rows[0]?.time_s ?? 0;
      const lastTime = currentSong.rows[currentSong.rows.length - 1]?.time_s ?? firstTime;
      ctx.fillStyle = '#475569';
      ctx.font = '12px Segoe UI, Arial';
      ctx.fillText(`${{firstTime.toFixed(1)}} s`, pad.left - 8, pad.top + h + 20);
      ctx.fillText(`${{lastTime.toFixed(1)}} s`, pad.left + w - 34, pad.top + h + 20);

      const currentTime = currentSong.rows[currentIndex]?.time_s ?? 0;
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
      ctx.fillText('Wartość valence/arousal [-1, 1]', 0, 0);
      ctx.restore();
    }}

    function renderStrips() {{
      if (currentSong.has_true) {{
        renderStrip('trueStrip', 'true_emotion');
      }}
      renderStrip('predStrip', 'pred_emotion');
      updateStrips();
    }}

    function renderStrip(id, key) {{
      const root = document.getElementById(id);
      root.innerHTML = '';
      currentSong.rows.forEach((row, index) => {{
        const div = document.createElement('div');
        div.className = 'strip-segment';
        div.dataset.index = index;
        div.dataset.color = emotionColor(row[key]);
        div.dataset.emotion = row[key];
        div.style.width = `${{100 / currentSong.rows.length}}%`;
        div.style.background = '#e2e8f0';
        div.title = `${{index + 1}}: ${{row[key]}}`;
        root.appendChild(div);
      }});
    }}

    function updateStrip(id) {{
      const segments = document.getElementById(id).children;
      Array.from(segments).forEach((segment, index) => {{
        const isVisible = index <= currentIndex;
        segment.style.background = isVisible ? segment.dataset.color : '#e2e8f0';
        segment.classList.toggle('current', index === currentIndex);
      }});
    }}

    function updateStrips() {{
      if (currentSong.has_true) {{
        updateStrip('trueStrip');
      }}
      updateStrip('predStrip');
    }}

    function draw() {{
      if (!currentSong || !currentSong.rows.length) return;
      const row = currentSong.rows[currentIndex];
      document.getElementById('segmentText').textContent = `${{currentIndex + 1}} / ${{currentSong.rows.length}}`;
      document.getElementById('timeText').textContent = `${{row.time_s.toFixed(1)}} s`;
      if (currentSong.has_true) {{
        document.getElementById('trueStatLabel').textContent = 'True emotion';
        document.getElementById('trueEmotionText').textContent = row.true_emotion;
        document.getElementById('finalStatLabel').textContent = 'Accuracy final';
      }} else {{
        document.getElementById('trueStatLabel').textContent = 'Spotify target';
        document.getElementById('trueEmotionText').textContent =
          `${{currentSong.target_valence_spotify.toFixed(2)}} / ${{currentSong.target_arousal_spotify.toFixed(2)}}`;
        document.getElementById('finalStatLabel').textContent = 'Mean pred / MAE';
      }}
      document.getElementById('predEmotionText').textContent = row.pred_emotion;
      if (currentSong.has_true) {{
        const final = Math.round(currentSong.emotion_accuracy * 100);
        document.getElementById('accuracyText').textContent = trackCompleted ? `${{final}}%` : '-';
      }} else {{
        document.getElementById('accuracyText').textContent = trackCompleted
          ? `${{currentSong.mean_pred_valence_spotify.toFixed(3)}} / ${{currentSong.mean_pred_arousal_spotify.toFixed(3)}} | ${{currentSong.mean_abs_error.toFixed(3)}}`
          : '-';
      }}
      updateStrips();
      drawPlane();
      drawTimeline();
    }}

    songSelect.addEventListener('change', () => loadSong(songSelect.value));
    playBtn.addEventListener('click', () => timer ? stop() : play());
    frameSlider.addEventListener('input', () => {{
      currentIndex = Number(frameSlider.value);
      trackCompleted = false;
      syncAudioToCurrentSegment();
      draw();
    }});
    speedSelect.addEventListener('change', () => {{
      audioPlayer.playbackRate = 900 / Number(speedSelect.value);
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


def main() -> None:
    args = parse_args()
    df = load_predictions(args.predictions, args.song_id, args.max_songs)
    audio_dir = None if args.no_audio else args.audio_dir
    payload = build_payload(df, audio_dir, args.output)
    if not args.no_external:
        add_external_track(
            payload,
            predictions_path=args.external_predictions,
            summary_path=args.external_summary,
            audio_path=args.external_audio,
            output_path=args.output,
        )

    if not payload["song_ids"]:
        raise RuntimeError("No songs available for visualization.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(payload, args.predictions), encoding="utf-8")
    print(f"Saved live visualization to: {args.output}")
    print(f"Songs included: {len(payload['song_ids'])}")


if __name__ == "__main__":
    main()
