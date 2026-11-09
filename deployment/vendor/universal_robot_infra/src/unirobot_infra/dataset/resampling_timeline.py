from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from unirobot_infra.dataset.convert_config import ConvertTactileForceConfig, ConvertTactileImageConfig, FieldSpec
from unirobot_infra.dataset.episode_reader import EpisodeReader
from unirobot_infra.dataset.resampler import resample_episode


BACKGROUND = "#fafaf9"
GRID_COLOR = "#1f2937"
STALE_COLOR = "#dc2626"
TRACK_LABEL = "#111827"
AXIS_COLOR = "#6b7280"


@dataclass(frozen=True)
class StreamSpec:
    name: str
    color: str
    raw_times: np.ndarray
    matched_times: np.ndarray
    gaps: np.ndarray
    stale_mask: np.ndarray


@dataclass(frozen=True)
class TimelinePayload:
    episode_name: str
    target_fps: int
    grid: list[float]
    streams: list[dict[str, object]]
    summary: list[str]


def render_resampling_timeline_html(
    episode_dir: str | Path,
    target_fps: int,
    *,
    state_keys: tuple[FieldSpec, ...] | None = None,
    camera_names: tuple[str, ...] | None = None,
    tactile_images: tuple[ConvertTactileImageConfig, ...] = (),
    tactile_force: ConvertTactileForceConfig | None = None,
    include_action: bool = False,
) -> str:
    payload = _timeline_payload(
        episode_dir,
        int(target_fps),
        state_keys=state_keys,
        camera_names=camera_names,
        tactile_images=tactile_images,
        tactile_force=tactile_force,
        include_action=include_action,
    )
    return _html_document(payload)


def write_resampling_timeline_html(
    episode_dir: str | Path,
    target_fps: int,
    *,
    state_keys: tuple[FieldSpec, ...] | None = None,
    camera_names: tuple[str, ...] | None = None,
    tactile_images: tuple[ConvertTactileImageConfig, ...] = (),
    tactile_force: ConvertTactileForceConfig | None = None,
    include_action: bool = False,
) -> Path:
    episode_path = Path(episode_dir)
    output_path = episode_path / f"resampling_timeline_{int(target_fps)}hz.html"
    output_path.write_text(
        render_resampling_timeline_html(
            episode_path,
            int(target_fps),
            state_keys=state_keys,
            camera_names=camera_names,
            tactile_images=tactile_images,
            tactile_force=tactile_force,
            include_action=include_action,
        ),
        encoding="utf-8",
    )
    return output_path


def _timeline_payload(
    episode_dir: str | Path,
    target_fps: int,
    *,
    state_keys: tuple[FieldSpec, ...] | None,
    camera_names: tuple[str, ...] | None,
    tactile_images: tuple[ConvertTactileImageConfig, ...],
    tactile_force: ConvertTactileForceConfig | None,
    include_action: bool,
) -> TimelinePayload:
    episode_path = Path(episode_dir)
    with EpisodeReader(episode_path) as reader:
        state_specs = _default_state_keys(reader) if state_keys is None else state_keys
        cameras = reader.camera_names if camera_names is None else camera_names
        grid, streams = _build_streams(
            reader,
            int(target_fps),
            state_keys=state_specs,
            camera_names=cameras,
            tactile_images=tactile_images,
            tactile_force=tactile_force,
            include_action=include_action,
        )

    if len(grid) == 0:
        raise ValueError("Resampling grid is empty")
    base_time = float(grid[0])
    start_time = base_time
    end_time = float(grid[-1])
    if start_time >= end_time:
        raise ValueError("Episode duration is empty after resampling")

    summary = [
        f"grid: frames={len(grid)}, start={_fmt_s(grid[0] - base_time)}, end={_fmt_s(grid[-1] - base_time)}",
        *(_stream_summary(stream) for stream in streams),
    ]
    return TimelinePayload(
        episode_name=episode_path.name,
        target_fps=int(target_fps),
        grid=_relative_seconds(grid, base_time).tolist(),
        streams=[_stream_payload(stream, base_time) for stream in streams],
        summary=summary,
    )


def _build_streams(
    reader: EpisodeReader,
    target_fps: int,
    *,
    state_keys: tuple[FieldSpec, ...],
    camera_names: tuple[str, ...],
    tactile_images: tuple[ConvertTactileImageConfig, ...],
    tactile_force: ConvertTactileForceConfig | None,
    include_action: bool,
) -> tuple[np.ndarray, list[StreamSpec]]:
    resampled = resample_episode(
        reader,
        target_fps,
        state_keys=state_keys,
        camera_names=camera_names,
        tactile_images=tactile_images,
        tactile_force=tactile_force,
        include_action=include_action,
    )
    grid = resampled.timestamps
    streams = []
    palette = _palette()

    for index, stream_name in enumerate(_unique_streams(state_keys)):
        match = resampled.matches[f"state:{stream_name}"]
        streams.append(
            StreamSpec(
                name=f"state:{stream_name}",
                color=palette[index % len(palette)],
                raw_times=match.raw_times,
                matched_times=match.raw_times[match.indices],
                gaps=match.gaps,
                stale_mask=match.stale_mask,
            )
        )

    offset = len(streams)
    for index, camera_name in enumerate(camera_names):
        match = resampled.matches[f"camera:{camera_name}"]
        streams.append(
            StreamSpec(
                name=f"camera:{camera_name}",
                color=palette[(offset + index) % len(palette)],
                raw_times=match.raw_times,
                matched_times=match.raw_times[match.indices],
                gaps=match.gaps,
                stale_mask=match.stale_mask,
            )
        )

    offset = len(streams)
    for index, tactile_image in enumerate(tactile_images):
        match = resampled.matches[f"tactile:{tactile_image.name}"]
        streams.append(
            StreamSpec(
                name=f"tactile:{tactile_image.name}",
                color=palette[(offset + index) % len(palette)],
                raw_times=match.raw_times,
                matched_times=match.raw_times[match.indices],
                gaps=match.gaps,
                stale_mask=match.stale_mask,
            )
        )

    if tactile_force is not None:
        offset = len(streams)
        for index, sensor in enumerate(tactile_force.sensors):
            match = resampled.matches[f"tactile_force:{sensor.name}"]
            streams.append(
                StreamSpec(
                    name=f"tactile_force:{sensor.name}",
                    color=palette[(offset + index) % len(palette)],
                    raw_times=match.raw_times,
                    matched_times=match.raw_times[match.indices],
                    gaps=match.gaps,
                    stale_mask=match.stale_mask,
                )
            )

    if include_action:
        match = resampled.matches["action"]
        streams.append(
            StreamSpec(
                name="action",
                color=palette[len(streams) % len(palette)],
                raw_times=match.raw_times,
                matched_times=match.raw_times[match.indices],
                gaps=match.gaps,
                stale_mask=match.stale_mask,
            )
        )

    return grid, streams


def _default_state_keys(reader: EpisodeReader) -> tuple[FieldSpec, ...]:
    return tuple(FieldSpec(stream=stream_name, field="sys_time_ns") for stream_name in reader.stream_names)


def _unique_streams(state_keys: tuple[FieldSpec, ...]) -> tuple[str, ...]:
    streams = []
    for spec in state_keys:
        if spec.stream not in streams:
            streams.append(spec.stream)
    return tuple(streams)


def _palette() -> tuple[str, ...]:
    return (
        "#2563eb",
        "#ea580c",
        "#059669",
        "#7c3aed",
        "#db2777",
        "#0891b2",
        "#ca8a04",
        "#4f46e5",
    )


def _fmt_s(seconds: float) -> str:
    return f"{seconds:.3f}s"


def _stream_summary(stream: StreamSpec) -> str:
    return (
        f"{stream.name}: raw={len(stream.raw_times)}, matched={len(stream.matched_times)}, "
        f"gap mean={stream.gaps.mean() * 1000:.1f} ms, gap max={stream.gaps.max() * 1000:.1f} ms"
    )


def _relative_seconds(values: np.ndarray, base_time: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) - float(base_time)


def _stream_payload(stream: StreamSpec, base_time: float) -> dict[str, object]:
    return {
        "name": stream.name,
        "color": stream.color,
        "rawTimes": _relative_seconds(stream.raw_times, base_time).tolist(),
        "matchedTimes": _relative_seconds(stream.matched_times, base_time).tolist(),
        "gapsMs": (np.asarray(stream.gaps, dtype=np.float64) * 1000.0).tolist(),
        "stale": np.asarray(stream.stale_mask, dtype=bool).tolist(),
    }


def _html_document(payload: TimelinePayload) -> str:
    payload_json = _script_json(asdict(payload))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Resampling timeline - {_escape_html(payload.episode_name)}</title>
<style>
:root {{
  color-scheme: light;
  --background: {BACKGROUND};
  --text: {TRACK_LABEL};
  --muted: {AXIS_COLOR};
  --line: #d1d5db;
  --panel: #ffffff;
  --border: #d6d3d1;
  --stale: {STALE_COLOR};
}}
* {{
  box-sizing: border-box;
}}
body {{
  margin: 0;
  background: var(--background);
  color: var(--text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 2;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  background: rgba(250, 250, 249, 0.94);
  backdrop-filter: blur(6px);
}}
h1 {{
  margin: 0;
  font-size: 16px;
  font-weight: 700;
}}
.meta {{
  margin-top: 4px;
  color: var(--muted);
  font-size: 12px;
}}
.controls {{
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}}
button {{
  height: 30px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--panel);
  color: var(--text);
  font: inherit;
  cursor: pointer;
}}
button:hover {{
  border-color: #a8a29e;
}}
main {{
  padding: 12px 16px 18px;
}}
#summary {{
  margin-bottom: 10px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
}}
.canvas-wrap {{
  position: relative;
  overflow: auto;
  border: 1px solid var(--border);
  background: var(--panel);
}}
canvas {{
  display: block;
  min-width: 100%;
}}
#tooltip {{
  position: fixed;
  z-index: 5;
  display: none;
  max-width: 420px;
  padding: 8px 10px;
  border: 1px solid #78716c;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.97);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.14);
  color: var(--text);
  font-size: 12px;
  line-height: 1.45;
  pointer-events: none;
  white-space: pre-wrap;
}}
.hint {{
  margin-top: 8px;
  color: var(--muted);
  font-size: 12px;
}}
</style>
</head>
<body>
<header>
  <div>
    <h1 id="title"></h1>
    <div class="meta" id="windowMeta"></div>
  </div>
  <div class="controls">
    <button id="zoomOut" type="button">Zoom Out</button>
    <button id="zoomIn" type="button">Zoom In</button>
    <button id="resetView" type="button">Reset</button>
    <button id="staleView" type="button">Stale</button>
  </div>
</header>
<main>
  <div id="summary"></div>
  <div class="canvas-wrap" id="canvasWrap">
    <canvas id="timeline"></canvas>
  </div>
  <div class="hint">Wheel or trackpad scroll zooms around the pointer. Drag horizontally to pan. Hover over marks for frame, timestamp, and gap details.</div>
</main>
<div id="tooltip"></div>
<script id="timeline-data" type="application/json">{payload_json}</script>
<script>
(() => {{
  "use strict";

  const payload = JSON.parse(document.getElementById("timeline-data").textContent);
  const canvas = document.getElementById("timeline");
  const wrap = document.getElementById("canvasWrap");
  const tooltip = document.getElementById("tooltip");
  const ctx = canvas.getContext("2d");

  const layout = {{
    leftPad: 250,
    rightPad: 40,
    topPad: 34,
    bottomPad: 54,
    trackHeight: 42,
    trackGap: 20,
    minPlotWidth: 1200,
    pointRadius: 3.2,
  }};
  const colors = {{
    background: "{BACKGROUND}",
    grid: "{GRID_COLOR}",
    stale: "{STALE_COLOR}",
    axis: "{AXIS_COLOR}",
    label: "{TRACK_LABEL}",
    line: "#d1d5db",
    guide: "#e5e7eb",
  }};

  let viewStart = 0;
  let viewEnd = Math.max(payload.grid[payload.grid.length - 1], 1 / payload.target_fps);
  let dragging = false;
  let dragStartX = 0;
  let dragStartView = [viewStart, viewEnd];

  document.getElementById("title").textContent =
    `${{payload.episode_name}} | target_fps=${{payload.target_fps}}`;
  document.getElementById("summary").textContent = payload.summary.join("\\n");
  document.getElementById("zoomOut").addEventListener("click", () => zoomAt(0.5, 2.0));
  document.getElementById("zoomIn").addEventListener("click", () => zoomAt(0.5, 0.5));
  document.getElementById("resetView").addEventListener("click", () => {{
    viewStart = 0;
    viewEnd = duration();
    draw();
  }});
  document.getElementById("staleView").addEventListener("click", () => {{
    const staleWindow = staleBounds();
    if (staleWindow === null) {{
      return;
    }}
    viewStart = staleWindow[0];
    viewEnd = staleWindow[1];
    draw();
  }});

  new ResizeObserver(() => draw()).observe(wrap);

  canvas.addEventListener("wheel", (event) => {{
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const ratio = clamp((x - layout.leftPad) / plotWidth(), 0, 1);
    zoomAt(ratio, event.deltaY < 0 ? 0.82 : 1.22);
  }}, {{ passive: false }});

  canvas.addEventListener("pointerdown", (event) => {{
    dragging = true;
    canvas.setPointerCapture(event.pointerId);
    dragStartX = event.clientX;
    dragStartView = [viewStart, viewEnd];
  }});

  canvas.addEventListener("pointermove", (event) => {{
    if (dragging) {{
      const dx = event.clientX - dragStartX;
      const span = dragStartView[1] - dragStartView[0];
      const dt = -dx / plotWidth() * span;
      setView(dragStartView[0] + dt, dragStartView[1] + dt);
      draw();
      return;
    }}
    showTooltip(event);
  }});

  canvas.addEventListener("pointerup", (event) => {{
    dragging = false;
    canvas.releasePointerCapture(event.pointerId);
  }});

  canvas.addEventListener("pointerleave", () => {{
    dragging = false;
    tooltip.style.display = "none";
  }});

  function duration() {{
    return Math.max(payload.grid[payload.grid.length - 1], 1 / payload.target_fps);
  }}

  function canvasHeight() {{
    const tracks = 1 + payload.streams.length;
    return layout.topPad + layout.bottomPad + tracks * layout.trackHeight + (tracks - 1) * layout.trackGap;
  }}

  function canvasWidth() {{
    return Math.max(wrap.clientWidth - 2, layout.leftPad + layout.rightPad + layout.minPlotWidth);
  }}

  function plotWidth() {{
    return Math.max(1, canvas.width / (window.devicePixelRatio || 1) - layout.leftPad - layout.rightPad);
  }}

  function draw() {{
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvasWidth();
    const cssHeight = canvasHeight();
    canvas.style.width = `${{cssWidth}}px`;
    canvas.style.height = `${{cssHeight}}px`;
    canvas.width = Math.floor(cssWidth * dpr);
    canvas.height = Math.floor(cssHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    ctx.fillStyle = colors.background;
    ctx.fillRect(0, 0, cssWidth, cssHeight);

    drawAxis(cssWidth, cssHeight);
    drawTrack("grid", 0, cssWidth);
    drawGridMarks();
    payload.streams.forEach((stream, index) => {{
      drawTrack(stream.name, index + 1, cssWidth);
      drawRawMarks(stream, index + 1);
      drawMatchedMarks(stream, index + 1);
    }});
    document.getElementById("windowMeta").textContent =
      `window=${{fmtS(viewStart)}} to ${{fmtS(viewEnd)}} | span=${{fmtS(viewEnd - viewStart)}}`;
  }}

  function drawAxis(cssWidth, cssHeight) {{
    const step = pickTickStep(viewEnd - viewStart);
    const first = Math.ceil(viewStart / step) * step;
    const top = trackY(0) - layout.trackHeight / 2;
    const bottom = trackY(payload.streams.length) + layout.trackHeight / 2;
    ctx.save();
    ctx.font = "11px ui-monospace, Menlo, Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let t = first; t <= viewEnd + 1e-9; t += step) {{
      const x = xPos(t);
      ctx.strokeStyle = colors.guide;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.fillStyle = colors.axis;
      ctx.fillText(fmtS(t), x, bottom + 18);
    }}
    ctx.restore();
  }}

  function drawTrack(name, index, cssWidth) {{
    const y = trackY(index);
    ctx.save();
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(layout.leftPad, y);
    ctx.lineTo(cssWidth - layout.rightPad, y);
    ctx.stroke();
    ctx.fillStyle = colors.label;
    ctx.font = "13px ui-monospace, Menlo, Consolas, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(name, layout.leftPad - 12, y);
    ctx.restore();
  }}

  function drawGridMarks() {{
    const y = trackY(0);
    ctx.save();
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1.6;
    forVisible(payload.grid, (time) => {{
      const x = xPos(time);
      ctx.beginPath();
      ctx.moveTo(x, y - 8);
      ctx.lineTo(x, y + 8);
      ctx.stroke();
    }});
    ctx.restore();
  }}

  function drawRawMarks(stream, trackIndex) {{
    const y = trackY(trackIndex);
    const times = visibleBucketed(stream.rawTimes);
    ctx.save();
    ctx.strokeStyle = stream.color;
    ctx.fillStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    for (const time of times) {{
      const x = xPos(time);
      ctx.beginPath();
      ctx.arc(x, y, layout.pointRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }}
    ctx.restore();
  }}

  function drawMatchedMarks(stream, trackIndex) {{
    const y = trackY(trackIndex);
    const start = lowerBound(stream.matchedTimes, viewStart);
    const stop = upperBound(stream.matchedTimes, viewEnd);
    const stride = displayStride(stop - start);
    ctx.save();
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    for (let i = start; i < stop; i += stride) {{
      drawMatchedLine(stream, i, y);
    }}
    if (stride > 1) {{
      for (let i = start; i < stop; i += 1) {{
        if (stream.stale[i]) {{
          drawMatchedLine(stream, i, y);
        }}
      }}
    }}
    ctx.restore();
  }}

  function drawMatchedLine(stream, index, y) {{
    const x = xPos(stream.matchedTimes[index]);
    ctx.strokeStyle = stream.stale[index] ? colors.stale : stream.color;
    ctx.beginPath();
    ctx.moveTo(x, y - 11);
    ctx.lineTo(x, y + 11);
    ctx.stroke();
  }}

  function showTooltip(event) {{
    const hit = hitTest(event);
    if (hit === null) {{
      tooltip.style.display = "none";
      return;
    }}
    tooltip.textContent = hit;
    tooltip.style.left = `${{Math.min(event.clientX + 14, window.innerWidth - 440)}}px`;
    tooltip.style.top = `${{Math.min(event.clientY + 14, window.innerHeight - 160)}}px`;
    tooltip.style.display = "block";
  }}

  function hitTest(event) {{
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const track = Math.round((y - layout.topPad - layout.trackHeight / 2) / (layout.trackHeight + layout.trackGap));
    if (track < 0 || track > payload.streams.length) {{
      return null;
    }}
    const trackDistance = Math.abs(y - trackY(track));
    if (trackDistance > 16) {{
      return null;
    }}
    const time = timeAtX(x);
    if (track === 0) {{
      const index = nearestIndex(payload.grid, time);
      if (index < 0 || Math.abs(xPos(payload.grid[index]) - x) > 6) {{
        return null;
      }}
      return `grid\\nframe=${{index}}\\ntime=${{fmtS(payload.grid[index])}}`;
    }}
    const stream = payload.streams[track - 1];
    const matchedIndex = nearestIndex(stream.matchedTimes, time);
    if (matchedIndex >= 0 && Math.abs(xPos(stream.matchedTimes[matchedIndex]) - x) <= 7) {{
      return [
        stream.name,
        `frame=${{matchedIndex}}`,
        `matched=${{fmtS(stream.matchedTimes[matchedIndex])}}`,
        `gap=${{stream.gapsMs[matchedIndex].toFixed(1)}} ms`,
        `stale=${{Boolean(stream.stale[matchedIndex])}}`,
      ].join("\\n");
    }}
    const rawIndex = nearestIndex(stream.rawTimes, time);
    if (rawIndex >= 0 && Math.abs(xPos(stream.rawTimes[rawIndex]) - x) <= 7) {{
      return [
        stream.name,
        `raw index=${{rawIndex}}`,
        `time=${{fmtS(stream.rawTimes[rawIndex])}}`,
      ].join("\\n");
    }}
    return null;
  }}

  function forVisible(times, callback) {{
    const start = lowerBound(times, viewStart);
    const stop = upperBound(times, viewEnd);
    for (let i = start; i < stop; i += 1) {{
      callback(times[i], i);
    }}
  }}

  function visibleBucketed(times) {{
    const start = lowerBound(times, viewStart);
    const stop = upperBound(times, viewEnd);
    const stride = Math.max(1, Math.ceil((stop - start) / Math.max(1, Math.floor(plotWidth()))));
    const output = [];
    for (let i = start; i < stop; i += stride) {{
      output.push(times[i]);
    }}
    return output;
  }}

  function displayStride(count) {{
    const maxLines = Math.max(1, Math.floor(plotWidth() * 1.5));
    return Math.max(1, Math.ceil(count / maxLines));
  }}

  function xPos(time) {{
    return layout.leftPad + (time - viewStart) / (viewEnd - viewStart) * plotWidth();
  }}

  function timeAtX(x) {{
    return viewStart + (x - layout.leftPad) / plotWidth() * (viewEnd - viewStart);
  }}

  function trackY(index) {{
    return layout.topPad + index * (layout.trackHeight + layout.trackGap) + layout.trackHeight / 2;
  }}

  function zoomAt(ratio, factor) {{
    const span = viewEnd - viewStart;
    const center = viewStart + span * ratio;
    const nextSpan = clamp(span * factor, 1 / payload.target_fps, duration());
    setView(center - nextSpan * ratio, center + nextSpan * (1 - ratio));
    draw();
  }}

  function setView(start, end) {{
    const total = duration();
    const span = Math.min(end - start, total);
    let nextStart = start;
    let nextEnd = start + span;
    if (nextStart < 0) {{
      nextEnd -= nextStart;
      nextStart = 0;
    }}
    if (nextEnd > total) {{
      nextStart -= nextEnd - total;
      nextEnd = total;
    }}
    viewStart = Math.max(0, nextStart);
    viewEnd = Math.min(total, nextEnd);
  }}

  function staleBounds() {{
    let minTime = Infinity;
    let maxTime = -Infinity;
    for (const stream of payload.streams) {{
      for (let i = 0; i < stream.stale.length; i += 1) {{
        if (stream.stale[i]) {{
          minTime = Math.min(minTime, stream.matchedTimes[i]);
          maxTime = Math.max(maxTime, stream.matchedTimes[i]);
        }}
      }}
    }}
    if (!Number.isFinite(minTime)) {{
      return null;
    }}
    const pad = Math.max(2 / payload.target_fps, (maxTime - minTime) * 0.2);
    return [Math.max(0, minTime - pad), Math.min(duration(), maxTime + pad)];
  }}

  function nearestIndex(values, target) {{
    if (values.length === 0) {{
      return -1;
    }}
    const right = lowerBound(values, target);
    if (right <= 0) {{
      return 0;
    }}
    if (right >= values.length) {{
      return values.length - 1;
    }}
    return Math.abs(values[right - 1] - target) <= Math.abs(values[right] - target) ? right - 1 : right;
  }}

  function lowerBound(values, target) {{
    let lo = 0;
    let hi = values.length;
    while (lo < hi) {{
      const mid = (lo + hi) >> 1;
      if (values[mid] < target) {{
        lo = mid + 1;
      }} else {{
        hi = mid;
      }}
    }}
    return lo;
  }}

  function upperBound(values, target) {{
    let lo = 0;
    let hi = values.length;
    while (lo < hi) {{
      const mid = (lo + hi) >> 1;
      if (values[mid] <= target) {{
        lo = mid + 1;
      }} else {{
        hi = mid;
      }}
    }}
    return lo;
  }}

  function pickTickStep(span) {{
    const candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600];
    const target = Math.max(span / 8, 1e-6);
    for (const step of candidates) {{
      if (step >= target) {{
        return step;
      }}
    }}
    return candidates[candidates.length - 1];
  }}

  function fmtS(value) {{
    return `${{value.toFixed(3)}}s`;
  }}

  function clamp(value, low, high) {{
    return Math.max(low, Math.min(high, value));
  }}

  draw();
}})();
</script>
</body>
</html>
"""


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _script_json(value) -> str:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
