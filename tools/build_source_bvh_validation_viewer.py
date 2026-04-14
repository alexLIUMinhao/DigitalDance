#!/usr/bin/env python3
"""Build a self-contained HTML viewer for official-vs-source BVH validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "music-motion-lab" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_motion_lab.source_bvh_validation import (  # noqa: E402
    BRIDGE_BONE_ORDER_22,
    PARENT_BY_BONE_22,
    canonicalize_preview_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-id", default="168", help="Sequence id.")
    parser.add_argument(
        "--official-joints",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "previewCache" / "upstream_baselines" / "finedance" / "168" / "finedance_168_official_mesh_joints.json",
        help="Official mesh joints json.",
    )
    parser.add_argument(
        "--fk-joints",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "previewCache" / "layer_compare" / "finedance" / "168" / "finedance_168_source_fk_joints.json",
        help="Source FK joints json.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "previewCache" / "layer_compare" / "finedance" / "168" / "finedance_168_source_validation_report.json",
        help="Validation report json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "previewCache" / "layer_compare" / "finedance" / "168" / "finedance_168_source_validation_viewer.html",
        help="Output html path.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _preview_space(payload: dict) -> list:
    preview = payload.get("previewSpacePoses")
    if preview:
        return preview
    positions = np.asarray(payload["poses"], dtype=np.float64)
    return canonicalize_preview_positions(positions).round(decimals=8).tolist()


def _safe_video_uri(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        return ""
    return path.as_uri()


def build_viewer_document(
    *,
    sequence_id: str,
    official_payload: dict,
    fk_payload: dict,
    report_payload: dict,
) -> str:
    frame_count = min(len(official_payload.get("frameIndices", [])), len(fk_payload.get("frameIndices", [])))
    fps = float(official_payload.get("previewFps", 30) or 30)
    duration_seconds = frame_count / fps if fps > 0 else 0.0
    bone_order = list(official_payload.get("boneOrder", []) or fk_payload.get("boneOrder", []) or BRIDGE_BONE_ORDER_22)
    edges = [
        {"parent": parent_name, "child": child_name}
        for child_name, parent_name in PARENT_BY_BONE_22.items()
        if parent_name in bone_order and child_name in bone_order
    ]
    payload = {
        "sequenceId": str(sequence_id),
        "fps": fps,
        "frameCount": frame_count,
        "durationSeconds": duration_seconds,
        "boneOrder": bone_order,
        "edges": edges,
        "officialVideoUri": _safe_video_uri(str(official_payload.get("sourceVideoPath", ""))),
        "official": {
            "previewSpacePoses": _preview_space(official_payload)[:frame_count],
        },
        "sourceFk": {
            "previewSpacePoses": _preview_space(fk_payload)[:frame_count],
        },
        "report": report_payload,
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FineDance {sequence_id} Source BVH Validation Viewer</title>
  <style>
    :root {{
      --bg: #07131f;
      --panel: #102236;
      --panel-2: #0c1b2c;
      --line: rgba(255,255,255,0.1);
      --text: #eef5fb;
      --muted: #9fb4c7;
      --accent: #ff7a4f;
      --accent-2: #54b3ff;
      --good: #52d273;
      --warn: #ffd166;
      --bad: #ff5d73;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #12304a 0%, var(--bg) 50%);
      color: var(--text);
    }}
    .shell {{
      width: min(1500px, calc(100vw - 32px));
      margin: 18px auto 48px;
      display: grid;
      gap: 16px;
    }}
    .hero, .panel {{
      background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px 20px;
      backdrop-filter: blur(10px);
    }}
    .hero h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0.02em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .stat {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
    }}
    .stat .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .stat .value {{
      margin-top: 6px;
      font-size: 24px;
      font-weight: 700;
    }}
    .video-panel video {{
      width: 100%;
      border-radius: 14px;
      background: #000;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto 1fr auto auto;
      gap: 12px;
      align-items: center;
    }}
    button, input[type="range"] {{
      width: 100%;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--accent), #ffae4f);
      color: #09131e;
      font-weight: 800;
      padding: 12px 18px;
      cursor: pointer;
    }}
    .frame-label, .toggle {{
      color: var(--muted);
      font-size: 14px;
    }}
    .view-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .canvas-card {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }}
    .canvas-card h2 {{
      margin: 0 0 10px;
      font-size: 16px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    canvas {{
      width: 100%;
      height: auto;
      aspect-ratio: 4 / 3;
      background: linear-gradient(180deg, #091520, #061019);
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.06);
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 16px;
    }}
    .list-card {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }}
    .list-card h3 {{
      margin: 0 0 10px;
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }}
    .list-card ol, .list-card ul {{
      margin: 0;
      padding-left: 20px;
      display: grid;
      gap: 8px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 6px 10px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.08);
      color: var(--muted);
      font-size: 13px;
      margin-right: 8px;
      margin-top: 8px;
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }}
    @media (max-width: 1000px) {{
      .stats, .view-grid, .detail-grid, .controls {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>FineDance {sequence_id} Source BVH Validation Viewer</h1>
      <p>Official mesh-preview joints vs source BVH direct FK, aligned to the same 60s / 30fps / 1800-frame window.</p>
      <div class="stats">
        <div class="stat"><div class="label">Frames</div><div class="value" id="frameCountValue"></div></div>
        <div class="stat"><div class="label">Duration</div><div class="value" id="durationValue"></div></div>
        <div class="stat"><div class="label">FK Mean Error</div><div class="value" id="meanErrorValue"></div></div>
        <div class="stat"><div class="label">Likely Root Cause</div><div class="value" id="diagnosisValue" style="font-size:16px; line-height:1.35;"></div></div>
      </div>
    </section>

    <section class="panel video-panel">
      <video id="officialVideo" controls preload="metadata"></video>
    </section>

    <section class="panel">
      <div class="controls">
        <button id="playPauseButton" type="button">Play</button>
        <input id="frameSlider" type="range" min="0" max="0" value="0" step="1" />
        <div class="frame-label" id="frameLabel">Frame 0 / 0</div>
        <label class="toggle"><input id="syncVideoToggle" type="checkbox" checked /> Sync Video</label>
      </div>
      <div id="summaryPills"></div>
    </section>

    <section class="view-grid">
      <div class="canvas-card">
        <h2>Official Joint Truth</h2>
        <canvas id="officialCanvas" width="840" height="630"></canvas>
      </div>
      <div class="canvas-card">
        <h2>Source BVH FK</h2>
        <canvas id="sourceCanvas" width="840" height="630"></canvas>
      </div>
    </section>

    <section class="detail-grid">
      <div class="list-card">
        <h3>Current Frame Joint Error</h3>
        <ol id="currentFrameJointErrors"></ol>
      </div>
      <div class="list-card">
        <h3>Top Bone Direction Mismatch</h3>
        <ol id="boneDirectionErrors"></ol>
      </div>
      <div class="list-card">
        <h3>Top Bone Length Mismatch</h3>
        <ol id="boneLengthErrors"></ol>
      </div>
      <div class="list-card">
        <h3>Notes</h3>
        <p class="note" id="notesText"></p>
      </div>
    </section>
  </div>

  <script id="validationPayload" type="application/json">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('validationPayload').textContent);
    const officialFrames = payload.official.previewSpacePoses;
    const sourceFrames = payload.sourceFk.previewSpacePoses;
    const frameCount = payload.frameCount;
    const fps = payload.fps || 30;
    const durationSeconds = payload.durationSeconds || (frameCount / fps);
    const boneOrder = payload.boneOrder;
    const edges = payload.edges;
    const jointIndex = new Map(boneOrder.map((name, index) => [name, index]));
    const report = payload.report || {{}};
    const fkReport = (report.comparisonResults || {{}}).fk_direct || {{}};
    const boneDiagnostics = fkReport.boneVectorDiagnostics || {{}};
    const meanError = ((fkReport.jointPositionError || {{}}).mean || 0).toFixed(4);
    const p95Error = ((fkReport.jointPositionError || {{}}).p95 || 0).toFixed(4);
    const diagnosis = ((report.diagnosis || {{}}).likely_root_cause || 'unknown').replaceAll('_', ' ');
    const diagnosisReason = (report.diagnosis || {{}}).reason || 'No diagnosis available.';

    const officialCanvas = document.getElementById('officialCanvas');
    const sourceCanvas = document.getElementById('sourceCanvas');
    const officialCtx = officialCanvas.getContext('2d');
    const sourceCtx = sourceCanvas.getContext('2d');
    const slider = document.getElementById('frameSlider');
    const frameLabel = document.getElementById('frameLabel');
    const playPauseButton = document.getElementById('playPauseButton');
    const video = document.getElementById('officialVideo');
    const syncVideoToggle = document.getElementById('syncVideoToggle');
    const currentFrameJointErrors = document.getElementById('currentFrameJointErrors');
    const boneDirectionErrors = document.getElementById('boneDirectionErrors');
    const boneLengthErrors = document.getElementById('boneLengthErrors');
    const summaryPills = document.getElementById('summaryPills');

    document.getElementById('frameCountValue').textContent = String(frameCount);
    document.getElementById('durationValue').textContent = `${{durationSeconds.toFixed(2)}}s @ ${{fps.toFixed(0)}}fps`;
    document.getElementById('meanErrorValue').textContent = `${{meanError}} m`;
    document.getElementById('diagnosisValue').textContent = diagnosis;
    document.getElementById('notesText').textContent = `${{diagnosisReason}} Blender import status: ${{((report.comparisonResults || {{}}).blender_import || {{}}).status || 'not_run'}}. FK p95 joint error: ${{p95Error}} m.`;

    slider.max = String(Math.max(0, frameCount - 1));
    if (payload.officialVideoUri) {{
      video.src = payload.officialVideoUri;
    }} else {{
      video.style.display = 'none';
    }}

    const pills = [
      `FK mean: ${{meanError}} m`,
      `FK p95: ${{p95Error}} m`,
      `Top mismatch: ${{(boneDiagnostics.topByDirectionAngleP95 || [])[0]?.bone || 'n/a'}}`,
    ];
    for (const text of pills) {{
      const span = document.createElement('span');
      span.className = 'pill';
      span.textContent = text;
      summaryPills.appendChild(span);
    }}

    let currentFrame = 0;
    let playing = false;
    let rafId = null;
    let lastTickMs = 0;

    function project(point, bounds, width, height) {{
      const x = point[0];
      const depth = point[1];
      const y = point[2];
      const px = x + depth * 0.35;
      const py = y - depth * 0.12;
      const normalizedX = (px - bounds.minX) / Math.max(bounds.maxX - bounds.minX, 1e-5);
      const normalizedY = (py - bounds.minY) / Math.max(bounds.maxY - bounds.minY, 1e-5);
      return [
        48 + normalizedX * (width - 96),
        height - (48 + normalizedY * (height - 96)),
      ];
    }}

    function computeBounds() {{
      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      for (const sequence of [officialFrames, sourceFrames]) {{
        for (const frame of sequence) {{
          for (const point of frame) {{
            const px = point[0] + point[1] * 0.35;
            const py = point[2] - point[1] * 0.12;
            minX = Math.min(minX, px);
            maxX = Math.max(maxX, px);
            minY = Math.min(minY, py);
            maxY = Math.max(maxY, py);
          }}
        }}
      }}
      return {{ minX, maxX, minY, maxY }};
    }}

    const bounds = computeBounds();

    function jointErrorAtFrame(frameIndex) {{
      const official = officialFrames[frameIndex];
      const source = sourceFrames[frameIndex];
      return boneOrder.map((name, index) => {{
        const a = official[index];
        const b = source[index];
        const dx = a[0] - b[0];
        const dy = a[1] - b[1];
        const dz = a[2] - b[2];
        return {{
          bone: name,
          error: Math.sqrt(dx * dx + dy * dy + dz * dz),
        }};
      }});
    }}

    function colorForError(error) {{
      const t = Math.max(0, Math.min(1, error / 0.8));
      const r = Math.round(84 + (255 - 84) * t);
      const g = Math.round(179 + (93 - 179) * t);
      const b = Math.round(255 + (115 - 255) * t);
      return `rgb(${{r}}, ${{g}}, ${{b}})`;
    }}

    function drawSkeleton(ctx, frame, options) {{
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.fillStyle = '#07131f';
      ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.lineWidth = 1;
      for (let i = 0; i < 6; i += 1) {{
        const y = 48 + (ctx.canvas.height - 96) * (i / 5);
        ctx.beginPath();
        ctx.moveTo(36, y);
        ctx.lineTo(ctx.canvas.width - 36, y);
        ctx.stroke();
      }}

      const projected = frame.map((point) => project(point, bounds, ctx.canvas.width, ctx.canvas.height));
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (const edge of edges) {{
        const parentIndex = jointIndex.get(edge.parent);
        const childIndex = jointIndex.get(edge.child);
        if (parentIndex == null || childIndex == null) continue;
        const a = projected[parentIndex];
        const b = projected[childIndex];
        ctx.beginPath();
        ctx.strokeStyle = options.edgeColor;
        ctx.lineWidth = 5;
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
      }}
      for (let index = 0; index < projected.length; index += 1) {{
        const point = projected[index];
        ctx.beginPath();
        ctx.fillStyle = typeof options.jointColor === 'function' ? options.jointColor(index) : options.jointColor;
        ctx.arc(point[0], point[1], 7, 0, Math.PI * 2);
        ctx.fill();
      }}
    }}

    function renderFrame(frameIndex) {{
      currentFrame = Math.max(0, Math.min(frameCount - 1, frameIndex));
      slider.value = String(currentFrame);
      frameLabel.textContent = `Frame ${{currentFrame + 1}} / ${{frameCount}}  •  t=${{(currentFrame / fps).toFixed(2)}}s`;

      const currentErrors = jointErrorAtFrame(currentFrame).sort((a, b) => b.error - a.error);
      drawSkeleton(officialCtx, officialFrames[currentFrame], {{
        edgeColor: 'rgba(84,179,255,0.8)',
        jointColor: '#7dd6ff',
      }});
      drawSkeleton(sourceCtx, sourceFrames[currentFrame], {{
        edgeColor: 'rgba(255,122,79,0.82)',
        jointColor: (index) => colorForError(currentErrors.find((entry) => entry.bone === boneOrder[index])?.error || 0),
      }});

      currentFrameJointErrors.replaceChildren();
      for (const item of currentErrors.slice(0, 8)) {{
        const li = document.createElement('li');
        li.textContent = `${{item.bone}}  •  ${{item.error.toFixed(4)}} m`;
        currentFrameJointErrors.appendChild(li);
      }}
    }}

    function renderTopLists() {{
      boneDirectionErrors.replaceChildren();
      for (const item of boneDiagnostics.topByDirectionAngleP95 || []) {{
        const li = document.createElement('li');
        li.textContent = `${{item.parent}} -> ${{item.bone}}  •  p95 ${{item.p95.toFixed(2)}}°`;
        boneDirectionErrors.appendChild(li);
      }}
      boneLengthErrors.replaceChildren();
      for (const item of boneDiagnostics.topByLengthErrorMean || []) {{
        const li = document.createElement('li');
        li.textContent = `${{item.parent}} -> ${{item.bone}}  •  mean ${{item.mean.toFixed(4)}} m`;
        boneLengthErrors.appendChild(li);
      }}
    }}

    function step(timestamp) {{
      if (!playing) return;
      if (syncVideoToggle.checked && !video.paused && Number.isFinite(video.currentTime)) {{
        renderFrame(Math.min(frameCount - 1, Math.round(video.currentTime * fps)));
      }} else {{
        if (!lastTickMs) lastTickMs = timestamp;
        const delta = timestamp - lastTickMs;
        if (delta >= (1000 / fps)) {{
          lastTickMs = timestamp;
          if (currentFrame >= frameCount - 1) {{
            playing = false;
            playPauseButton.textContent = 'Play';
          }} else {{
            renderFrame(currentFrame + 1);
          }}
        }}
      }}
      if (playing) rafId = window.requestAnimationFrame(step);
    }}

    function setPlaying(nextPlaying) {{
      playing = nextPlaying;
      playPauseButton.textContent = playing ? 'Pause' : 'Play';
      if (playing) {{
        if (syncVideoToggle.checked && video.src) {{
          video.currentTime = currentFrame / fps;
          video.play().catch(() => null);
        }}
        lastTickMs = 0;
        rafId = window.requestAnimationFrame(step);
      }} else {{
        if (!video.paused) video.pause();
        if (rafId) {{
          window.cancelAnimationFrame(rafId);
          rafId = null;
        }}
      }}
    }}

    slider.addEventListener('input', () => {{
      renderFrame(Number(slider.value));
      if (syncVideoToggle.checked && video.src) {{
        video.currentTime = currentFrame / fps;
      }}
    }});
    playPauseButton.addEventListener('click', () => setPlaying(!playing));
    video.addEventListener('timeupdate', () => {{
      if (syncVideoToggle.checked) {{
        renderFrame(Math.min(frameCount - 1, Math.round(video.currentTime * fps)));
      }}
    }});
    video.addEventListener('ended', () => setPlaying(false));

    renderTopLists();
    renderFrame(0);
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    official_payload = _load_json(args.official_joints.expanduser().resolve())
    fk_payload = _load_json(args.fk_joints.expanduser().resolve())
    report_payload = _load_json(args.report.expanduser().resolve())
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_viewer_document(
            sequence_id=str(args.sequence_id),
            official_payload=official_payload,
            fk_payload=fk_payload,
            report_payload=report_payload,
        ),
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
