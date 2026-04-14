from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..utils import utc_now_iso

PARENT_BY_BONE = {
    "left_hip": "pelvis",
    "right_hip": "pelvis",
    "spine1": "pelvis",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "spine2": "spine1",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "spine3": "spine2",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
    "neck": "spine3",
    "left_collar": "spine3",
    "right_collar": "spine3",
    "head": "neck",
    "left_shoulder": "left_collar",
    "right_shoulder": "right_collar",
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
}

DATASET_SOURCE_FPS = {
    "finedance": 30.0,
    "aistpp": 60.0,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _unity_to_view(coords: list[float]) -> tuple[float, float, float]:
    x, y, z = coords
    return (x, z, y)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_frame(frame_a: list[list[float]], frame_b: list[list[float]], t: float) -> list[list[float]]:
    blended: list[list[float]] = []
    for point_a, point_b in zip(frame_a, frame_b):
        blended.append(
            [
                _lerp(float(point_a[0]), float(point_b[0]), t),
                _lerp(float(point_a[1]), float(point_b[1]), t),
                _lerp(float(point_a[2]), float(point_b[2]), t),
            ]
        )
    return blended


def _resample_frames(frames: list[list[list[float]]], target_frame_count: int) -> list[list[list[float]]]:
    if not frames:
        return []
    if target_frame_count <= 1:
        return [frames[0]]
    if len(frames) == 1:
        return [frames[0] for _ in range(target_frame_count)]

    resampled: list[list[list[float]]] = []
    for frame_index in range(target_frame_count):
        position = frame_index * (len(frames) - 1) / (target_frame_count - 1)
        left_index = int(math.floor(position))
        right_index = min(left_index + 1, len(frames) - 1)
        mix = position - left_index
        resampled.append(_lerp_frame(frames[left_index], frames[right_index], mix))
    return resampled


def _rotate_frame_xz(frame: list[list[float]], angle_rad: float) -> list[list[float]]:
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    rotated: list[list[float]] = []
    for x, depth, height in frame:
        rotated.append(
            [
                x * cos_theta - depth * sin_theta,
                x * sin_theta + depth * cos_theta,
                height,
            ]
        )
    return rotated


def _translate_frame(frame: list[list[float]], delta: tuple[float, float, float]) -> list[list[float]]:
    dx, dz, dy = delta
    translated: list[list[float]] = []
    for x, depth, height in frame:
        translated.append([x + dx, depth + dz, height + dy])
    return translated


def _facing_angle(frame: list[list[float]], bone_order: list[str]) -> float:
    def point(name: str) -> list[float]:
        return frame[bone_order.index(name)]

    if all(name in bone_order for name in ("left_shoulder", "right_shoulder")):
        left = point("left_shoulder")
        right = point("right_shoulder")
    elif all(name in bone_order for name in ("left_hip", "right_hip")):
        left = point("left_hip")
        right = point("right_hip")
    else:
        return 0.0

    across_x = float(right[0]) - float(left[0])
    across_z = float(right[1]) - float(left[1])
    forward_x = -across_z
    forward_z = across_x
    if abs(forward_x) < 1e-6 and abs(forward_z) < 1e-6:
        return 0.0
    return math.atan2(forward_z, forward_x)


def _step_duration_sec(step: dict[str, Any], song_event_map: dict[str, Any]) -> tuple[float, float]:
    beats = list(song_event_map.get("beats", []))
    if not beats:
        return (0.0, 0.0)

    def beat_time(beat_index: float) -> float:
        if beat_index <= 0:
            return float(beats[0]["time_sec"])
        left_index = int(math.floor(beat_index))
        right_index = int(math.ceil(beat_index))
        if right_index >= len(beats):
            if len(beats) == 1:
                return float(beats[0]["time_sec"])
            interval = float(beats[-1]["time_sec"]) - float(beats[-2]["time_sec"])
            return float(beats[-1]["time_sec"]) + interval * float(beat_index - (len(beats) - 1))
        if left_index == right_index:
            return float(beats[left_index]["time_sec"])
        left_time = float(beats[left_index]["time_sec"])
        right_time = float(beats[right_index]["time_sec"])
        mix = beat_index - left_index
        return _lerp(left_time, right_time, mix)

    start_beat = float(step.get("start_beat", 0.0) or 0.0)
    target_beats = float(step.get("target_beats", 0.0) or 0.0)
    start_time = beat_time(start_beat)
    end_time = beat_time(start_beat + target_beats)
    return start_time, max(end_time, start_time + 1e-3)


def _extract_clip_frames(
    pose_driver_path: Path,
    start: int,
    end_exclusive: int,
    target_fps: int,
    explicit_frame_stride: int,
    target_frame_count: int | None,
) -> tuple[list[str], list[list[list[float]]]]:
    payload = _load_json(pose_driver_path)
    bone_order = list(payload["boneOrder"])
    joint_count = int(payload["jointCount"])
    frame_count = int(payload["frameCount"])
    flat_positions = list(payload["positions"])
    dataset_name = str(payload.get("datasetName", "") or "").strip().lower()
    source_fps = DATASET_SOURCE_FPS.get(dataset_name, 30.0)

    start = max(0, min(start, frame_count - 1))
    end_exclusive = max(start + 1, min(end_exclusive, frame_count))
    raw_frames: list[list[list[float]]] = []
    first_pelvis: tuple[float, float, float] | None = None

    for frame_index in range(start, end_exclusive):
        offset = frame_index * joint_count * 3
        frame: list[list[float]] = []
        for joint_index in range(joint_count):
            base = offset + joint_index * 3
            x, depth, height = _unity_to_view(flat_positions[base : base + 3])
            frame.append([x, depth, height])
        if first_pelvis is None:
            first_pelvis = (float(frame[0][0]), float(frame[0][1]), float(frame[0][2]))
        pelvis_x, pelvis_depth, pelvis_height = first_pelvis
        normalized: list[list[float]] = []
        for x, depth, height in frame:
            normalized.append([x - pelvis_x, depth - pelvis_depth, height - pelvis_height])
        raw_frames.append(normalized)

    if target_frame_count is not None:
        return bone_order, _resample_frames(raw_frames, max(2, target_frame_count))

    stride = explicit_frame_stride if explicit_frame_stride > 0 else max(1, int(round(source_fps / max(target_fps, 1))))
    sampled = raw_frames[::stride]
    if not sampled:
        sampled = [raw_frames[0]]
    return bone_order, sampled


def _align_clip_to_previous(
    previous_last_frame: list[list[float]],
    clip_frames: list[list[list[float]]],
    bone_order: list[str],
    transition_frames: int,
) -> list[list[list[float]]]:
    if not clip_frames:
        return clip_frames

    previous_facing = _facing_angle(previous_last_frame, bone_order)
    current_facing = _facing_angle(clip_frames[0], bone_order)
    rotation_delta = previous_facing - current_facing
    rotated_frames = [_rotate_frame_xz(frame, rotation_delta) for frame in clip_frames]

    previous_pelvis = previous_last_frame[0]
    translated_frames = [_translate_frame(frame, (previous_pelvis[0], previous_pelvis[1], previous_pelvis[2])) for frame in rotated_frames]

    blend_count = min(max(0, transition_frames), len(translated_frames))
    for blend_index in range(blend_count):
        mix = float(blend_index + 1) / float(blend_count + 1)
        translated_frames[blend_index] = _lerp_frame(previous_last_frame, translated_frames[blend_index], mix)
    return translated_frames


def build_web_preview_document(
    preview_job: dict[str, Any],
    plan: dict[str, Any] | None = None,
    song_event_map: dict[str, Any] | None = None,
    audio_href: str | None = None,
    max_steps: int = 6,
    frame_stride: int = 0,
    fps: int = 24,
    transition_frames: int = 6,
) -> str:
    bone_order: list[str] | None = None
    frames: list[list[list[float]]] = []
    frame_times_sec: list[float] = []
    step_markers: list[dict[str, Any]] = []
    plan_steps_by_index = {int(step["index"]): step for step in (plan or {}).get("steps", [])}
    selected_steps = list(preview_job.get("steps", []))
    if max_steps > 0:
        selected_steps = selected_steps[: max_steps]

    for preview_step in selected_steps:
        step_index = int(preview_step.get("index", 0))
        plan_step = plan_steps_by_index.get(step_index, {})
        start_time_sec = None
        end_time_sec = None
        target_frame_count = None
        if plan_step and song_event_map:
            start_time_sec, end_time_sec = _step_duration_sec(plan_step, song_event_map)
            target_frame_count = max(2, int(round((end_time_sec - start_time_sec) * fps)))

        pose_driver_path = Path(preview_step["reference_artifacts"]["pose_driver_json"])
        frame_range = dict(preview_step.get("frame_range", {}))
        current_bone_order, clip_frames = _extract_clip_frames(
            pose_driver_path=pose_driver_path,
            start=int(frame_range.get("start", 0)),
            end_exclusive=int(frame_range.get("end_exclusive", 1)),
            target_fps=fps,
            explicit_frame_stride=frame_stride,
            target_frame_count=target_frame_count,
        )
        if bone_order is None:
            bone_order = current_bone_order
        if current_bone_order != bone_order:
            raise ValueError(f"Incompatible bone order in {pose_driver_path}")

        if frames:
            clip_frames = _align_clip_to_previous(frames[-1], clip_frames, bone_order, transition_frames=transition_frames)

        start_frame_index = len(frames)
        frames.extend(clip_frames)
        if start_time_sec is not None and end_time_sec is not None:
            for local_index in range(len(clip_frames)):
                mix = float(local_index) / float(max(1, len(clip_frames) - 1))
                frame_times_sec.append(_lerp(start_time_sec, end_time_sec, mix))

        step_markers.append(
            {
                "index": preview_step.get("index"),
                "unit_id": preview_step.get("unit_id"),
                "source_sequence": preview_step.get("source_sequence"),
                "section_label": plan_step.get("section_label"),
                "start_frame": start_frame_index,
                "frame_count": len(clip_frames),
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
                "target_energy": plan_step.get("switch_reason", {}).get("target_energy"),
                "target_travel": plan_step.get("switch_reason", {}).get("target_travel"),
                "transition_mode": plan_step.get("switch_reason", {}).get("transition_mode"),
                "compatible_from_previous": plan_step.get("switch_reason", {}).get("compatible_from_previous"),
                "accent_count": plan_step.get("switch_reason", {}).get("accent_count", 0),
                "downbeat_count": plan_step.get("switch_reason", {}).get("downbeat_count", 0),
                "accent_density_per_beat": plan_step.get("switch_reason", {}).get("accent_density_per_beat", 0.0),
                "mean_accent_strength": plan_step.get("switch_reason", {}).get("mean_accent_strength", 0.0),
            }
        )

    if not bone_order or not frames:
        raise ValueError("No frames available for web preview")

    edges = [
        [bone_order.index(parent), bone_order.index(child)]
        for child, parent in PARENT_BY_BONE.items()
        if child in bone_order and parent in bone_order
    ]

    preview_start_time_sec = next(
        (float(marker["start_time_sec"]) for marker in step_markers if marker.get("start_time_sec") is not None),
        0.0,
    )
    preview_end_time_sec = max(
        [float(marker["end_time_sec"]) for marker in step_markers if marker.get("end_time_sec") is not None] or [preview_start_time_sec + (len(frames) / max(fps, 1))]
    )
    section_windows = []
    accent_windows: list[dict[str, Any]] = []
    downbeat_windows: list[dict[str, Any]] = []
    beat_windows: list[dict[str, Any]] = []
    if song_event_map:
        for section in song_event_map.get("sections", []):
            start_time_sec = float(section.get("start_time_sec", preview_start_time_sec) or preview_start_time_sec)
            end_time_sec = float(section.get("end_time_sec", preview_end_time_sec) or preview_end_time_sec)
            if end_time_sec < preview_start_time_sec or start_time_sec > preview_end_time_sec:
                continue
            section_windows.append(
                {
                    "label": section.get("label", "section"),
                    "start_time_sec": max(preview_start_time_sec, start_time_sec),
                    "end_time_sec": min(preview_end_time_sec, end_time_sec),
                }
            )
        accent_windows = [
            {
                "time_sec": float(item.get("time_sec", 0.0) or 0.0),
                "strength": float(item.get("strength", 0.0) or 0.0),
            }
            for item in song_event_map.get("accents", [])
            if preview_start_time_sec <= float(item.get("time_sec", -1.0) or -1.0) <= preview_end_time_sec
        ]
        downbeat_windows = [
            {"time_sec": float(item.get("time_sec", 0.0) or 0.0)}
            for item in song_event_map.get("downbeats", [])
            if preview_start_time_sec <= float(item.get("time_sec", -1.0) or -1.0) <= preview_end_time_sec
        ]
        beat_windows = [
            {
                "time_sec": float(item.get("time_sec", 0.0) or 0.0),
                "is_downbeat": bool(item.get("is_downbeat", False)),
            }
            for item in song_event_map.get("beats", [])
            if preview_start_time_sec <= float(item.get("time_sec", -1.0) or -1.0) <= preview_end_time_sec
        ]

    payload = {
        "plan_id": preview_job["plan_id"],
        "generated_at_utc": utc_now_iso(),
        "fps": fps,
        "audio_href": audio_href,
        "preview_start_time_sec": preview_start_time_sec,
        "preview_end_time_sec": preview_end_time_sec,
        "frame_times_sec": frame_times_sec,
        "bone_order": bone_order,
        "edges": edges,
        "frames": frames,
        "step_markers": step_markers,
        "sections": section_windows,
        "accents": accent_windows,
        "downbeats": downbeat_windows,
        "beats": beat_windows,
    }
    data_json = json.dumps(payload, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{preview_job['plan_id']} web preview</title>
  <style>
    :root {{
      --bg: #071118;
      --panel: rgba(14, 28, 42, 0.88);
      --ink: #eef4f7;
      --muted: #8da5b5;
      --accent: #ffb84d;
      --bone: #5ed4ff;
      --active: rgba(255, 184, 77, 0.16);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
      background:
        radial-gradient(circle at top, rgba(54, 102, 145, 0.36), transparent 34%),
        linear-gradient(160deg, #04090d 0%, #0a1620 45%, #132839 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0.02em;
    }}
    .meta {{
      color: var(--muted);
      margin-bottom: 18px;
      max-width: 880px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 24px 80px rgba(0,0,0,0.35);
      backdrop-filter: blur(12px);
    }}
    canvas {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 16px;
      background:
        radial-gradient(circle at 50% 8%, rgba(255,255,255,0.05), transparent 42%),
        linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
    }}
    .controls {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
      margin-top: 14px;
    }}
    button {{
      border: none;
      border-radius: 999px;
      padding: 11px 18px;
      background: var(--accent);
      color: #10151c;
      font-weight: 800;
      cursor: pointer;
    }}
    .badge {{
      display: inline-block;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      color: var(--muted);
    }}
    input[type="range"] {{
      width: min(440px, 100%);
    }}
    audio {{
      width: min(420px, 100%);
    }}
    #timeline {{
      margin-top: 14px;
      height: 120px;
    }}
    .steps {{
      margin-top: 16px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      max-height: 280px;
      overflow: auto;
      padding-right: 6px;
    }}
    .step {{
      padding: 13px;
      border-radius: 16px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.06);
      transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
    }}
    .step.active {{
      background: var(--active);
      border-color: rgba(255, 184, 77, 0.48);
      transform: translateY(-2px);
    }}
    .step strong {{
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Web Preview: {preview_job['plan_id']}</h1>
    <div class="meta">Music-aligned mesh-like preview with auto scaling, smoother step continuity, and optional audio sync. This is still a lightweight diagnostic renderer, but it should now read much closer to a character pass than a raw skeleton debug view.</div>
    <div class="panel">
      <canvas id="stage" width="1280" height="820"></canvas>
      <div class="controls">
        <button id="toggle">Pause</button>
        <span class="badge" id="frameLabel"></span>
        <span class="badge" id="stepLabel"></span>
        <span class="badge" id="timeLabel"></span>
        <input id="scrub" type="range" min="0" max="0" step="1" value="0">
        <audio id="audio" controls preload="auto"></audio>
      </div>
      <canvas id="timeline" width="1280" height="120"></canvas>
      <div class="steps" id="stepList"></div>
    </div>
  </div>
  <script>
    const preview = {data_json};
    const canvas = document.getElementById("stage");
    const ctx = canvas.getContext("2d");
    const timelineCanvas = document.getElementById("timeline");
    const timelineCtx = timelineCanvas.getContext("2d");
    const frameLabel = document.getElementById("frameLabel");
    const stepLabel = document.getElementById("stepLabel");
    const timeLabel = document.getElementById("timeLabel");
    const stepList = document.getElementById("stepList");
    const toggle = document.getElementById("toggle");
    const scrub = document.getElementById("scrub");
    const audio = document.getElementById("audio");
    let playing = true;
    let frameIndex = 0;
    let playbackTime = preview.frame_times_sec.length ? preview.frame_times_sec[0] : 0;
    let lastTickMs = null;
    let useAudioSync = Boolean(preview.audio_href && preview.frame_times_sec && preview.frame_times_sec.length === preview.frames.length);
    const azimuth = -0.52;
    const depthTilt = 0.22;
    const ENERGY_COLORS = {{
      low_energy: "#76d6a8",
      mid_energy: "#5ed4ff",
      high_energy: "#ff8c42",
    }};
    const boneIndex = Object.fromEntries(preview.bone_order.map((name, index) => [name, index]));

    if (preview.audio_href) {{
      audio.src = preview.audio_href;
      audio.currentTime = preview.preview_start_time_sec || 0;
    }} else {{
      audio.style.display = "none";
    }}

    const stepCards = [];
    for (const marker of preview.step_markers) {{
      const card = document.createElement("div");
      card.className = "step";
      const timeText = marker.start_time_sec == null ? "unsynced" : `${{marker.start_time_sec.toFixed(2)}}s -> ${{marker.end_time_sec.toFixed(2)}}s`;
      card.innerHTML = `<strong>step ${{marker.index}}</strong><br>section: ${{marker.section_label || "n/a"}}<br>energy: ${{marker.target_energy || "n/a"}}<br>unit: ${{marker.unit_id}}<br>transition: ${{marker.compatible_from_previous ? "compatible" : "fallback"}}<br>window: ${{timeText}}`;
      stepList.appendChild(card);
      stepCards.push(card);
    }}

    scrub.max = String(Math.max(0, preview.frames.length - 1));

    function rotatedPoint(point) {{
      const [x, depth, height] = point;
      return {{
        x: x * Math.cos(azimuth) - depth * Math.sin(azimuth),
        depth: x * Math.sin(azimuth) + depth * Math.cos(azimuth),
        y: height,
      }};
    }}

    function computeBounds() {{
      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      for (const frame of preview.frames) {{
        for (const point of frame) {{
          const rotated = rotatedPoint(point);
          const px = rotated.x;
          const py = rotated.y + rotated.depth * depthTilt;
          minX = Math.min(minX, px);
          maxX = Math.max(maxX, px);
          minY = Math.min(minY, py);
          maxY = Math.max(maxY, py);
        }}
      }}
      return {{ minX, maxX, minY, maxY }};
    }}

    const bounds = computeBounds();
    const spanX = Math.max(bounds.maxX - bounds.minX, 1e-3);
    const spanY = Math.max(bounds.maxY - bounds.minY, 1e-3);
    const sceneScale = Math.min((canvas.width * 0.64) / spanX, (canvas.height * 0.70) / spanY);
    const sceneCenterX = (bounds.minX + bounds.maxX) * 0.5;
    const sceneCenterY = (bounds.minY + bounds.maxY) * 0.5;

    function project(point) {{
      const rotated = rotatedPoint(point);
      const projectedX = (rotated.x - sceneCenterX) * sceneScale;
      const projectedY = (rotated.y + rotated.depth * depthTilt - sceneCenterY) * sceneScale;
      return {{
        x: canvas.width * 0.5 + projectedX,
        y: canvas.height * 0.60 - projectedY,
      }};
    }}

    function byName(points, name) {{
      const index = boneIndex[name];
      return Number.isInteger(index) ? points[index] : null;
    }}

    function mixHex(baseHex, targetHex, t) {{
      const clampT = Math.max(0, Math.min(1, t));
      const parse = (hex) => [
        Number.parseInt(hex.slice(1, 3), 16),
        Number.parseInt(hex.slice(3, 5), 16),
        Number.parseInt(hex.slice(5, 7), 16),
      ];
      const [ar, ag, ab] = parse(baseHex);
      const [br, bg, bb] = parse(targetHex);
      const r = Math.round(ar + (br - ar) * clampT);
      const g = Math.round(ag + (bg - ag) * clampT);
      const b = Math.round(ab + (bb - ab) * clampT);
      return `rgb(${{r}}, ${{g}}, ${{b}})`;
    }}

    function averagePoints(points) {{
      const valid = points.filter(Boolean);
      if (!valid.length) {{
        return null;
      }}
      const sum = valid.reduce((acc, point) => {{
        acc.x += point.x;
        acc.y += point.y;
        return acc;
      }}, {{ x: 0, y: 0 }});
      return {{ x: sum.x / valid.length, y: sum.y / valid.length }};
    }}

    function quadPoint(a, b, amount) {{
      return {{
        x: a.x + (b.x - a.x) * amount,
        y: a.y + (b.y - a.y) * amount,
      }};
    }}

    function drawPolygon(points, fillStyle, strokeStyle, lineWidth = 1.4, alpha = 1.0) {{
      const valid = points.filter(Boolean);
      if (valid.length < 3) {{
        return;
      }}
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.moveTo(valid[0].x, valid[0].y);
      for (let i = 1; i < valid.length; i += 1) {{
        ctx.lineTo(valid[i].x, valid[i].y);
      }}
      ctx.closePath();
      ctx.fillStyle = fillStyle;
      ctx.fill();
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
      ctx.restore();
    }}

    function drawCapsule(a, b, radius, fillStyle, strokeStyle, alpha = 1.0) {{
      if (!a || !b) {{
        return;
      }}
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const length = Math.hypot(dx, dy);
      if (length < 1e-3) {{
        return;
      }}
      const nx = -dy / length;
      const ny = dx / length;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.moveTo(a.x + nx * radius, a.y + ny * radius);
      ctx.lineTo(b.x + nx * radius, b.y + ny * radius);
      ctx.arc(b.x, b.y, radius, Math.atan2(ny, nx), Math.atan2(-ny, -nx));
      ctx.lineTo(a.x - nx * radius, a.y - ny * radius);
      ctx.arc(a.x, a.y, radius, Math.atan2(-ny, -nx), Math.atan2(ny, nx));
      ctx.closePath();
      ctx.fillStyle = fillStyle;
      ctx.fill();
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = Math.max(1.2, radius * 0.18);
      ctx.stroke();
      ctx.restore();
    }}

    function drawHumanoid(projected, energyColor, pulse) {{
      const pelvis = byName(projected, "pelvis");
      const spine3 = byName(projected, "spine3") || byName(projected, "spine2") || byName(projected, "spine1");
      const neck = byName(projected, "neck") || spine3;
      const head = byName(projected, "head") || neck;
      const leftShoulder = byName(projected, "left_shoulder") || byName(projected, "left_collar");
      const rightShoulder = byName(projected, "right_shoulder") || byName(projected, "right_collar");
      const leftHip = byName(projected, "left_hip") || pelvis;
      const rightHip = byName(projected, "right_hip") || pelvis;
      const leftKnee = byName(projected, "left_knee");
      const rightKnee = byName(projected, "right_knee");
      const leftAnkle = byName(projected, "left_ankle") || byName(projected, "left_foot");
      const rightAnkle = byName(projected, "right_ankle") || byName(projected, "right_foot");
      const leftFoot = byName(projected, "left_foot") || leftAnkle;
      const rightFoot = byName(projected, "right_foot") || rightAnkle;
      const leftElbow = byName(projected, "left_elbow");
      const rightElbow = byName(projected, "right_elbow");
      const leftWrist = byName(projected, "left_wrist");
      const rightWrist = byName(projected, "right_wrist");

      const allY = projected.map((point) => point.y);
      const minY = Math.min(...allY);
      const maxY = Math.max(...allY);
      const bodyHeight = Math.max(60, maxY - minY);
      const torsoColor = mixHex("#1b3d52", energyColor, 0.42);
      const limbColor = mixHex("#113147", energyColor, 0.28);
      const accentColor = mixHex("#f5d9ab", energyColor, 0.24);
      const outlineColor = mixHex("#d8f1ff", energyColor, 0.14);
      const torsoStroke = mixHex(energyColor, "#ffffff", 0.2);
      const torsoWidth = Math.max(10, bodyHeight * 0.072);
      const limbRadius = Math.max(7, bodyHeight * 0.032 + pulse * 2.4);
      const forearmRadius = limbRadius * 0.82;
      const calfRadius = limbRadius * 0.88;
      const neckBase = neck && head ? quadPoint(neck, head, 0.24) : neck;

      if (leftFoot && rightFoot) {{
        const shadowCenter = averagePoints([leftFoot, rightFoot, pelvis]);
        if (shadowCenter) {{
          ctx.save();
          ctx.translate(shadowCenter.x, Math.max(leftFoot.y, rightFoot.y) + bodyHeight * 0.02);
          ctx.scale(1.2, 0.36);
          ctx.fillStyle = "rgba(0, 0, 0, " + (0.16 + pulse * 0.05) + ")";
          ctx.beginPath();
          ctx.arc(0, 0, bodyHeight * 0.18, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }}
      }}

      drawCapsule(leftHip, leftKnee, limbRadius, limbColor, outlineColor, 0.95);
      drawCapsule(rightHip, rightKnee, limbRadius, limbColor, outlineColor, 0.95);
      drawCapsule(leftKnee, leftAnkle, calfRadius, limbColor, outlineColor, 0.95);
      drawCapsule(rightKnee, rightAnkle, calfRadius, limbColor, outlineColor, 0.95);
      drawCapsule(leftShoulder, leftElbow, limbRadius * 0.88, accentColor, outlineColor, 0.92);
      drawCapsule(rightShoulder, rightElbow, limbRadius * 0.88, accentColor, outlineColor, 0.92);
      drawCapsule(leftElbow, leftWrist, forearmRadius, accentColor, outlineColor, 0.92);
      drawCapsule(rightElbow, rightWrist, forearmRadius, accentColor, outlineColor, 0.92);

      if (leftHip && rightHip && leftShoulder && rightShoulder) {{
        const hipMid = averagePoints([leftHip, rightHip]);
        const shoulderMid = averagePoints([leftShoulder, rightShoulder]);
        const upperLeft = quadPoint(leftHip, leftShoulder, 0.16);
        const upperRight = quadPoint(rightHip, rightShoulder, 0.16);
        const lowerLeft = quadPoint(leftHip, leftShoulder, -0.05);
        const lowerRight = quadPoint(rightHip, rightShoulder, -0.05);
        drawPolygon(
          [lowerLeft, lowerRight, upperRight, rightShoulder, shoulderMid, leftShoulder, upperLeft],
          torsoColor,
          torsoStroke,
          2.0,
          0.98,
        );
        if (hipMid && shoulderMid) {{
          drawCapsule(hipMid, shoulderMid, torsoWidth * 0.34, mixHex(torsoColor, "#ffffff", 0.08), torsoStroke, 0.55);
        }}
      }}

      drawCapsule(pelvis, neckBase, torsoWidth * 0.28, mixHex(torsoColor, "#ffffff", 0.12), torsoStroke, 0.6);
      drawCapsule(leftAnkle, leftFoot, calfRadius * 0.68, mixHex(limbColor, "#ffffff", 0.08), outlineColor, 0.96);
      drawCapsule(rightAnkle, rightFoot, calfRadius * 0.68, mixHex(limbColor, "#ffffff", 0.08), outlineColor, 0.96);

      if (head) {{
        const headRadius = Math.max(12, bodyHeight * 0.06 + pulse * 1.4);
        ctx.save();
        ctx.fillStyle = mixHex("#f3ddbf", energyColor, 0.12);
        ctx.beginPath();
        ctx.arc(head.x, head.y - headRadius * 0.12, headRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = mixHex("#ffffff", energyColor, 0.18);
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();
      }}

      for (const [parentIndex, childIndex] of preview.edges) {{
        const a = projected[parentIndex];
        const b = projected[childIndex];
        ctx.strokeStyle = "rgba(255,255,255,0.10)";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }}
    }}

    function markerForFrame(index) {{
      for (const marker of preview.step_markers) {{
        if (index >= marker.start_frame && index < marker.start_frame + marker.frame_count) {{
          return marker;
        }}
      }}
      return preview.step_markers[preview.step_markers.length - 1];
    }}

    function frameSampleForTime(currentTime) {{
      const times = preview.frame_times_sec;
      if (!times.length) {{
        const rawIndex = Math.max(0, Math.min(preview.frames.length - 1, currentTime * preview.fps));
        const leftIndex = Math.floor(rawIndex);
        const rightIndex = Math.min(preview.frames.length - 1, leftIndex + 1);
        return {{ leftIndex, rightIndex, mix: rawIndex - leftIndex }};
      }}
      if (currentTime <= times[0]) {{
        return {{ leftIndex: 0, rightIndex: 0, mix: 0 }};
      }}
      if (currentTime >= times[times.length - 1]) {{
        return {{ leftIndex: times.length - 1, rightIndex: times.length - 1, mix: 0 }};
      }}
      let lo = 0;
      let hi = times.length - 1;
      while (lo < hi) {{
        const mid = Math.floor((lo + hi) / 2);
        if (times[mid] < currentTime) {{
          lo = mid + 1;
        }} else {{
          hi = mid;
        }}
      }}
      const rightIndex = Math.max(1, lo);
      const leftIndex = Math.max(0, rightIndex - 1);
      const leftTime = times[leftIndex];
      const rightTime = times[rightIndex];
      const mix = rightTime <= leftTime ? 0 : (currentTime - leftTime) / (rightTime - leftTime);
      return {{ leftIndex, rightIndex, mix }};
    }}

    function interpolateFrame(leftIndex, rightIndex, mix) {{
      const left = preview.frames[leftIndex];
      const right = preview.frames[rightIndex];
      if (leftIndex === rightIndex) {{
        return left;
      }}
      const blended = [];
      for (let i = 0; i < left.length; i += 1) {{
        blended.push([
          left[i][0] + (right[i][0] - left[i][0]) * mix,
          left[i][1] + (right[i][1] - left[i][1]) * mix,
          left[i][2] + (right[i][2] - left[i][2]) * mix,
        ]);
      }}
      return blended;
    }}

    function pulseForTime(currentTime) {{
      let pulse = 0;
      for (const accent of preview.accents) {{
        const delta = Math.abs(accent.time_sec - currentTime);
        if (delta <= 0.16) {{
          pulse = Math.max(pulse, (1 - delta / 0.16) * (0.35 + accent.strength));
        }}
      }}
      for (const downbeat of preview.downbeats) {{
        const delta = Math.abs(downbeat.time_sec - currentTime);
        if (delta <= 0.18) {{
          pulse = Math.max(pulse, (1 - delta / 0.18) * 0.55);
        }}
      }}
      return Math.min(1.0, pulse);
    }}

    function timelineX(currentTime) {{
      const span = Math.max(0.001, preview.preview_end_time_sec - preview.preview_start_time_sec);
      return ((currentTime - preview.preview_start_time_sec) / span) * timelineCanvas.width;
    }}

    function drawTimeline(currentTime, marker) {{
      timelineCtx.clearRect(0, 0, timelineCanvas.width, timelineCanvas.height);
      timelineCtx.fillStyle = "rgba(255,255,255,0.03)";
      timelineCtx.fillRect(0, 0, timelineCanvas.width, timelineCanvas.height);

      const sectionPalette = {{
        intro: "rgba(118,214,168,0.18)",
        verse: "rgba(94,212,255,0.16)",
        chorus: "rgba(255,140,66,0.18)",
        bridge: "rgba(255,214,102,0.16)",
        outro: "rgba(196,168,255,0.16)",
      }};
      for (const section of preview.sections) {{
        timelineCtx.fillStyle = sectionPalette[section.label] || "rgba(255,255,255,0.08)";
        const startX = timelineX(section.start_time_sec);
        const endX = timelineX(section.end_time_sec);
        timelineCtx.fillRect(startX, 8, Math.max(2, endX - startX), 28);
        timelineCtx.fillStyle = "rgba(255,255,255,0.85)";
        timelineCtx.font = "12px Avenir Next";
        timelineCtx.fillText(section.label, startX + 6, 26);
      }}

      for (const beat of preview.beats) {{
        const x = timelineX(beat.time_sec);
        timelineCtx.strokeStyle = beat.is_downbeat ? "rgba(255,255,255,0.65)" : "rgba(255,255,255,0.18)";
        timelineCtx.lineWidth = beat.is_downbeat ? 2 : 1;
        timelineCtx.beginPath();
        timelineCtx.moveTo(x, 42);
        timelineCtx.lineTo(x, 60);
        timelineCtx.stroke();
      }}

      for (const accent of preview.accents) {{
        const x = timelineX(accent.time_sec);
        const radius = 2 + accent.strength * 7;
        timelineCtx.fillStyle = "rgba(255,184,77,0.82)";
        timelineCtx.beginPath();
        timelineCtx.arc(x, 70, radius, 0, Math.PI * 2);
        timelineCtx.fill();
      }}

      for (const step of preview.step_markers) {{
        if (step.start_time_sec == null || step.end_time_sec == null) {{
          continue;
        }}
        const x = timelineX(step.start_time_sec);
        const w = Math.max(3, timelineX(step.end_time_sec) - x);
        timelineCtx.fillStyle = ENERGY_COLORS[step.target_energy] || "rgba(255,255,255,0.12)";
        timelineCtx.globalAlpha = 0.84;
        timelineCtx.fillRect(x, 84, w, 18);
        timelineCtx.globalAlpha = 1.0;
        timelineCtx.strokeStyle = step.compatible_from_previous ? "rgba(118,214,168,0.9)" : "rgba(255,120,120,0.9)";
        timelineCtx.lineWidth = marker.index === step.index ? 3 : 1.5;
        timelineCtx.strokeRect(x, 84, w, 18);
      }}

      const playheadX = timelineX(currentTime);
      timelineCtx.strokeStyle = "#ffffff";
      timelineCtx.lineWidth = 2;
      timelineCtx.beginPath();
      timelineCtx.moveTo(playheadX, 0);
      timelineCtx.lineTo(playheadX, timelineCanvas.height);
      timelineCtx.stroke();
    }}

    function draw(currentTime) {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const sample = frameSampleForTime(currentTime);
      const frame = interpolateFrame(sample.leftIndex, sample.rightIndex, sample.mix);
      frameIndex = sample.leftIndex;
      const pulse = pulseForTime(currentTime);
      const marker = markerForFrame(frameIndex);
      const energyColor = ENERGY_COLORS[marker.target_energy] || "#5ed4ff";

      ctx.fillStyle = "rgba(255,255,255," + (0.02 + pulse * 0.06) + ")";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, canvas.height * 0.78);
      ctx.lineTo(canvas.width, canvas.height * 0.78);
      ctx.stroke();

      const projected = frame.map(project);
      drawHumanoid(projected, energyColor, pulse);

      scrub.value = String(frameIndex);
      frameLabel.textContent = `frame ${{frameIndex + 1}} / ${{preview.frames.length}}`;
      stepLabel.textContent = `step ${{marker.index}} | ${{marker.section_label || "n/a"}} | ${{marker.target_energy || "n/a"}} | compatibility=${{marker.compatible_from_previous ? "ok" : "fallback"}}`;
      if (preview.frame_times_sec.length === preview.frames.length) {{
        timeLabel.textContent = `song t = ${{currentTime.toFixed(2)}}s | accents=${{marker.accent_count}} | downbeats=${{marker.downbeat_count}}`;
      }} else {{
        timeLabel.textContent = `preview fps = ${{preview.fps}}`;
      }}

      stepCards.forEach((card, cardIndex) => {{
        const cardMarker = preview.step_markers[cardIndex];
        const active = marker.index === cardMarker.index;
        card.classList.toggle("active", active);
        card.style.borderColor = active
          ? (cardMarker.compatible_from_previous ? "rgba(118,214,168,0.68)" : "rgba(255,120,120,0.68)")
          : "rgba(255,255,255,0.06)";
      }});
      drawTimeline(currentTime, marker);
    }}

    function tick(nowMs) {{
      if (lastTickMs == null) {{
        lastTickMs = nowMs;
      }}
      const deltaSec = Math.min(0.05, (nowMs - lastTickMs) / 1000);
      lastTickMs = nowMs;

      if (playing) {{
        if (useAudioSync && !audio.paused && !audio.ended) {{
          playbackTime = Math.min(preview.preview_end_time_sec, Math.max(preview.preview_start_time_sec, audio.currentTime));
        }} else if (preview.frame_times_sec.length) {{
          playbackTime += deltaSec;
          if (playbackTime > preview.preview_end_time_sec) {{
            playbackTime = preview.preview_start_time_sec;
            if (useAudioSync) {{
              audio.currentTime = playbackTime;
            }}
          }}
        }} else {{
          playbackTime += deltaSec;
        }}
      }}
      draw(playbackTime);
      requestAnimationFrame(tick);
    }}

    toggle.addEventListener("click", async () => {{
      playing = !playing;
      toggle.textContent = playing ? "Pause" : "Play";
      if (useAudioSync) {{
        if (playing) {{
          audio.currentTime = playbackTime;
          await audio.play().catch(() => null);
        }} else {{
          audio.pause();
        }}
      }}
    }});

    scrub.addEventListener("input", () => {{
      frameIndex = Number(scrub.value);
      if (useAudioSync && preview.frame_times_sec.length === preview.frames.length) {{
        playbackTime = preview.frame_times_sec[frameIndex];
        audio.currentTime = playbackTime;
      }} else {{
        playbackTime = preview.frame_times_sec.length === preview.frames.length
          ? preview.frame_times_sec[frameIndex]
          : frameIndex / preview.fps;
      }}
      draw(playbackTime);
    }});

    audio.addEventListener("play", () => {{
      playing = true;
      toggle.textContent = "Pause";
    }});

    audio.addEventListener("pause", () => {{
      if (!audio.ended) {{
        playing = false;
        toggle.textContent = "Play";
      }}
    }});

    audio.addEventListener("ended", () => {{
      playing = false;
      toggle.textContent = "Play";
    }});

    draw(playbackTime);
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""
