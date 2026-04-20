from __future__ import annotations

import html
import json
from typing import Any


def _json_script(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def _step_summary(step: dict[str, Any]) -> dict[str, Any]:
    root_alignment = dict(step.get("root_alignment", {}))
    return {
        "index": step.get("index"),
        "unit_id": step.get("unit_id"),
        "source_sequence": step.get("source_sequence"),
        "section_label": step.get("section_label"),
        "start_time_sec": step.get("start_time_sec"),
        "end_time_sec": step.get("end_time_sec"),
        "scene_frame_start": step.get("scene_frame_start"),
        "scene_frame_end": step.get("scene_frame_end"),
        "blend_in_frames": step.get("blend_in_frames"),
        "blend_out_frames": step.get("blend_out_frames"),
        "speed_scale": step.get("speed_scale"),
        "transition_score": step.get("transition_score"),
        "rhythm_locks": list(step.get("rhythm_locks", [])),
        "entry_anchor": dict(root_alignment.get("entry_anchor", {})),
        "exit_anchor": dict(root_alignment.get("exit_anchor", {})),
    }


def build_smplx_stitch_web_preview_document(
    manifest: dict[str, Any],
    song_event_map: dict[str, Any] | None = None,
    audio_href: str | None = None,
) -> str:
    steps = [_step_summary(dict(step)) for step in manifest.get("steps", [])]
    preview_payload = {
        "manifest": {
            "manifest_id": manifest.get("manifest_id"),
            "plan_id": manifest.get("plan_id"),
            "song_id": manifest.get("song_id"),
            "library_id": manifest.get("library_id"),
            "fps": manifest.get("fps", 30),
            "mesh_backend": manifest.get("mesh_backend"),
            "scene": dict(manifest.get("scene", {})),
            "blend_policy": dict(manifest.get("blend_policy", {})),
        },
        "steps": steps,
        "transitions": list(manifest.get("transitions", [])),
        "cache_requests": list(manifest.get("cache_requests", [])),
        "song": {
            "beats": list((song_event_map or {}).get("beats", [])),
            "downbeats": list((song_event_map or {}).get("downbeats", [])),
            "accents": list((song_event_map or {}).get("accents", [])),
            "sections": list((song_event_map or {}).get("sections", [])),
            "duration_sec": (song_event_map or {}).get("duration_sec"),
        },
        "audio_href": audio_href,
    }
    title = html.escape(str(manifest.get("manifest_id") or "SMPL-X stitch preview"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101214;
      --panel: #171b1f;
      --panel-2: #20262b;
      --text: #f2f0e8;
      --muted: #aab4b6;
      --line: rgba(255,255,255,0.13);
      --blue: #6bb8ff;
      --green: #7bd88f;
      --gold: #f4c95d;
      --coral: #ff7b63;
      --lav: #bba7ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    .shell {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: 100vh;
    }}
    main {{
      display: grid;
      grid-template-rows: minmax(360px, 1fr) auto;
      min-width: 0;
    }}
    .stage {{
      position: relative;
      min-height: 420px;
      border-right: 1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(107,184,255,0.05), transparent 34%),
        radial-gradient(circle at 50% 110%, rgba(123,216,143,0.12), transparent 34%),
        #111417;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 100%;
    }}
    .hud {{
      position: absolute;
      left: 18px;
      top: 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    button {{
      border: 1px solid rgba(255,255,255,0.18);
      background: #f2f0e8;
      color: #141414;
      height: 34px;
      padding: 0 14px;
      border-radius: 6px;
      font-weight: 700;
      cursor: pointer;
    }}
    .pill {{
      min-height: 28px;
      display: inline-flex;
      align-items: center;
      padding: 4px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(0,0,0,0.28);
      color: var(--muted);
      white-space: nowrap;
    }}
    .timeline {{
      border-top: 1px solid var(--line);
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 14px 18px 18px;
    }}
    .timeline-canvas {{
      height: 132px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #111417;
    }}
    aside {{
      background: var(--panel);
      min-width: 0;
      overflow: auto;
      max-height: 100vh;
    }}
    .meta {{
      padding: 18px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      font-size: 20px;
      line-height: 1.15;
      margin: 0 0 10px;
      font-weight: 800;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }}
    .metric {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      min-width: 0;
    }}
    .metric b {{
      display: block;
      font-size: 16px;
      color: var(--text);
      overflow-wrap: anywhere;
    }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    .step-list {{
      display: grid;
      gap: 8px;
      padding: 14px;
    }}
    .step {{
      border: 1px solid var(--line);
      border-left: 4px solid var(--blue);
      background: #15191d;
      border-radius: 6px;
      padding: 10px;
      min-width: 0;
    }}
    .step.active {{
      border-left-color: var(--gold);
      background: #202119;
    }}
    .step h2 {{
      margin: 0 0 6px;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .step p {{
      margin: 3px 0;
      color: var(--muted);
      font-size: 12px;
    }}
    audio {{
      width: min(340px, calc(100vw - 36px));
      height: 34px;
    }}
    @media (max-width: 920px) {{
      .shell {{ grid-template-columns: 1fr; }}
      aside {{ max-height: none; }}
      .stage {{ border-right: 0; min-height: 420px; }}
      .timeline {{ border-right: 0; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <main>
      <section class="stage">
        <canvas id="stageCanvas"></canvas>
        <div class="hud">
          <button id="playButton" type="button">Play</button>
          <span class="pill" id="timeLabel">0.00s</span>
          <span class="pill" id="stepLabel">step</span>
          <span class="pill" id="lockLabel">locks</span>
          {f'<audio id="audio" controls src="{html.escape(audio_href)}"></audio>' if audio_href else '<span class="pill">no audio linked</span>'}
        </div>
      </section>
      <section class="timeline">
        <canvas class="timeline-canvas" id="timelineCanvas"></canvas>
      </section>
    </main>
    <aside>
      <section class="meta">
        <h1>{title}</h1>
        <div class="grid">
          <div class="metric"><b id="stepCount">0</b><span>steps</span></div>
          <div class="metric"><b id="transitionCount">0</b><span>transitions</span></div>
          <div class="metric"><b id="durationLabel">0s</b><span>duration</span></div>
          <div class="metric"><b id="fpsLabel">30</b><span>fps</span></div>
        </div>
      </section>
      <section class="step-list" id="stepList"></section>
    </aside>
  </div>
  <script id="previewData" type="application/json">{_json_script(preview_payload)}</script>
  <script>
    const data = JSON.parse(document.getElementById("previewData").textContent);
    const stageCanvas = document.getElementById("stageCanvas");
    const stageCtx = stageCanvas.getContext("2d");
    const timelineCanvas = document.getElementById("timelineCanvas");
    const timelineCtx = timelineCanvas.getContext("2d");
    const playButton = document.getElementById("playButton");
    const audio = document.getElementById("audio");
    const stepLabel = document.getElementById("stepLabel");
    const timeLabel = document.getElementById("timeLabel");
    const lockLabel = document.getElementById("lockLabel");
    const stepList = document.getElementById("stepList");
    let playing = false;
    let localStart = 0;
    let localTime = 0;
    const steps = data.steps || [];
    const transitions = data.transitions || [];
    const duration = Math.max(1, ...steps.map((s) => Number(s.end_time_sec || 0)));
    document.getElementById("stepCount").textContent = steps.length;
    document.getElementById("transitionCount").textContent = transitions.length;
    document.getElementById("durationLabel").textContent = `${{duration.toFixed(2)}}s`;
    document.getElementById("fpsLabel").textContent = data.manifest.fps || 30;

    const resizeCanvas = (canvas) => {{
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.floor(rect.width * dpr));
      const height = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== width || canvas.height !== height) {{
        canvas.width = width;
        canvas.height = height;
      }}
    }};

    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const lerp = (a, b, t) => a + (b - a) * t;
    const activeStepAt = (time) => steps.find((s) => time >= Number(s.start_time_sec || 0) && time <= Number(s.end_time_sec || 0)) || steps[steps.length - 1] || null;
    const stepProgress = (step, time) => {{
      if (!step) return 0;
      const start = Number(step.start_time_sec || 0);
      const end = Number(step.end_time_sec || start + 1);
      return clamp((time - start) / Math.max(0.001, end - start), 0, 1);
    }};
    const anchorVec = (anchor) => Array.isArray(anchor?.root_translation) ? anchor.root_translation.map(Number) : [0, 0, 0];
    const yaw = (anchor) => Number(anchor?.root_yaw_deg || 0) * Math.PI / 180;
    const poseFor = (step, time) => {{
      if (!step) return {{x: 0, z: 0, yaw: 0, progress: 0}};
      const p = stepProgress(step, time);
      const a = anchorVec(step.entry_anchor);
      const b = anchorVec(step.exit_anchor);
      const ya = yaw(step.entry_anchor);
      const yb = yaw(step.exit_anchor);
      return {{
        x: lerp(a[0], b[0], p),
        z: lerp(a[2], b[2], p),
        yaw: lerp(ya, yb, p),
        progress: p,
      }};
    }};

    const allPoints = steps.flatMap((step) => [anchorVec(step.entry_anchor), anchorVec(step.exit_anchor)]);
    const minX = Math.min(...allPoints.map((p) => p[0]), -1);
    const maxX = Math.max(...allPoints.map((p) => p[0]), 1);
    const minZ = Math.min(...allPoints.map((p) => p[2]), -1);
    const maxZ = Math.max(...allPoints.map((p) => p[2]), 1);
    const worldToCanvas = (x, z, w, h) => {{
      const pad = Math.min(w, h) * 0.15;
      const sx = (w - pad * 2) / Math.max(0.4, maxX - minX);
      const sz = (h - pad * 2) / Math.max(0.4, maxZ - minZ);
      const s = Math.min(sx, sz);
      return {{
        x: w / 2 + (x - (minX + maxX) / 2) * s,
        y: h / 2 + (z - (minZ + maxZ) / 2) * s,
        scale: s,
      }};
    }};

    const drawStage = (time) => {{
      resizeCanvas(stageCanvas);
      const ctx = stageCtx;
      const w = stageCanvas.width;
      const h = stageCanvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#101417";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 1;
      for (let x = 0; x <= w; x += 48) {{ ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }}
      for (let y = 0; y <= h; y += 48) {{ ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }}
      for (const step of steps) {{
        const a = anchorVec(step.entry_anchor);
        const b = anchorVec(step.exit_anchor);
        const pa = worldToCanvas(a[0], a[2], w, h);
        const pb = worldToCanvas(b[0], b[2], w, h);
        ctx.strokeStyle = step === activeStepAt(time) ? "#f4c95d" : "rgba(107,184,255,0.42)";
        ctx.lineWidth = step === activeStepAt(time) ? 5 : 2;
        ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
      }}
      const active = activeStepAt(time);
      const pose = poseFor(active, time);
      const p = worldToCanvas(pose.x, pose.z, w, h);
      const body = clamp(p.scale * 0.18, 34, 86);
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(pose.yaw);
      ctx.fillStyle = "#f2f0e8";
      ctx.strokeStyle = "#101214";
      ctx.lineWidth = 4;
      ctx.beginPath(); ctx.arc(0, -body * 0.58, body * 0.16, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = "#f2f0e8"; ctx.lineWidth = Math.max(5, body * 0.08); ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(0, -body * 0.42); ctx.lineTo(0, body * 0.25); ctx.stroke();
      ctx.strokeStyle = "#7bd88f";
      ctx.beginPath(); ctx.moveTo(-body * 0.38, -body * 0.16); ctx.lineTo(body * 0.38, -body * 0.16); ctx.stroke();
      ctx.strokeStyle = "#6bb8ff";
      ctx.beginPath(); ctx.moveTo(-body * 0.22, body * 0.25); ctx.lineTo(-body * 0.46, body * 0.68); ctx.moveTo(body * 0.22, body * 0.25); ctx.lineTo(body * 0.46, body * 0.68); ctx.stroke();
      ctx.fillStyle = "#ff7b63";
      ctx.beginPath(); ctx.moveTo(0, -body * 0.78); ctx.lineTo(-body * 0.11, -body * 0.5); ctx.lineTo(body * 0.11, -body * 0.5); ctx.closePath(); ctx.fill();
      ctx.restore();
      const locks = active?.rhythm_locks || [];
      for (const lock of locks) {{
        const dt = Math.abs(Number(lock.time_sec || 0) - time);
        if (dt > 0.35) continue;
        const r = 20 + (1 - dt / 0.35) * 70;
        ctx.strokeStyle = `rgba(244,201,93,${{(1 - dt / 0.35).toFixed(3)}})`;
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.stroke();
      }}
      stepLabel.textContent = active ? `step ${{active.index}} | seq ${{active.source_sequence}}` : "step";
      timeLabel.textContent = `${{time.toFixed(2)}}s`;
      lockLabel.textContent = active ? `locks ${{(active.rhythm_locks || []).length}}` : "locks";
    }};

    const drawTimeline = (time) => {{
      resizeCanvas(timelineCanvas);
      const ctx = timelineCtx;
      const w = timelineCanvas.width;
      const h = timelineCanvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#111417"; ctx.fillRect(0, 0, w, h);
      const xFor = (t) => clamp(Number(t || 0) / duration, 0, 1) * w;
      for (const step of steps) {{
        const x = xFor(step.start_time_sec);
        const x2 = xFor(step.end_time_sec);
        ctx.fillStyle = step === activeStepAt(time) ? "#f4c95d" : "#6bb8ff";
        ctx.globalAlpha = step === activeStepAt(time) ? 0.88 : 0.48;
        ctx.fillRect(x, 34, Math.max(2, x2 - x), 38);
        ctx.globalAlpha = 1;
        ctx.fillStyle = "#f2f0e8";
        ctx.font = `${{Math.max(11, Math.round(12 * (window.devicePixelRatio || 1)))}}px sans-serif`;
        ctx.fillText(String(step.index), x + 5, 58);
        ctx.fillStyle = "#f4c95d";
        for (const lock of step.rhythm_locks || []) {{
          const lx = xFor(lock.time_sec);
          ctx.beginPath(); ctx.arc(lx, 92, 5, 0, Math.PI * 2); ctx.fill();
        }}
      }}
      ctx.strokeStyle = "#ff7b63"; ctx.lineWidth = 3;
      for (const transition of transitions) {{
        const step = steps.find((item) => Number(item.index) === Number(transition.incoming_step));
        if (!step) continue;
        const tx = xFor(step.start_time_sec);
        ctx.beginPath(); ctx.moveTo(tx, 18); ctx.lineTo(tx, h - 18); ctx.stroke();
      }}
      const playhead = xFor(time);
      ctx.strokeStyle = "#f2f0e8"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(playhead, 0); ctx.lineTo(playhead, h); ctx.stroke();
    }};

    const renderStepList = () => {{
      stepList.innerHTML = "";
      for (const step of steps) {{
        const el = document.createElement("article");
        el.className = "step";
        el.dataset.index = step.index;
        el.innerHTML = `
          <h2>${{step.index}} · ${{step.source_sequence}} · ${{step.section_label || "section"}}</h2>
          <p>${{Number(step.start_time_sec || 0).toFixed(2)}}s to ${{Number(step.end_time_sec || 0).toFixed(2)}}s · frames ${{step.scene_frame_start}}-${{step.scene_frame_end}}</p>
          <p>unit ${{step.unit_id}}</p>
          <p>blend in/out ${{step.blend_in_frames}}/${{step.blend_out_frames}} · transition ${{Number(step.transition_score || 0).toFixed(2)}}</p>
        `;
        stepList.appendChild(el);
      }}
    }};

    const tick = (now) => {{
      if (audio && !audio.paused) {{
        localTime = audio.currentTime;
      }} else if (playing) {{
        localTime = ((now - localStart) / 1000) % duration;
      }}
      drawStage(localTime);
      drawTimeline(localTime);
      for (const el of stepList.querySelectorAll(".step")) {{
        const active = activeStepAt(localTime);
        el.classList.toggle("active", active && String(active.index) === el.dataset.index);
      }}
      requestAnimationFrame(tick);
    }};

    playButton.addEventListener("click", () => {{
      if (audio) {{
        if (audio.paused) {{ audio.play(); playButton.textContent = "Pause"; }}
        else {{ audio.pause(); playButton.textContent = "Play"; }}
        return;
      }}
      playing = !playing;
      if (playing) localStart = performance.now() - localTime * 1000;
      playButton.textContent = playing ? "Pause" : "Play";
    }});
    if (audio) {{
      audio.addEventListener("play", () => {{ playButton.textContent = "Pause"; }});
      audio.addEventListener("pause", () => {{ playButton.textContent = "Play"; }});
    }}
    window.addEventListener("resize", () => {{ drawStage(localTime); drawTimeline(localTime); }});
    renderStepList();
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""
