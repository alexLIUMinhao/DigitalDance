# music-motion-lab

An isolated `Python + Blender` experiment workspace for rebuilding the music-driven 3D dance system outside the Unity project.

## Isolation Rules

- This project lives at `/Users/alex/Desktop/codex project/3d-digital/music-motion-lab`.
- The Unity project remains separate at `/Users/alex/Desktop/codex project/3d-digital/3d-digital-human`.
- Shared roots under `../music` and `../motion-base-assets` are treated as read-only inputs.
- All generated JSON, diagnostics, and preview job files stay inside `music-motion-lab/outputs/`.
- This project does not import Unity runtime code and does not write into `Assets/`, `Packages/`, `ProjectSettings/`, or `StreamingAssets/`.

## Layout

- `src/music_motion_lab/`
  - `pipelines/music_analysis.py`
  - `pipelines/motion_library.py`
  - `pipelines/planner.py`
  - `pipelines/preview.py`
- `config/paths.json`
  - shared read-only input roots
- `contracts/`
  - stable JSON schemas and examples
- `outputs/`
  - generated artifacts only
- `tests/`
  - smoke tests and path-policy tests

## Quick Start

Create or reuse the isolated virtual environment:

```bash
cd "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"
python3 -m venv .venv
```

Run the first music analysis smoke test:

```bash
cd "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"
PYTHONPATH=src python3 -m music_motion_lab.cli analyze-song \
  --input ../music/audio.mp3 \
  --song-id audio
```

Run the same command with manual correction hooks:

```bash
cd "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"
PYTHONPATH=src python3 -m music_motion_lab.cli analyze-song \
  --input ../music/audio.mp3 \
  --song-id audio \
  --overrides config/song_overrides.template.json \
  --output outputs/song_event_maps/audio_song_event_map.override_preview.json
```

Build an initial base-action motion library from the shared preview diagnostics:

```bash
cd "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"
PYTHONPATH=src python3 -m music_motion_lab.cli build-motion-library \
  --unit-beats 4 \
  --stride-beats 1
```

This also emits a stage-review summary:

```bash
cd "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"
cat outputs/motion_libraries/motion_unit_library_showcase.json
```

Build a stage review bundle after planning and preview reporting:

```bash
cd "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"
PYTHONPATH=src python3 -m music_motion_lab.cli build-review-bundle \
  --song-event-map outputs/song_event_maps/audio_song_event_map.json \
  --motion-library-showcase outputs/motion_libraries/motion_unit_library_showcase.json \
  --plan outputs/choreography_plans/audio_choreography_plan.json \
  --retarget-report outputs/retarget_reports/audio_choreography_plan_retarget_report.json
```

Build a browser-based mesh-like preview from the generated preview job:

```bash
cd "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"
PYTHONPATH=src python3 -m music_motion_lab.cli build-web-preview \
  --preview-job outputs/preview_jobs/audio_choreography_plan_preview_job.json \
  --plan outputs/choreography_plans/audio_choreography_plan.json \
  --song-event-map outputs/song_event_maps/audio_song_event_map.json \
  --audio ../music/audio.mp3 \
  --fps 24 \
  --transition-frames 8
```

`build-web-preview` now includes all plan steps by default. Pass `--max-steps N` only when you want a shorter focused review.

Build a source-truth validation page for FineDance raw motion and its matching `music_wav`:

```bash
cd "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"
PYTHONPATH=src python3 -m music_motion_lab.cli build-dataset-truth-preview \
  --dataset finedance \
  --sequence-ids 161 164 168
```

This writes an HTML review page inside `outputs/renders/` that pairs the original dataset `motion/*.npy`, the matching raw `music_wav/*.wav`, and the existing upstream official mesh preview clip. It also writes a sidecar audio-feature report inside `outputs/reports/` for the same sequence set (BPM, energy, beats, strong beats, drum hits). The truth viewer now targets a 24-second clip window when source media length allows.

To place every locally registered FineDance sequence into the same searchable HTML page:

```bash
cd "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"
PYTHONPATH=src python3 -m music_motion_lab.cli build-dataset-truth-preview \
  --dataset finedance \
  --all-sequences
```

Build a true Blender mesh preview manifest and launch it in Blender UI:

```bash
cd "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"
PYTHONPATH=src python3 -m music_motion_lab.cli build-mesh-preview \
  --preview-job outputs/preview_jobs/audio_choreography_plan_preview_job.json \
  --plan outputs/choreography_plans/audio_choreography_plan.json \
  --song-event-map outputs/song_event_maps/audio_song_event_map.json \
  --basis-mode world_hybrid

open -n -a Blender --args --factory-startup \
  --python "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab/tools/blender_build_mesh_preview.py" \
  -- \
  --manifest "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab/outputs/mesh_preview_jobs/audio_choreography_plan_mesh_preview.json" \
  --output-blend "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab/outputs/renders/audio_choreography_plan_mesh_preview.blend"
```

`build-mesh-preview` now prefers the existing full-sequence preview FBXs under `previewCache/dataset_sequences/` when they already contain a skinned mesh. That direct-mesh path is much more stable than re-retargeting the same sequence again at preview time.

Plan choreography from the generated contracts:

```bash
cd "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"
PYTHONPATH=src python3 -m music_motion_lab.cli plan-choreography \
  --song-event-map outputs/song_event_maps/audio_song_event_map.json \
  --motion-library outputs/motion_libraries/motion_unit_library.json
```

Build the preview job and retarget report:

```bash
cd "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"
PYTHONPATH=src python3 -m music_motion_lab.cli build-preview \
  --plan outputs/choreography_plans/audio_choreography_plan.json \
  --motion-library outputs/motion_libraries/motion_unit_library.json
```

If Blender is not on `PATH`, set `config/paths.json -> tooling.blenderPath` or pass it explicitly:

```bash
cd "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"
PYTHONPATH=src python3 -m music_motion_lab.cli build-preview \
  --plan outputs/choreography_plans/audio_choreography_plan.json \
  --motion-library outputs/motion_libraries/motion_unit_library.json \
  --blender-path /Applications/Blender.app/Contents/MacOS/Blender
```

Run tests:

```bash
cd "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Current Scope

This initial version focuses on:

1. layered `song_event_map.json` generation
2. overlapping-window `motion_unit_library.json` generation from shared dataset diagnostics
3. rule-based `choreography_plan.json` generation
4. independent `preview_job.json` and `retarget_report.json` output
5. mesh-like browser review for full-plan playback without Unity
6. Blender `.blend` mesh preview assembly from existing preview-cache FBX sequences

## Manual Correction Hooks

`analyze-song` now accepts `--overrides` with optional manual truth for:

- `bpm`
- `beats_per_bar`
- `beats`
- `downbeats`
- `accents`
- `phrases`
- `sections`
- `manual_review_required`

Use [song_overrides.template.json](/Users/alex/Desktop/codex project/3d-digital/music-motion-lab/config/song_overrides.template.json) as the starting shape for slow songs, ancient-style songs, and any track where automatic beat or phrase tracking needs hand correction.

If Blender is not installed, preview execution is reported as `backend_unavailable` while the rest of the pipeline remains usable.
