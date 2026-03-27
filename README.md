# DigitalDance

Unity 2022.3 LTS VRM dance demo. The project drives `willa.vrm` with a data-driven music-to-dance pipeline:

- offline Python generates song and motion metadata
- Unity loads JSON from `StreamingAssets`
- runtime choreography selects motion variants from a local Mixamo-style FBX library
- Playables drive the avatar with beat-aware speed retiming and clip switching

## 1. What Is In This Repo

Current committed content:

- Unity project version: `2022.3.62f3`
- VRM avatar: `willa.vrm`
- songs: `audio`, `audio1`, `audio2`, `audio3`, `audio4`
- motion library: `40` tagged FBX dance clips
- runtime data:
  - `song_catalog.json`
  - per-song analysis JSON under `Assets/StreamingAssets/DanceData/Songs`
  - `motion_manifest.json`

The project is desktop-first. macOS standalone is the primary validated target.

## 2. Repo Layout

- `Assets/MainScene.unity`
  - main runtime scene
- `Assets/Scripts/Dance`
  - runtime systems:
    - `DanceDemoBootstrap`
    - `MusicConductor`
    - `ChoreographyEngine`
    - `CharacterDanceController`
    - `DanceDebugUI`
- `Assets/Resources/music`
  - selectable audio files
- `Assets/Resources/Dance/Clips`
  - local FBX dance clips
- `Assets/StreamingAssets/DanceData`
  - runtime JSON data
- `Assets/StreamingAssets/Avatars`
  - runtime VRM source
- `tools/dance_pipeline`
  - Python preprocessing and validation
- `docs`
  - build notes, architecture, testing, and debug references

## 3. Quick Start On Another Device

### Requirements

- macOS
- Unity `2022.3.62f3`
- Python 3.10+ only if you want to regenerate JSON

### Open And Run

1. Clone the repository.
2. Open the folder in Unity Hub as a Unity `2022.3.62f3` project.
3. Wait for package restore and asset import to finish.
4. Open `Assets/MainScene.unity`.
5. Press Play.

Expected startup behavior:

- `willa` is in the default pose
- music is not playing
- no dance starts automatically
- choose a song in the right-bottom card
- click `Start` to begin
- click `Pause` to freeze both music and dance

### Build Desktop App

In Unity:

- `Tools > Dance Demo > Build macOS Standalone`

Or batch mode:

```bash
/Applications/Unity/Unity.app/Contents/MacOS/Unity \
  -batchmode \
  -quit \
  -projectPath "/absolute/path/to/3d-digital-human" \
  -executeMethod DanceDemo.Editor.DanceBuildUtility.PerformBuildFromCommandLine
```

## 4. Runtime Usage Flow

### Playback Flow

1. Launch app.
2. Confirm avatar is idle.
3. Select a song from the compact card.
4. Click `Start`.
5. Click `Pause` to freeze.
6. Click `Start` again to resume.
7. Change song to reset pose and return to idle.

### UI Layout

- right-bottom compact card:
  - song selector
  - `Start`
  - `Pause`
  - compact choreography summary
- left debug panel:
  - press `Tab` to show/hide
  - shows BPM, clip ids, clip time, strong beat reason, and status

## 5. How The System Works

Short version:

1. Python analyzes songs and builds JSON.
2. Unity loads song data and motion data.
3. `MusicConductor` turns the song into a live rhythm snapshot.
4. `ChoreographyEngine` scores motion variants.
5. `CharacterDanceController` queues and blends animation clips.
6. `DanceDebugUI` exposes playback and debug state.

Detailed architecture and flowcharts:

- [docs/system-architecture.md](docs/system-architecture.md)

## 6. Song Data Pipeline

### Analyze A Song

Create a local virtual environment once:

```bash
python3 -m venv .venv
.venv/bin/pip install -r tools/dance_pipeline/requirements.txt
```

Example:

```bash
.venv/bin/python tools/dance_pipeline/analyze_song.py \
  --input Assets/Resources/music/audio1.mp3 \
  --song-id audio1_mp3 \
  --audio-resource-path music/audio1 \
  --overrides tools/dance_pipeline/config/song_overrides.audio1.json \
  --output Assets/StreamingAssets/DanceData/Songs/audio1_mp3.json
```

What the script outputs:

- `bpm`
- `beats`
- `beatWindows`
- `downbeats`
- `grooveEnvelope`
- `tempoMap`
- `segments`

### Manual Beat Overrides

For songs where automatic beat tracking is musically wrong, use overrides JSON.

Supported override fields include:

- `beats`
- `downbeats`
- `beats_per_bar`
- `segments`
- `bpm`
- `display_name`
- `audio_resource_path`

`audio1` currently uses a manual half-time beat grid so runtime dance BPM follows the intended slow pulse.

## 7. Motion Library Pipeline

Generate the runtime motion manifest:

```bash
.venv/bin/python tools/dance_pipeline/build_motion_manifest.py \
  --clips-root Assets/Resources/Dance/Clips \
  --tags tools/dance_pipeline/config/motion_tags.json \
  --output Assets/StreamingAssets/DanceData/motion_manifest.json
```

The manifest stores per-clip authoring data such as:

- `energyBand`
- `preferredSegments`
- `nativeBpm`
- `speedMin`
- `speedMax`
- `retimeProfile`
- `varietyGroup`
- `role`
- `entryOffsetsBeats`
- `sliceBeatsOptions`

This is the main quality-control surface for choreography stability.

## 8. Updating Content

### Add A New Song

1. Put the file under `Assets/Resources/music`.
2. Add a catalog entry in `Assets/StreamingAssets/DanceData/song_catalog.json`.
3. Create or update an overrides file if needed.
4. Run `analyze_song.py`.
5. Reopen Unity or click `Reload JSON`.

### Add New Dance Clips

1. Put FBX files under `Assets/Resources/Dance/Clips`.
2. Ensure Unity import type is Humanoid if the clip should be playable.
3. Tag the clip in `tools/dance_pipeline/config/motion_tags.json`.
4. Rebuild `motion_manifest.json`.
5. Reopen Unity or click `Reload JSON`.

## 9. Validation And Testing

Validate song inputs:

```bash
python3 tools/dance_pipeline/validate_runtime_inputs.py --song-id audio_mp3
python3 tools/dance_pipeline/validate_runtime_inputs.py --song-id audio1_mp3
```

Test references:

- [docs/testing/manual-experience-checklist.md](docs/testing/manual-experience-checklist.md)
- [docs/testing/functional-regression-matrix.md](docs/testing/functional-regression-matrix.md)
- [docs/testing/automation-test-checklist.md](docs/testing/automation-test-checklist.md)
- [docs/testing/defect-triage-and-log-collection.md](docs/testing/defect-triage-and-log-collection.md)
- [docs/audio1-debug-checklist.md](docs/audio1-debug-checklist.md)

## 10. Git / Deployment Notes

This repository is structured so another device can open and run the project without committing Unity cache folders.

Do commit:

- `Assets`
- `Packages`
- `ProjectSettings`
- `tools`
- `docs`
- `README.md`

Do not commit:

- `Library`
- `Logs`
- `Builds`
- `.venv`
- `UserSettings`

Those exclusions are already encoded in `.gitignore`.

## 11. Limitations

- No Mixamo scraping or auto-download is implemented.
- WebGL is not treated as a parity target.
- Music feel still depends on good beat input. Slow songs may require manual beat overrides.
- The next likely quality improvement after beat correctness is transition compatibility scoring.
