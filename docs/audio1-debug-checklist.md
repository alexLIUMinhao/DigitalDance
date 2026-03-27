# audio1 Debug Checklist

This checklist is for two failures:

- `audio1` does not start dancing after `Start`
- `audio1` dances at the wrong speed or does not fit the music

## Fast Classification

Use this order every time:

1. Verify startup state
2. Verify song switch state
3. Verify `Start` bootstraps the first clip
4. Verify clip time grows
5. Verify `Dance BPM` matches the manual slow-song beat grid

## Runtime UI Checks

Left debug panel must expose at least:

- `Playback`
- `Song`
- `Tempo: Raw / Local / Dance`
- `Mode: grouping / energy`
- `Speed`
- `Current Clip`
- `Clip Time`

Expected `audio1` values:

- `Playback` is `idle` after startup
- `Song` becomes `audio1`
- `Dance BPM` is close to the intended slow-song pulse
- `Energy Mode` is `gentle`
- `Current Clip` changes from `n/a` to a concrete clip after `Start`
- `Clip Time` keeps increasing while playing

## Expected State Machine

### Startup

Expected:

- `willa` is in default pose
- music is not playing
- no dancing has been scheduled yet
- status reads `Ready. Press Start.`

Log must not contain `Configured dance slot` before any `Start`.

### Song Change

Expected:

- current music stops immediately
- avatar returns to default pose
- status reads `Song changed. Press Start.`
- no dancing starts automatically

### Start

Expected log sequence:

1. `UI start requested. Current playback state: Idle`
2. within about 0.5s:
   - `Queued clip ...`
   - `Scheduled initial variant ...`
   - `Configured dance slot ...`

If step 1 appears without step 2, the failure is in startup scheduling, not in tempo adaptation.

### Pause

Expected:

- music time stops
- `Clip Time` stops
- pose freezes
- pressing `Start` again resumes from the same point

## Input Validation

Run:

```bash
python3 /Users/alex/Desktop/codex\ project/3d-digital/3d-digital-human/tools/dance_pipeline/validate_runtime_inputs.py --song-id audio1_mp3
```

What to check:

- `audio1_mp3` exists in `song_catalog.json`
- `playbackProfile` uses:
  - `beatGrouping = 1`
  - `energyMode = gentle`
- `Raw BPM` and `Dance BPM` are both close to the manual slow-song beat grid
- beat list is strictly increasing
- beat windows all have positive duration
- motion manifest has at least 3 `low_energy` clips

If validation passes but the song still feels wrong, the next suspect is beat analysis quality, not schema wiring.

## Log Collection

Run:

```bash
tail -n 150 ~/Library/Logs/DefaultCompany/3d-digital-human/Player.log
```

Key lines to look for:

- `Bootstrap received song selection: audio1_mp3`
- `Dance demo selected audio 'audio1'`
- `UI start requested. Current playback state: Idle`
- `Start request queued while reloading song data.`
- `Playback started from beginning for song audio1_mp3.`
- `Queued clip ...`
- `Scheduled initial variant ...`
- `Configured dance slot ...`

## Symptom -> Suspect Map

### `Start` pressed, nothing moves

Check:

- does the log stop at `UI start requested`?

If yes:

- failure is in start/reload sequencing
- check whether the song was still loading when `Start` was pressed

### `Current Clip` changes, but `Clip Time` stays at `0`

Likely:

- PlayableGraph is not advancing
- controller is still paused
- transition was queued but never activated

### `Clip Time` grows, but avatar does not move

Likely:

- clip is not retargeting correctly
- imported clip is not valid Humanoid motion
- animator graph is active but output pose is invalid

### avatar moves, but does not fit `audio1`

Likely:

- manual beat override is missing or stale
- `Dance BPM` is still too high for the intended slow-song pulse
- `gentle` mode weights are not strict enough

At that point the next fix is to refresh the manual beat/segment override for `audio1`, not more random retiming guesses.

## Automated Coverage In Repo

Current edit-time coverage added in:

- `/Users/alex/Desktop/codex project/3d-digital/3d-digital-human/Assets/Editor/DancePlaybackProfileTests.cs`

Covers:

- playback profile defaulting
- catalog entry fallback defaults
- gentle-mode choreography preferring low-energy loop candidates

## Batch Validation Commands

If the main Unity project is already open in the editor, run compile/build against a temporary copy.

Compile:

```bash
/Applications/Unity/Unity.app/Contents/MacOS/Unity -batchmode -quit -projectPath /tmp/3d-digital-human-debug.QfgLS2 -logFile /tmp/dance_unity_compile_copy4.log
```

Build:

```bash
/Applications/Unity/Unity.app/Contents/MacOS/Unity -batchmode -quit -projectPath /tmp/3d-digital-human-debug.QfgLS2 -executeMethod DanceDemo.Editor.DanceBuildUtility.PerformBuildFromCommandLine -logFile /tmp/dance_unity_build_copy4.log
```

## User Repro Path

Use this exact order when reporting a bug:

1. Launch the app and do not click anything
2. Screenshot the idle state
3. Select `audio1`
4. Wait for `Song changed. Press Start.`
5. Screenshot the UI
6. Press `Start`
7. Wait 2 seconds
8. Screenshot the UI again
9. Attach the last 150 log lines

The three screenshots plus the log tail are enough to separate:

- state machine failure
- start scheduling failure
- playable update failure
- tempo input failure
