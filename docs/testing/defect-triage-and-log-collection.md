# Defect Triage And Log Collection

Use this document to turn a vague demo complaint into a classified defect with reproducible evidence.

## Evidence Package

Every defect report should include:

- app version or build timestamp
- selected song
- playback state when failure occurred
- one screenshot with left debug panel visible
- one screenshot with right compact card visible
- last 150 lines of `Player.log`

Log collection:

```bash
tail -n 150 ~/Library/Logs/DefaultCompany/3d-digital-human/Player.log
```

## Required UI Fields In Evidence

Capture these fields whenever possible:

- `Playback`
- `Song`
- `Tempo: Raw / Local / Dance`
- `Rhythm: Group xN | Mode`
- `Speed`
- `Current Clip`
- `Clip Time`
- `Strong Beat / Reason`
- `Status`

## Defect Categories

### 1. Startup / State Machine

Symptoms:

- app starts dancing automatically
- `Start` does nothing
- `Pause` affects only audio or only animation
- switching songs auto-starts playback

Primary suspects:

- `DanceDemoBootstrap`
- reload / start ordering
- busy-state gating

Key logs:

- `UI start requested`
- `Start request queued while reloading song data`
- `Playback started from beginning`

### 2. Scheduling / Playables

Symptoms:

- `Current Clip` changes but avatar does not move
- `Clip Time` does not increase
- avatar freezes mid-song

Primary suspects:

- `CharacterDanceController`
- queued transition activation
- playable pause/resume state

Key logs:

- `Queued clip`
- `Scheduled initial variant`
- `Configured dance slot`

### 3. Rhythm / Tempo Interpretation

Symptoms:

- avatar moves but does not match song feel
- `audio1` looks too fast or too energetic
- strong beats do not feel emphasized

Primary suspects:

- song analysis data
- `SongPlaybackProfile`
- `MusicConductor`
- `ChoreographyEngine`

Key checks:

- `Dance BPM` follows the manual slow-song beat grid for `audio1`
- `Energy Mode = gentle` for `audio1`
- high-energy clip frequency during slow-song playback

### 4. Avatar / Visual Stability

Symptoms:

- sinking through floor
- missing body
- bad camera framing
- abrupt pose snaps at transitions

Primary suspects:

- avatar load path
- ground correction
- camera fallback framing
- clip compatibility

## Symptom To Root-Cause Map

### `Start` pressed, nothing moves

If logs stop at `UI start requested`:

- classify as startup sequencing bug

If logs include `Configured dance slot` but `Clip Time` stays at `0`:

- classify as playable activation or pause-state bug

### `Clip Time` grows, avatar still not moving

- classify as Humanoid retargeting / animation application bug

### Avatar moves, but slow song feels wrong

- classify as rhythm interpretation bug
- check analysis BPM, `Dance BPM`, energy mode, and selected clip energy bands

## Severity Rules

- `P0`: cannot start playback at all
- `P1`: song plays but avatar cannot dance, freezes, or disappears
- `P2`: avatar dances but state machine, switching, or rhythm quality is clearly broken
- `P3`: cosmetic UI or non-blocking debug display issues

## Related References

- `/Users/alex/Desktop/codex project/3d-digital/3d-digital-human/docs/audio1-debug-checklist.md`
- `/Users/alex/Desktop/codex project/3d-digital/3d-digital-human/tools/dance_pipeline/validate_runtime_inputs.py`
