# Manual Experience Checklist

This checklist is the product-facing smoke pass for the desktop demo.

## Preconditions

- Build target is macOS desktop.
- App under test is:
  - `/Users/alex/Desktop/codex project/3d-digital/3d-digital-human/Builds/macOS/DanceDemo.app`
- Available songs:
  - `audio`
  - `audio1`

## 1. Launch Experience

Expected on launch:

- `willa` is visible in a stable full-body framing.
- Avatar is in the default pose.
- No music is playing.
- No dance motion is playing.
- Right-bottom compact card is visible and does not block the avatar.
- Left debug panel can be shown or hidden with `Tab`.
- Status reads `Ready. Press Start.`

Fail if any of these happen:

- Music starts automatically.
- Avatar starts dancing automatically.
- Avatar is partially out of frame, underground, frozen in a dance pose, or missing.

## 2. Playback Experience

Steps:

1. With the default selected song, press `Start`.
2. Watch the first 10 seconds.

Expected:

- Music starts once.
- Avatar begins dancing within about 0.5 seconds.
- `Playback` becomes `playing`.
- `Current Clip` becomes a concrete clip id.
- `Clip Time` begins increasing.
- Camera remains stable during playback.

Fail if:

- `Start` does nothing.
- Music starts but avatar does not move.
- Avatar moves before music starts.
- Obvious hard cuts, severe foot sinking, or full-body freezing occur immediately.

## 3. Pause / Resume Experience

Steps:

1. While playing, press `Pause`.
2. Wait 2 seconds.
3. Press `Start`.

Expected:

- `Pause` freezes both music and pose.
- `Clip Time` stops increasing while paused.
- Resume continues from the paused point, not from the beginning.

Fail if:

- Audio pauses but pose keeps moving.
- Pose freezes but audio continues.
- Resume restarts from the beginning instead of continuing.

## 4. Song Switching Experience

Steps:

1. While playing `audio`, switch to `audio1`.
2. Observe the avatar and status before pressing `Start`.
3. Press `Start`.

Expected after switch, before `Start`:

- Current song stops immediately.
- Avatar returns to the default pose.
- No dance is active.
- Status reads `Song changed. Press Start.`

Expected after `Start`:

- `audio1` starts cleanly.
- Avatar begins dancing only after `Start`.
- Slow-song motion feels more restrained than `audio`.

Fail if:

- Switching songs auto-starts playback.
- Avatar keeps the previous dance pose after song switch.
- `audio1` stays completely still after `Start`.

## 5. Rhythm / Choreography Experience

### audio

Expected:

- Strong beats cause more noticeable motion changes.
- Motion speed varies without visible stutter.
- Choreography changes feel energetic but not fragmented.

### audio1

Expected:

- Motion changes are less frequent.
- Overall movement feels calmer and slower than `audio`.
- High-energy clips appear less often.
- Retiming feels continuous rather than twitchy.

Fail if:

- Music and movement feel unrelated.
- `audio1` looks as aggressive or as fast as `audio`.
- Motion repeatedly locks on one pose or one clip.

## 6. End-of-Song Experience

Steps:

1. Let the selected song play to the end.

Expected:

- Music stops at the end.
- Avatar returns to the default pose.
- State becomes stopped/idle-like, not auto-looping.
- Pressing `Start` restarts the song from the beginning.

Fail if:

- Song auto-restarts.
- Avatar remains frozen in the last dance frame.
- `Start` after completion does not restart the song.

## 7. Visual Stability Checks

Run throughout the session:

- No severe mesh clipping into the floor.
- No falling through the plane.
- No disappearing avatar.
- No camera jumps to invalid angles.
- No debug UI overlap that blocks the avatar torso or legs.

## Evidence To Capture When Failing

- One screenshot with the left debug panel visible.
- One screenshot with the compact card visible.
- Last 150 lines of:

```bash
tail -n 150 ~/Library/Logs/DefaultCompany/3d-digital-human/Player.log
```
