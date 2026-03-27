# Functional Regression Matrix

Use this as the stable regression table after any change to playback, song switching, choreography, timing, or resource loading.

| Area | Case ID | Scenario | Expected Result |
| --- | --- | --- | --- |
| Startup | F-001 | Launch app | State is `idle`, avatar default pose, no music, no scheduled clip |
| Startup | F-002 | Launch app with left panel hidden | `Tab` toggles debug panel without affecting playback state |
| Playback | F-010 | `idle -> Start` | Playback enters `playing`, first clip is scheduled, clip time grows |
| Playback | F-011 | `playing -> Pause` | Music and avatar freeze together |
| Playback | F-012 | `paused -> Start` | Playback resumes from paused point |
| Playback | F-013 | `stopped_at_end -> Start` | Song restarts from beginning |
| Song select | F-020 | Switch songs while `idle` | Selected song changes, state remains `idle`, no auto-play |
| Song select | F-021 | Switch songs while `paused` | Old song is cleared, avatar resets to default pose, state returns to `idle` |
| Song select | F-022 | Switch songs while `playing` | Old song stops immediately, avatar resets, no auto-play after switch |
| Song select | F-023 | Press `Start` immediately after switching songs | Start is queued or disabled safely; no swallowed input, no broken state |
| UI | F-030 | Busy reload state | `Start`, `Pause`, and song selectors are disabled while reloading |
| UI | F-031 | Compact card visibility | Compact card stays visible and readable without blocking the avatar |
| Logging | F-040 | Successful `Start` | Log sequence includes `UI start requested`, `Queued clip`, `Scheduled initial variant`, `Configured dance slot` |
| Logging | F-041 | Song switch | Log includes selected song id and audio clip name |
| Rhythm | F-050 | `audio` playback | Strong beats produce more active clip changes and speed shifts |
| Rhythm | F-051 | `audio1` playback | `Dance BPM` follows the manual slow-song beat grid and switching is calmer than `audio` |
| Rhythm | F-052 | Pause on `audio1` | `Current Clip Time` stops and resumes correctly |
| Choreography | F-060 | Strong beat clip change | Clip changes do not produce obvious hard-cuts every few beats |
| Choreography | F-061 | Slow-song behavior | High-energy clips are less frequent on `audio1` |
| Choreography | F-062 | Variant diversity | Same clip/variant should not dominate long playback windows |
| Avatar stability | F-070 | Normal playback | Avatar does not sink, float away, or disappear |
| Avatar stability | F-071 | Long playback | No frozen pose, no dropped plane collision, no invalid camera framing |
| Data | F-080 | Valid song catalog | Every selectable song resolves to analysis JSON + audio resource |
| Data | F-081 | Missing analysis file | Invalid song is excluded or rejected safely |
| Data | F-082 | Motion manifest coverage | Runtime has valid Humanoid clips and expected energy-band distribution |
| Build | F-090 | Batch compile | Unity batch compile succeeds |
| Build | F-091 | macOS build | Desktop build succeeds and launches |
| Build | F-092 | Launch built app | Built app opens without crash or black screen |

## Required Regression Bundles

### Bundle A: Playback State

Run after changes to:

- `DanceDemoBootstrap`
- `MusicConductor`
- `CharacterDanceController`
- UI play/pause handlers

Cases:

- `F-001`, `F-010`, `F-011`, `F-012`, `F-013`, `F-040`

### Bundle B: Song Switching

Run after changes to:

- song catalog
- song selection UI
- reload flow
- runtime data loading

Cases:

- `F-020`, `F-021`, `F-022`, `F-023`, `F-031`, `F-041`, `F-080`, `F-081`

### Bundle C: Rhythm / Choreography

Run after changes to:

- `MusicConductor`
- `ChoreographyEngine`
- song analysis JSON
- motion manifest

Cases:

- `F-050`, `F-051`, `F-052`, `F-060`, `F-061`, `F-062`, `F-082`

### Bundle D: Stability / Delivery

Run after changes to:

- avatar loading
- clip import settings
- build pipeline
- camera / floor handling

Cases:

- `F-070`, `F-071`, `F-090`, `F-091`, `F-092`
