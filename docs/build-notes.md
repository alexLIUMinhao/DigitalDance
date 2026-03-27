# Build Notes

## Primary Target

- Primary milestone target: macOS standalone build from Unity 2022.3 LTS
- Scene list: `Assets/MainScene.unity`
- Build menu: `Tools > Dance Demo > Build macOS Standalone`
- Output path: `Builds/macOS/DanceDemo.app`

## Required Local Assets Before Milestone Sign-off

- 1 manually placed Chinese ancient-style song under `Assets/Resources/Dance/Audio`
- 10 or more manually placed Mixamo dance FBX clips under `Assets/Resources/Dance/Clips`
- regenerated `Assets/StreamingAssets/DanceData/song_analysis.json`
- regenerated `Assets/StreamingAssets/DanceData/motion_manifest.json`

## Runtime Expectations

- `willa.vrm` remains the only avatar in the first milestone
- root motion stays disabled at runtime
- beat and segment timing comes from `song_analysis.json`, not from runtime audio analysis
- animation selection remains data-driven and independent from hardcoded Animator state graphs

## WebGL Caveats

WebGL is intentionally not a primary target in this milestone because:

- `StreamingAssets` access differs from desktop and can require web request-based loading
- VRM + animation library payload size is likely too large for a smooth demo build
- audio decode behavior and latency differ enough to make beat timing less predictable
- imported FBX animation access through `Resources` is acceptable for desktop prototyping, but not ideal for a browser delivery target

Treat WebGL as a future demo-only experiment after the desktop build is stable.
