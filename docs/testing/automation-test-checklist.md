# Automation Test Checklist

This file defines the automated coverage target for the current desktop demo.

## 1. Data Validation

Run before build or manual QA:

```bash
python3 "/Users/alex/Desktop/codex project/3d-digital/3d-digital-human/tools/dance_pipeline/validate_runtime_inputs.py" --song-id audio_mp3
python3 "/Users/alex/Desktop/codex project/3d-digital/3d-digital-human/tools/dance_pipeline/validate_runtime_inputs.py" --song-id audio1_mp3
```

Must validate:

- `song_catalog.json` contains selectable songs
- analysis file exists and parses
- beats are monotonic
- beat windows have positive duration
- `audio1` resolves to the manual slow-song beat grid and stays in `gentle` mode
- motion manifest contains enough `low_energy` clips

## 2. EditMode Coverage

Primary file:

- `/Users/alex/Desktop/codex project/3d-digital/3d-digital-human/Assets/Editor/DancePlaybackProfileTests.cs`

Current coverage target:

- playback profile default resolution
- song catalog fallback defaults
- `audio1` profile correctness
- motion manifest low-energy coverage
- gentle-mode choreography preferring low-energy loop candidates
- song analysis monotonic beat data and positive beat windows

## 3. PlayMode Coverage

Primary file:

- `/Users/alex/Desktop/codex project/3d-digital/3d-digital-human/Assets/Tests/PlayMode/DanceDemoPlaybackFlowTests.cs`

Coverage target:

- startup enters `idle`
- startup does not auto-play
- song switch returns to `idle`
- `Start` schedules an initial clip
- `Pause` freezes clip time
- `Start` after pause resumes clip time

## 4. Batch Validation

If the main project is open in Unity, run compile/build against a temporary copy.

Compile:

```bash
/Applications/Unity/Unity.app/Contents/MacOS/Unity -batchmode -quit -projectPath /tmp/3d-digital-human-debug.QfgLS2 -logFile /tmp/dance_unity_compile_copy.log
```

Build:

```bash
/Applications/Unity/Unity.app/Contents/MacOS/Unity -batchmode -quit -projectPath /tmp/3d-digital-human-debug.QfgLS2 -executeMethod DanceDemo.Editor.DanceBuildUtility.PerformBuildFromCommandLine -logFile /tmp/dance_unity_build_copy.log
```

## 5. Automation Exit Criteria

Automation is considered acceptable when:

- data validation passes for all selectable songs
- EditMode tests compile and pass
- PlayMode tests compile and pass
- batch compile succeeds
- macOS build succeeds

## 6. Known Gaps To Track

- Automated tests do not replace human judgement for music feel
- `audio1` may still need manual beat/segment refresh if the perceived pulse changes
- camera composition still needs human approval as a product-facing experience check
