# M18 Hybrid State Machine, Conservative Retime, And Outro Recover

## Goal

M18 upgrades the M17 `repeat12_total5_center` baseline with a generic any-song strategy:

- keep the first `5s` as synthetic neutral upright idle;
- use a streaming-safe hybrid choreography state machine for energy/accent/transition intent;
- retime conservatively against the visible beat/drum rail without letting non-tail dance speed exceed `1.15`;
- stop retrieval before the song end and gradually recover to synthetic neutral idle for the final window.

This remains FineDance SMPL-X only. Willa is not used.

## Implementation Summary

- Added `planner-version=m18` with default `initial_hold_sec=5`, `ending_hold_sec=5`, `ending_policy=gradual_recover`, `speed_retime_policy=conservative_lock`, and `state_machine_policy=hybrid`.
- Added streaming `music_state` fields to ticks: energy level, accent density, beat confidence, phrase phase, and melodic-motion proxy.
- Added M18 choreography states: `intro_idle`, `groove_low`, `groove_mid`, `groove_high`, `accent_prepare`, `accent_hit`, `transition`, and `recover_outro`.
- Added conservative retime spacing correction so over-dense automatic beat estimates do not force twitch-fast motion.
- Added synthetic `smplx_neutral_recover` plus final `smplx_neutral_idle` hold.
- Added review HTML state/retime columns and state-colored timeline bands.

## Generated Reviews

| Song | Review HTML | Gap | Future | Outro recover | Final pose | Max run | Max total uses | Max non-tail speed | High-conf lock err | Max temporal vertex delta |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| audio0 | `outputs/renders/unity_audio0_m18_state_retime_outro_mesh_review.html` | 0 | 0 | 11.81997s | `smplx_neutral_idle` | 2 | 5 | 1.09157 | 0.84 frames | 0.040092 |
| audio1 | `outputs/renders/unity_audio1_m18_state_retime_outro_mesh_review.html` | 0 | 0 | 5.00000s | `smplx_neutral_idle` | 2 | 5 | 1.06666 | 0.8001 frames | 0.045980 |
| audio2 | `outputs/renders/unity_audio2_m18_state_retime_outro_mesh_review.html` | 0 | 0 | 7.33997s | `smplx_neutral_idle` | 2 | 4 | 1.09157 | 0.48 frames | 0.045042 |
| audio3 | `outputs/renders/unity_audio3_m18_state_retime_outro_mesh_review.html` | 0 | 0 | 5.92522s | `smplx_neutral_idle` | 2 | 5 | 1.09157 | 0.8214 frames | 0.044800 |
| audio4 | `outputs/renders/unity_audio4_m18_state_retime_outro_mesh_review.html` | 0 | 0 | 10.21401s | `smplx_neutral_idle` | 2 | 5 | 1.09157 | 0.7893 frames | 0.045896 |

## Verification

- `env PYTHONPATH=music-motion-lab/src python3 -m unittest discover -s music-motion-lab/tests`
- Rendered M18 mesh reviews using `motion-base-assets/tools/venvs/official-baseline`.
- Reopened latest review page: `outputs/renders/unity_audio3_m18_state_retime_outro_mesh_review.html`.

## Notes

- Some outro windows are longer than exactly `5s` because M18 refuses to squeeze a short final dance segment into the area before recovery. This avoids the previous fast/twitchy tail behavior.
- The retime correction is generic: it reacts to over-dense streaming beat estimates by selecting a more natural segment duration instead of song-specific tempo tuning.
