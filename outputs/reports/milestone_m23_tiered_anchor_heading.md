# M23 Tiered Strong-Beat Lock, Slow-Tempo Sparsity, And Heading-Stable Retrieval

## Goal

M23 takes **M22-1** (`no054 + no066 + middle-nosynth`) as the new baseline and keeps the same any-song streaming contract:

- no song-specific tuning
- no future cheating beyond the streaming-visible window
- no middle synthetic bridge
- same FineDance rhythmic-first motion-library route

This milestone targets three user-visible problems:

1. Important strong beats should land more clearly instead of trying to hit every small beat.
2. Slow or smoother songs should use longer, less hurried motion units and tighter retime limits.
3. Retrieval should stay visually calmer, with fewer abrupt heading changes and fewer hard transitions.

## Retrieval Strategy

### Tiered Rhythm Anchors

M23 adds `anchor_tier` and scores visible anchors differently:

- `downbeat`
- `kick/snare strong hit`
- `phrase accent`
- `generic beat`

Only strong anchors are treated as hard rhythm constraints. Generic beats can be relaxed when forcing them would create jitter or over-retimed motion. This lets the planner preserve groove on weak beats while still making the important musical hits read clearly.

New M23 review/report metrics:

- `strong_anchor_error_frames`
- `tiered_anchor_hit_rate`
- `weak_anchor_relax_count`

### Tempo Mode

M23 adds adaptive `tempo_mode`:

- `slow_balanced`
- `normal`
- `high_drive`

`slow_balanced` prefers `4/8/16 beat` units, lower switch frequency, and stricter local retime limits (`0.92-1.05` preferred). This reduces the rushed or twitchy feeling that appears when slow songs are matched with too many short units.

### Two-Stage Filtering And Ranking

M23 retrieval now behaves as:

1. Use music intent + tempo mode + style/energy context to shrink the cohort.
2. Hard-filter candidates that fail:
   - strong-anchor lock
   - local anchor retime feasibility
   - heading stability / large yaw jumps
   - high transition risk
3. Rank only the surviving candidates by:
   - strong-anchor lock
   - heading continuity
   - transition smoothness
   - style fit
   - diversity

### Heading Stability

M23 adds `heading_state` and a phrase-level `turn_budget`:

- retrieval prefers same-direction continuation
- large yaw deltas are rejected unless the phrase/high-energy state justifies them
- heading flips are counted explicitly
- repeated large turns inside one phrase are suppressed

This pushes stability up into the planner instead of trying to hide everything later with mesh smoothing.

### Local Anchor Speed Feasibility

M23 keeps M21/M22 conservative segment-level retime, but also checks local speed between motion accent anchors and target strong-beat anchors. If a candidate would need excessive local squeeze/stretch to hit the important beats, it is rejected even when the whole-segment `speed_scale` still looks acceptable.

### Middle-NoSynthetic Fallback

M23 keeps the M22-1 constraint that the middle of the song should not fall back to `__synthetic__`. When no candidate passes, the recovery path prefers:

- continuing the current groove family
- longer low-risk units
- same-heading / same-energy / lower-stress motions

## Generated Artifacts

- `outputs/streaming/unity_audio[0-4]_m23_strongbeat_heading_stream_plan.jsonl`
- `outputs/reports/unity_audio[0-4]_m23_strongbeat_heading_planner_eval.json`
- `outputs/renders/unity_audio[0-4]_m23_strongbeat_heading_mesh_review.html`
- `outputs/renders/unity_audio[0-4]_m23_strongbeat_heading_mesh_preview.mp4`
- `outputs/renders/unity_audio[0-4]_m23_strongbeat_heading_mesh_strip.png`
- `outputs/reports/unity_audio[0-4]_m23_strongbeat_heading_mesh_report.json`

## Batch Metrics

| song | gap | future violation | strong anchor error | tiered anchor hit | heading flips | local retime p95 | max speed | max vertex delta | cross-sequence transitions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| audio0 | 0 | 0 | 1.4016 | 0.43912 | 10 | 0.66163 | 1.08333 | 0.052 | 51 |
| audio1 | 0 | 0 | 1.4763 | 0.47224 | 4 | 1.00000 | 1.15000 | 0.052 | 65 |
| audio2 | 0 | 0 | 1.5624 | 0.47394 | 4 | 0.55043 | 1.10000 | 0.052 | 47 |
| audio3 | 0 | 0 | 1.7631 | 0.53119 | 2 | 0.55000 | 1.11267 | 0.052 | 52 |
| audio4 | 0 | 0 | 1.4727 | 0.50591 | 0 | 0.55000 | 1.08451 | 0.052 | 38 |

## Comparison Against M22-1

The strongest structural gain is continuity pressure:

- cross-sequence transitions drop on every Unity song in this batch
- non-tail max speed is lower or equal on `audio0`, `audio2`, `audio3`, and `audio4`
- `audio4` reaches `heading_flip_count=0`
- all songs still keep `gap_count=0`, `future_visibility_violations=0`, and no `054/066` source usage

Selected metric comparison:

| song | M22-1 lock error | M23 strong-anchor error | M22-1 cross-seq | M23 cross-seq | M22-1 max speed | M23 max speed |
|---|---:|---:|---:|---:|---:|---:|
| audio0 | 1.4577 | 1.4016 | 77 | 51 | 1.13333 | 1.08333 |
| audio1 | 1.4484 | 1.4763 | 78 | 65 | 1.10000 | 1.15000 |
| audio2 | 1.5624 | 1.5624 | 72 | 47 | 1.13333 | 1.10000 |
| audio3 | 1.6404 | 1.7631 | 96 | 52 | 1.12354 | 1.11267 |
| audio4 | 1.6443 | 1.4727 | 70 | 38 | 1.12676 | 1.08451 |

This is a real trade:

- M23 is clearly calmer and more continuity-aware.
- `audio4` benefits the most.
- `audio1` and `audio3` still show that stronger continuity pressure alone does not guarantee the best strong-beat landing on every song.

## Validation Notes

- The M23 library input still excludes `054` and `066`.
- The generated M23 plans for `audio0-audio4` do not use `054` or `066`.
- The generated M23 plans also contain no middle-song synthetic motion segments.
- Review HTML now exposes tempo mode, strong-anchor score, heading continuity, and heading-flip flags so subjective review can focus on the exact M23 changes.

## Verification

- `env PYTHONPATH=music-motion-lab/src python3 -m unittest discover -s music-motion-lab/tests`
- Rendered real SMPL-X mesh review HTML/MP4/strip/report for `audio0-audio4` using the official-baseline Python environment.

## Remaining Work

M23 is a cleaner planner than M22-1, but not the final answer yet:

- `audio1` still has high local retime stress (`p95=1.0`), which means strong-beat alignment is still forcing some uncomfortable local speed changes.
- `audio3` reduces heading churn a lot, but its strong-anchor error is still not where we want it.
- The next milestone should likely split rhythm policy by song intent more aggressively:
  - stronger downbeat/kick emphasis for `drum_lock`
  - even sparser long-form phrasing for `smooth_flow`
- After that, the best next visual step is still pose/contact-aware transition smoothing so fewer transitions read as hard joins even when retrieval is reasonable.
