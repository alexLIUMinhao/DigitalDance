# Motion Base Review SOP

This SOP defines the manual quality gate for candidate FBX clips before they enter the formal motion library.

## Required Pass Criteria

Approve only if all of the following are true:

- `loopSeam != manual_fix_needed`
- `footStability != unstable`
- `styleClarity == clear`
- `reviewStatus == approved`

If any item fails:

- use `hold` when the candidate is salvageable with better slicing, metadata fixes, or provider re-export
- use `rejected` when the source or extracted motion is not worth carrying forward

## Review Procedure

1. Open `Tools > Motion Base > Studio` in Unity and switch to `Candidate Review`.
2. Check the source video first and confirm the intended dance style and phrase intent.
3. Watch the avatar preview at `1.0x` in the full-body camera.
4. Loop the seam at least three times.
5. Switch to the feet camera and inspect foot planting at `0.9x`.
6. Check the upper-body camera to make sure the style signature survives the extraction.
7. Compare `0.9x` and `1.1x` to rate `tempoTolerance`.
8. Fill in:
   - `decisionStatus`
   - `reviewStatus`
   - `loopSeam`
   - `footStability`
   - `styleClarity`
   - `tempoTolerance`
   - `qualityTier`
   - `licenseTier`
   - `issues`
   - `reviewer`
9. Save the review before moving to the next candidate.

## Rating Guidance

- `loopSeam`
  - `clean`: loop is not visibly disruptive
  - `accent_only`: acceptable only for accent phrases
  - `manual_fix_needed`: visible break or reset
- `footStability`
  - `stable`: feet read as planted when expected
  - `monitor`: acceptable but not fully clean
  - `unstable`: obvious skating or float
- `styleClarity`
  - `clear`: style reads correctly without explanation
  - `mixed`: style intent is diluted
  - `unclear`: cannot trust the clip as a style exemplar
- `tempoTolerance`
  - `narrow`: only one close tempo works
  - `medium`: small retiming range works
  - `wide`: broad retiming range still reads well
