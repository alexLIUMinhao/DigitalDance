# Contracts

These schemas define the first stable handoff files exported by `music-motion-lab`.

- `song_event_map.schema.json`
- `motion_unit_library.schema.json`
- `choreography_plan.schema.json`
- `smplx_stitch_preview.schema.json`
- `retarget_report.schema.json`
- `review_bundle.schema.json`

Contract rules:

- outputs are JSON files written under `outputs/`
- external roots are read-only inputs
- Unity is a reference consumer, not a source-code dependency

The `examples/` directory contains minimal example payloads for each contract.
